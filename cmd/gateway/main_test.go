package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/sacreman/thaipoai/internal/llm"
	"github.com/sacreman/thaipoai/internal/rollout"
)

func start(t *testing.T, replies ...string) (*httptest.Server, *[][]llm.Message, *server) {
	t.Helper()
	calls := &[][]llm.Message{}
	s := &server{timeout: time.Second, percent: func() int { return 99 }, metrics: newMetrics()}
	s.setRouting(rollout.Routing{ID: "r1", Stable: "test"})
	s.chat = func(_ context.Context, _ string, messages []llm.Message) (string, error) {
		*calls = append(*calls, messages)
		if len(replies) == 0 {
			return "", errors.New("no reply")
		}
		reply := replies[0]
		replies = replies[1:]
		return reply, nil
	}
	ts := httptest.NewServer(s.routes())
	t.Cleanup(ts.Close)
	return ts, calls, s
}

func post(t *testing.T, ts *httptest.Server, body string) (int, translateResponse, string) {
	t.Helper()
	resp, err := http.Post(ts.URL+"/v1/translate", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var buf bytes.Buffer
	buf.ReadFrom(resp.Body)
	var out translateResponse
	json.Unmarshal(buf.Bytes(), &out)
	return resp.StatusCode, out, buf.String()
}

const toThai = `{"source_lang":"en","target_lang":"th","text":"It costs 450 baht."}`

func TestAGoodTranslationPassesFirstTime(t *testing.T) {
	ts, calls, _ := start(t, "ราคา 450 บาท")
	code, out, _ := post(t, ts, toThai)
	if code != 200 || out.Status != "ok" || out.Translation != "ราคา 450 บาท" || len(*calls) != 1 || out.RequestID == "" || out.Model != "test" {
		t.Fatalf("code %d out %+v calls %d", code, out, len(*calls))
	}
}

func TestAFailedGuardIsRepairedOnceAndReportedHonestly(t *testing.T) {
	ts, calls, _ := start(t, "ราคา 540 บาท", "ราคา 450 บาท")
	_, out, _ := post(t, ts, toThai)
	if out.Status != "repaired" || out.Translation != "ราคา 450 บาท" || len(*calls) != 2 {
		t.Fatalf("out %+v calls %d", out, len(*calls))
	}
	if len(out.FirstTryFailures) != 1 || out.FirstTryFailures[0] != "numbers" || len(out.Failures) != 0 {
		t.Fatalf("failures not reported: %+v", out)
	}
	if !strings.Contains((*calls)[1][3]["content"], "numbers") {
		t.Fatalf("repair prompt %v", (*calls)[1])
	}
}

func TestAFailedRepairKeepsTheFirstAnswerAndSaysSo(t *testing.T) {
	ts, calls, _ := start(t, "ราคา 540 บาท", "ราคา 999 บาท")
	_, out, _ := post(t, ts, toThai)
	if out.Status != "failed" || out.Translation != "ราคา 540 บาท" || len(*calls) != 2 {
		t.Fatalf("out %+v", out)
	}
}

func TestSoftGuardsAreReportedButNotRepaired(t *testing.T) {
	ts, calls, _ := start(t, "ฉันจะไป")
	_, out, _ := post(t, ts, `{"source_lang":"en","target_lang":"th","text":"I am not going."}`)
	if out.Status != "ok" || len(*calls) != 1 || len(out.Failures) != 1 || out.Failures[0] != "negation" {
		t.Fatalf("out %+v calls %d", out, len(*calls))
	}
}

func TestContextAndSpeakerReachThePrompt(t *testing.T) {
	ts, calls, _ := start(t, "สวัสดีครับ")
	post(t, ts, `{"source_lang":"en","target_lang":"th","text":"Hello","context":[{"speaker":"A","text":"hi"}],"speaker":{"gender":"male"},"addressee":{"gender":"female"}}`)
	system := (*calls)[0][0]["content"]
	if !strings.Contains(system, "A: hi") || !strings.Contains(system, "writing this message is a man") || !strings.Contains(system, "reading it is a woman") {
		t.Fatalf("system %q", system)
	}
	if strings.Contains((*calls)[0][1]["content"], "A: hi") {
		t.Fatal("history reached the user turn")
	}
}

func TestBadRequestsAreRefused(t *testing.T) {
	ts, calls, _ := start(t)
	bodies := []string{
		`not json`,
		`{"source_lang":"en","target_lang":"fr","text":"x"}`,
		`{"source_lang":"en","target_lang":"th","text":"  "}`,
		`{"source_lang":"en","target_lang":"th","text":"x","speaker":{"gender":"other"}}`,
		`{"source_lang":"en","target_lang":"th","text":"` + strings.Repeat("a", maxTextRunes+1) + `"}`,
	}
	for _, body := range bodies {
		if code, _, _ := post(t, ts, body); code != 400 {
			t.Fatalf("%.40s: code %d", body, code)
		}
	}
	if len(*calls) != 0 {
		t.Fatalf("model called %d times", len(*calls))
	}
}

func TestModelFailureIsAnHonestError(t *testing.T) {
	ts, _, _ := start(t)
	if code, _, body := post(t, ts, toThai); code != 502 || !strings.Contains(body, "unavailable") {
		t.Fatalf("code %d body %s", code, body)
	}
}

func TestNothingPrivateIsLogged(t *testing.T) {
	var logs bytes.Buffer
	log.SetOutput(&logs)
	defer log.SetOutput(os.Stderr)
	ts, _, _ := start(t, "ข้อความลับ")
	post(t, ts, `{"source_lang":"en","target_lang":"th","text":"secret message"}`)
	if strings.Contains(logs.String(), "secret") || strings.Contains(logs.String(), "ข้อความลับ") || !strings.Contains(logs.String(), "status=ok") {
		t.Fatalf("logs %q", logs.String())
	}
}

func TestMetricsCountStatusesAndGuards(t *testing.T) {
	ts, _, s := start(t, "ราคา 540 บาท", "ราคา 450 บาท")
	post(t, ts, toThai)
	got := s.metrics.render()
	for _, want := range []string{`thaipoai_requests_total{status="repaired"} 1`, `thaipoai_guard_failures_total{stage="first",guard="numbers"} 1`, `thaipoai_request_duration_milliseconds_count 1`} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in %s", want, got)
		}
	}
}

func routed(t *testing.T, percent int, chat chatFunc) (*httptest.Server, *server) {
	t.Helper()
	s := &server{timeout: time.Second, percent: func() int { return percent }, metrics: newMetrics(), chat: chat}
	s.setRouting(rollout.Routing{ID: "r1", Stable: "stable", Candidate: "cand", CandidatePercent: 10})
	ts := httptest.NewServer(s.routes())
	t.Cleanup(ts.Close)
	return ts, s
}

func TestTrafficSplitsAtTheCanaryPercentage(t *testing.T) {
	var models []string
	chat := func(_ context.Context, model string, _ []llm.Message) (string, error) {
		models = append(models, model)
		return "ราคา 450 บาท", nil
	}
	for _, c := range []struct {
		roll int
		want string
	}{{9, "cand"}, {10, "stable"}} {
		ts, _ := routed(t, c.roll, chat)
		if _, out, _ := post(t, ts, toThai); out.Model != c.want {
			t.Fatalf("roll %d served by %s", c.roll, out.Model)
		}
	}
}

func TestACandidateOutageFallsBackToStableAndIsCountedAsInfrastructure(t *testing.T) {
	chat := func(_ context.Context, model string, _ []llm.Message) (string, error) {
		if model == "cand" {
			return "", errors.New("adapter not loaded")
		}
		return "ราคา 450 บาท", nil
	}
	ts, s := routed(t, 0, chat)
	code, out, _ := post(t, ts, toThai)
	if code != 200 || out.Model != "stable" {
		t.Fatalf("code %d out %+v", code, out)
	}
	if cand := s.metrics.adapters["cand"]; cand.InfraErrors != 1 || cand.Requests != 0 {
		t.Fatalf("%+v", cand)
	}
}

func TestPerAdapterCountsRestartWithANewRollout(t *testing.T) {
	ts, s := routed(t, 99, func(context.Context, string, []llm.Message) (string, error) { return "ราคา 540 บาท", nil })
	post(t, ts, toThai)
	if stable := s.metrics.adapters["stable"]; stable.Requests != 1 || stable.GuardFailures != 1 || stable.OutputChars == 0 {
		t.Fatalf("%+v", stable)
	}
	s.setRouting(rollout.Routing{ID: "r2", Stable: "stable"})
	if len(s.metrics.adapters) != 0 {
		t.Fatalf("%+v", s.metrics.adapters)
	}
}

func TestABrokenRoutingFileKeepsTheLastRouting(t *testing.T) {
	ts, s := routed(t, 99, nil)
	_ = ts
	file := t.TempDir() + "/routing.json"
	os.WriteFile(file, []byte(`{"id":"r2","stable":"new"}`), 0o644)
	s.followRouting(file)
	if s.route.Load().Stable != "new" {
		t.Fatalf("%+v", s.route.Load())
	}
	os.WriteFile(file, []byte(`{`), 0o644)
	s.followRouting(file)
	os.Remove(file)
	s.followRouting(file)
	if s.route.Load().Stable != "new" {
		t.Fatalf("%+v", s.route.Load())
	}
}

func TestAHungCandidateStillLeavesTimeForStable(t *testing.T) {
	chat := func(ctx context.Context, model string, _ []llm.Message) (string, error) {
		if model == "cand" {
			<-ctx.Done()
			return "", ctx.Err()
		}
		if err := ctx.Err(); err != nil {
			return "", err
		}
		return "ราคา 450 บาท", nil
	}
	ts, _ := routed(t, 0, chat)
	code, out, body := post(t, ts, toThai)
	if code != 200 || out.Model != "stable" {
		t.Fatalf("code %d body %s", code, body)
	}
}
