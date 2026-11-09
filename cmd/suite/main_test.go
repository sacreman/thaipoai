package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestAnUnknownFormatIsRefused(t *testing.T) {
	if _, err := messagesFor("bad", "Thai", "x", nil); err == nil {
		t.Fatal("unknown format accepted")
	}
}

func TestVariantsChooseTheContext(t *testing.T) {
	c := Case{Context: []string{"real"}, IrrelevantContext: []string{"other"}}
	for variant, want := range map[string][]string{"none": nil, "correct": {"real"}, "irrelevant": {"other"}} {
		got, err := contextFor(c, variant)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Fatalf("%s: got %v, %v", variant, got, err)
		}
	}
	if _, err := contextFor(c, "bad"); err == nil {
		t.Fatal("unknown variant accepted")
	}
}

func TestLiteralTrapAndEmptyAreFailures(t *testing.T) {
	c := Case{LiteralTrap: "kick the bucket"}
	if got := checks(c, "he will kick the bucket"); !reflect.DeepEqual(got, []string{"literal_trap"}) {
		t.Fatalf("got %v", got)
	}
	if got := checks(c, "  "); !reflect.DeepEqual(got, []string{"empty"}) {
		t.Fatalf("got %v", got)
	}
	if got := checks(c, "he will die"); got != nil {
		t.Fatalf("got %v", got)
	}
}

func run(t *testing.T, handler http.HandlerFunc, suite string, variant string) []Result {
	t.Helper()
	server := httptest.NewServer(handler)
	defer server.Close()
	cfg := config{endpoint: server.URL, model: "m", system: "s", variant: variant, format: "chat", client: &http.Client{Timeout: 5 * time.Second}}
	var out bytes.Buffer
	if _, err := runSuite(cfg, strings.NewReader(suite), &out); err != nil {
		t.Fatal(err)
	}
	var results []Result
	decoder := json.NewDecoder(&out)
	for decoder.More() {
		var r Result
		if err := decoder.Decode(&r); err != nil {
			t.Fatal(err)
		}
		results = append(results, r)
	}
	return results
}

func TestSendsThePromptAndRecordsTheOutput(t *testing.T) {
	var system, prompt string
	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		var request struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		json.Unmarshal(body, &request)
		system, prompt = request.Messages[0].Content, request.Messages[1].Content
		w.Write([]byte(`{"choices":[{"message":{"content":" สวัสดี "}}]}`))
	}
	results := run(t, handler, `{"id":"a","direction":"en-th","category":"c","source":"hello","context":["A: hi"]}`+"\n", "correct")
	if len(results) != 1 || results[0].Output != "สวัสดี" || results[0].Variant != "correct" {
		t.Fatalf("got %+v", results)
	}
	if !strings.Contains(system, "A: hi") || !strings.HasSuffix(prompt, "hello") {
		t.Fatalf("system %q prompt %q", system, prompt)
	}
}

func TestDeepSeekModeSendsAuthAndTurnsThinkingOff(t *testing.T) {
	var auth string
	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &request)
		w.Write([]byte(`{"choices":[{"message":{"content":"x"}}]}`))
	}))
	defer server.Close()
	cfg := config{endpoint: server.URL, model: "m", system: "s", variant: "correct", format: "chat", apiKey: "k",
		temperature: 0.7, noThinking: true, client: &http.Client{Timeout: 5 * time.Second}}
	result := cfg.run(Case{ID: "a", Direction: "en-th", Source: "hello", Context: []string{"A: hi"}})
	if result.Error != "" || auth != "Bearer k" || request["temperature"] != 0.7 || request["reasoning_effort"] != "none" {
		t.Fatalf("result %+v auth %q request %v", result, auth, request)
	}
	messages := request["messages"].([]any)
	system := messages[0].(map[string]any)["content"].(string)
	user := messages[1].(map[string]any)["content"].(string)
	if !strings.Contains(system, "A: hi") || strings.Contains(user, "A: hi") || !strings.HasSuffix(user, "hello") {
		t.Fatalf("messages %v", messages)
	}
}

func TestFailedRequestsStayInTheResults(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(500) }
	results := run(t, handler, `{"id":"a","direction":"th-en","source":"x"}`+"\n"+`{"id":"b","direction":"nope","source":"y"}`+"\n", "none")
	if len(results) != 2 || results[0].Error != "status 500" || results[1].Error == "" {
		t.Fatalf("got %+v", results)
	}
}
