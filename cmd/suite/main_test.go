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

func TestPromptMatchesTheTrainingFormat(t *testing.T) {
	got := lumenPrompt("Thai", "hello", []string{"A: hi", "B: yes"})
	want := "Please translate the following text into Thai.\n#guidelines\nUse this conversation only as context and do not translate it:\nA: hi\nB: yes\n#src_text\nhello"
	if got != want {
		t.Fatalf("got %q", got)
	}
	if got := lumenPrompt("English", "x", nil); got != "Please translate the following text into English.\n#guidelines\n\n#src_text\nx" {
		t.Fatalf("no context: got %q", got)
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
	cfg := config{endpoint: server.URL, model: "m", system: "s", variant: variant, client: &http.Client{Timeout: 5 * time.Second}}
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
	var prompt string
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
		prompt = request.Messages[0].Content
		w.Write([]byte(`{"choices":[{"message":{"content":" สวัสดี "}}]}`))
	}
	results := run(t, handler, `{"id":"a","direction":"en-th","category":"c","source":"hello","context":["A: hi"]}`+"\n", "correct")
	if len(results) != 1 || results[0].Output != "สวัสดี" || results[0].Variant != "correct" {
		t.Fatalf("got %+v", results)
	}
	if !strings.Contains(prompt, "A: hi") || !strings.HasSuffix(prompt, "#src_text\nhello") {
		t.Fatalf("prompt %q", prompt)
	}
}

func TestFailedRequestsStayInTheResults(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(500) }
	results := run(t, handler, `{"id":"a","direction":"th-en","source":"x"}`+"\n"+`{"id":"b","direction":"nope","source":"y"}`+"\n", "none")
	if len(results) != 2 || results[0].Error != "status 500" || results[1].Error == "" {
		t.Fatalf("got %+v", results)
	}
}
