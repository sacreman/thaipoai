// Command gateway is the translation API. It renders the prompt, calls the model, runs the
// guards and repairs once. It never logs a request or a translation.
package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	mathrand "math/rand/v2"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/sacreman/thaipoai/internal/guards"
	"github.com/sacreman/thaipoai/internal/llm"
	"github.com/sacreman/thaipoai/internal/rollout"
)

const (
	maxBody         = 64 * 1024
	maxTextRunes    = 4000
	maxContextTurns = 8
)

type turn struct {
	Speaker string `json:"speaker"`
	Text    string `json:"text"`
}

type person struct {
	Gender string `json:"gender"`
}

type translateRequest struct {
	SourceLang string `json:"source_lang"`
	TargetLang string `json:"target_lang"`
	Text       string `json:"text"`
	Context    []turn `json:"context"`
	Speaker    person `json:"speaker"`
	Addressee  person `json:"addressee"`
}

type translateResponse struct {
	Translation      string   `json:"translation"`
	Status           string   `json:"status"` // ok, repaired or failed
	FirstTryFailures []string `json:"first_try_failures,omitempty"`
	Failures         []string `json:"failures,omitempty"`
	Model            string   `json:"model"`
	RequestID        string   `json:"request_id"`
	Millis           int64    `json:"millis"`
}

type chatFunc func(ctx context.Context, model string, messages []llm.Message) (string, error)

type server struct {
	chat    chatFunc
	route   atomic.Pointer[rollout.Routing]
	percent func() int // a random number from 0 to 99
	timeout time.Duration
	metrics *metrics
}

func (s *server) setRouting(route rollout.Routing) {
	s.route.Store(&route)
	s.metrics.scope(route.ID)
}

func validate(r translateRequest) error {
	pair := r.SourceLang + ">" + r.TargetLang
	if pair != "en>th" && pair != "th>en" {
		return errors.New("source_lang and target_lang must be en and th")
	}
	if strings.TrimSpace(r.Text) == "" {
		return errors.New("text is required")
	}
	if utf8.RuneCountInString(r.Text) > maxTextRunes {
		return errors.New("text is too long")
	}
	if len(r.Context) > maxContextTurns {
		return errors.New("too many context turns")
	}
	for _, t := range r.Context {
		if utf8.RuneCountInString(t.Text) > maxTextRunes {
			return errors.New("context turn is too long")
		}
	}
	for _, gender := range []string{r.Speaker.Gender, r.Addressee.Gender} {
		if gender != "" && gender != "male" && gender != "female" {
			return errors.New("gender must be male or female")
		}
	}
	return nil
}

func promptFor(r translateRequest) []llm.Message {
	target := "Thai"
	if r.TargetLang == "en" {
		target = "English"
	}
	var history []string
	for _, t := range r.Context {
		history = append(history, t.Speaker+": "+t.Text)
	}
	return llm.ChatMessages(target, r.Text, history, llm.SpeakerNotes(r.Speaker.Gender, r.Addressee.Gender))
}

func (s *server) translate(ctx context.Context, r translateRequest) (translateResponse, error) {
	check := func(output string) []string {
		return guards.Check(guards.Request{Source: r.Text, Output: output, TargetThai: r.TargetLang == "th", SpeakerGender: r.Speaker.Gender})
	}
	messages := promptFor(r)
	route := *s.route.Load()
	model := route.Stable
	if route.Candidate != "" && s.percent() < route.CandidatePercent {
		model = route.Candidate
	}
	started := time.Now()
	var first string
	var err error
	if model == route.Candidate {
		// Only half the time, so a hung candidate still leaves room to answer from stable.
		attempt, cancel := context.WithTimeout(ctx, s.timeout/2)
		first, err = s.chat(attempt, model, messages)
		cancel()
		if err != nil {
			s.metrics.infraError(route.ID, model) // never the user's problem, and never the model's fault
			model = route.Stable
			first, err = s.chat(ctx, model, messages)
		}
	} else {
		first, err = s.chat(ctx, model, messages)
	}
	if err != nil {
		return translateResponse{}, err
	}
	firstFailures := check(first)
	result := translateResponse{Translation: first, Status: "ok", FirstTryFailures: firstFailures, Failures: firstFailures, Model: model}
	s.metrics.guardFailures("first", firstFailures)
	blocking := guards.Hard(firstFailures)
	s.metrics.observe(route.ID, model, len(blocking) > 0, utf8.RuneCountInString(r.Text), utf8.RuneCountInString(first), time.Since(started))
	if len(blocking) == 0 {
		return result, nil
	}
	repair := append(messages,
		llm.Message{"role": "assistant", "content": first},
		llm.Message{"role": "user", "content": "That translation has this problem: " + strings.Join(blocking, ", ") + ". Translate the message again. Output only the translation."},
	)
	second, err := s.chat(ctx, model, repair)
	if err != nil {
		result.Status = "failed"
		return result, nil
	}
	secondFailures := check(second)
	s.metrics.guardFailures("repair", secondFailures)
	if len(guards.Hard(secondFailures)) > 0 {
		result.Status = "failed"
		return result, nil
	}
	result.Translation, result.Failures, result.Status = second, secondFailures, "repaired"
	return result, nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	encoder := json.NewEncoder(w)
	encoder.SetEscapeHTML(false)
	encoder.Encode(value)
}

func requestID() string {
	buffer := make([]byte, 8)
	rand.Read(buffer)
	return hex.EncodeToString(buffer)
}

func (s *server) handleTranslate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
		return
	}
	var request translateRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBody)).Decode(&request); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON or request too large"})
		return
	}
	if err := validate(request); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), s.timeout)
	defer cancel()
	started := time.Now()
	result, err := s.translate(ctx, request)
	elapsed := time.Since(started)
	if err != nil {
		s.metrics.finish("error", elapsed)
		log.Printf("status=error millis=%d", elapsed.Milliseconds())
		writeJSON(w, http.StatusBadGateway, map[string]string{"error": "model unavailable"})
		return
	}
	result.RequestID, result.Millis = requestID(), elapsed.Milliseconds()
	s.metrics.finish(result.Status, elapsed)
	log.Printf("status=%s millis=%d", result.Status, result.Millis)
	writeJSON(w, http.StatusOK, result)
}

type metrics struct {
	mu        sync.Mutex
	statuses  map[string]int
	guards    map[string]int
	millis    int64
	rolloutID string
	models    map[string]*rollout.Stats
}

func newMetrics() *metrics {
	return &metrics{statuses: map[string]int{}, guards: map[string]int{}, models: map[string]*rollout.Stats{}}
}

// scope starts fresh per-model counts for a new rollout.
func (m *metrics) scope(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if id != m.rolloutID {
		m.rolloutID, m.models = id, map[string]*rollout.Stats{}
	}
}

func (m *metrics) model(name string) *rollout.Stats {
	if m.models[name] == nil {
		m.models[name] = &rollout.Stats{}
	}
	return m.models[name]
}

func (m *metrics) infraError(id, name string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if id == m.rolloutID {
		m.model(name).InfraErrors++
	}
}

func (m *metrics) observe(id, name string, failed bool, source, output int, elapsed time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if id != m.rolloutID {
		return // routing changed while this request was running
	}
	stats := m.model(name)
	stats.Requests++
	if failed {
		stats.GuardFailures++
	}
	stats.SourceChars += int64(source)
	stats.OutputChars += int64(output)
	stats.Millis += elapsed.Milliseconds()
}

func (m *metrics) stats() map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	models := make(map[string]rollout.Stats, len(m.models))
	for name, stats := range m.models {
		models[name] = *stats // a copy, because the caller encodes it outside the lock
	}
	return map[string]any{"rollout_id": m.rolloutID, "models": models}
}

func (m *metrics) guardFailures(stage string, failed []string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, name := range failed {
		m.guards[stage+"\x00"+name]++
	}
}

func (m *metrics) finish(status string, elapsed time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.statuses[status]++
	m.millis += elapsed.Milliseconds()
}

func (m *metrics) render() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var b strings.Builder
	total := 0
	for _, key := range sortedKeys(m.statuses) {
		fmt.Fprintf(&b, "thaipoai_requests_total{status=%q} %d\n", key, m.statuses[key])
		total += m.statuses[key]
	}
	for _, key := range sortedKeys(m.guards) {
		stage, name, _ := strings.Cut(key, "\x00")
		fmt.Fprintf(&b, "thaipoai_guard_failures_total{stage=%q,guard=%q} %d\n", stage, name, m.guards[key])
	}
	for name, stats := range m.models {
		fmt.Fprintf(&b, "thaipoai_model_requests_total{model=%q} %d\n", name, stats.Requests)
		fmt.Fprintf(&b, "thaipoai_model_guard_failures_total{model=%q} %d\n", name, stats.GuardFailures)
		fmt.Fprintf(&b, "thaipoai_model_infra_errors_total{model=%q} %d\n", name, stats.InfraErrors)
	}
	fmt.Fprintf(&b, "thaipoai_request_duration_milliseconds_sum %d\n", m.millis)
	fmt.Fprintf(&b, "thaipoai_request_duration_milliseconds_count %d\n", total)
	return b.String()
}

func sortedKeys(values map[string]int) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func (s *server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/translate", s.handleTranslate)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, _ *http.Request) { fmt.Fprint(w, s.metrics.render()) })
	mux.HandleFunc("/stats", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, http.StatusOK, s.metrics.stats()) })
	return mux
}

// followRouting applies the routing file when it changes. A missing or broken file changes
// nothing, so the gateway keeps its last weights if the controller is gone.
func (s *server) followRouting(file string) {
	raw, err := os.ReadFile(file)
	if err != nil {
		return
	}
	var route rollout.Routing
	if err := json.Unmarshal(raw, &route); err != nil || route.Stable == "" {
		log.Print("routing file is invalid, keeping the last routing")
		return
	}
	if current := s.route.Load(); current == nil || *current != route {
		s.setRouting(route)
		log.Printf("routing stable=%s candidate=%s percent=%d", route.Stable, route.Candidate, route.CandidatePercent)
	}
}

func main() {
	listen := flag.String("listen", "127.0.0.1:8088", "address to serve on")
	backend := flag.String("backend", "http://127.0.0.1:8766", "OpenAI-compatible model server")
	model := flag.String("model", "base", "model name used until a routing file exists")
	routingFile := flag.String("routing", "", "file naming the stable and candidate models")
	timeout := flag.Duration("timeout", 60*time.Second, "time allowed for one translation")
	flag.Parse()
	httpClient := &http.Client{Timeout: 2 * *timeout}
	chat := func(ctx context.Context, name string, messages []llm.Message) (string, error) {
		return llm.Client{Endpoint: strings.TrimRight(*backend, "/"), Model: name, HTTP: httpClient}.Chat(ctx, messages)
	}
	s := &server{chat: chat, percent: func() int { return mathrand.IntN(100) }, timeout: *timeout, metrics: newMetrics()}
	s.setRouting(rollout.Routing{Stable: *model})
	if *routingFile != "" {
		s.followRouting(*routingFile)
		go func() {
			for range time.Tick(2 * time.Second) {
				s.followRouting(*routingFile)
			}
		}()
	}
	log.Printf("listening on %s", *listen)
	log.Fatal(http.ListenAndServe(*listen, s.routes()))
}
