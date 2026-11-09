// Command fake-model behaves like vLLM with deterministic output, so a multi-replica rollout can
// be tested without a GPU. An adapter's name decides how it behaves.
package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync"
)

var digits = regexp.MustCompile(`\d+`)

type server struct {
	base     string
	mu       sync.Mutex
	adapters map[string]bool
	failLoad bool
}

func particle(system, model string) string {
	man, woman := strings.Contains(system, "writing this message is a man"), strings.Contains(system, "writing this message is a woman")
	if strings.HasPrefix(model, "flipped") {
		man, woman = woman, man
	}
	if man {
		return "ครับ"
	}
	if woman {
		return "ค่ะ"
	}
	return ""
}

func translate(system, user, model string) string {
	source := user[strings.Index(user, "\n\n")+2:]
	numbers := strings.Join(digits.FindAllString(source, -1), " ")
	output := "translated " + numbers
	if strings.HasPrefix(user, "Translate into Thai") {
		output = "แปลแล้ว " + numbers + particle(system, model)
	}
	if strings.HasPrefix(model, "bloat") {
		output = strings.Repeat(output+" ", 3)
	}
	return strings.TrimSpace(output)
}

func (s *server) reply(w http.ResponseWriter, code int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(body)
}

func (s *server) models(w http.ResponseWriter, _ *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	data := []map[string]any{{"id": s.base, "parent": nil}}
	names := make([]string, 0, len(s.adapters))
	for name := range s.adapters {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		data = append(data, map[string]any{"id": name, "parent": s.base})
	}
	s.reply(w, http.StatusOK, map[string]any{"data": data})
}

func (s *server) load(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name string `json:"lora_name"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.failLoad {
		s.reply(w, http.StatusBadRequest, map[string]string{"error": "injected load failure"})
		return
	}
	s.adapters[body.Name] = true
	s.reply(w, http.StatusOK, map[string]string{"status": "loaded"})
}

func (s *server) unload(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Name string `json:"lora_name"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.adapters, body.Name)
	s.reply(w, http.StatusOK, map[string]string{"status": "unloaded"})
}

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Model    string              `json:"model"`
		Messages []map[string]string `json:"messages"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	s.mu.Lock()
	known := body.Model == s.base || s.adapters[body.Model]
	s.mu.Unlock()
	if !known {
		s.reply(w, http.StatusNotFound, map[string]string{"error": "unknown model"})
		return
	}
	var system, user string
	for _, message := range body.Messages {
		switch message["role"] {
		case "system":
			system = message["content"]
		case "user":
			if user == "" {
				user = message["content"]
			}
		}
	}
	s.reply(w, http.StatusOK, map[string]any{"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": translate(system, user, body.Model)}}}})
}

// fault turns injected load failures on or off for this replica.
func (s *server) fault(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failLoad = r.URL.Query().Get("fail_load") == "true"
	s.reply(w, http.StatusOK, map[string]bool{"fail_load": s.failLoad})
}

func main() {
	listen := flag.String("listen", ":8000", "address to serve on")
	base := flag.String("base", "base", "base model name")
	flag.Parse()
	s := &server{base: *base, adapters: map[string]bool{}}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		s.reply(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("/v1/models", s.models)
	mux.HandleFunc("/v1/load_lora_adapter", s.load)
	mux.HandleFunc("/v1/unload_lora_adapter", s.unload)
	mux.HandleFunc("/v1/chat/completions", s.chat)
	mux.HandleFunc("/fault", s.fault)
	log.Fatal(http.ListenAndServe(*listen, mux))
}
