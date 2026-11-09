// Command fake-model behaves like an OpenAI-style model server with deterministic output, so a
// multi-replica rollout can be tested without a real model. A model's name decides how it behaves.
package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"regexp"
	"strings"
	"sync"
)

var digits = regexp.MustCompile(`\d+`)

type server struct {
	base   string
	mu     sync.Mutex
	hidden string // this replica does not serve models whose names start with this
}

// serves is true for any name, because here a model is only a name. "corrupt" stands for a
// digest no replica holds, and the hide fault stands for a replica that lacks a model.
func (s *server) serves(model string) bool {
	if model == s.base {
		return true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return !strings.HasPrefix(model, "corrupt") && (s.hidden == "" || !strings.HasPrefix(model, s.hidden))
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

func (s *server) model(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !s.serves(id) {
		s.reply(w, http.StatusNotFound, map[string]string{"error": "unknown model"})
		return
	}
	s.reply(w, http.StatusOK, map[string]string{"id": id, "object": "model"})
}

func (s *server) chat(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Model    string              `json:"model"`
		Messages []map[string]string `json:"messages"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	if !s.serves(body.Model) {
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

// fault makes this replica lack models whose names start with hide. An empty value clears it.
func (s *server) fault(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.hidden = r.URL.Query().Get("hide")
	s.reply(w, http.StatusOK, map[string]string{"hide": s.hidden})
}

func main() {
	listen := flag.String("listen", ":8000", "address to serve on")
	base := flag.String("base", "base", "base model name")
	flag.Parse()
	s := &server{base: *base}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		s.reply(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /v1/models/{id}", s.model)
	mux.HandleFunc("/v1/chat/completions", s.chat)
	mux.HandleFunc("/fault", s.fault)
	log.Fatal(http.ListenAndServe(*listen, mux))
}
