// Command suite runs translation test cases against any OpenAI-compatible endpoint
// and writes one JSON result per case. A failed request is recorded, never skipped,
// so failures stay in the denominator.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/sacreman/thaipoai/internal/guards"
	"github.com/sacreman/thaipoai/internal/llm"
)

// Case is one line of a suite file.
type Case struct {
	ID                string   `json:"id"`
	Direction         string   `json:"direction"` // "en-th" or "th-en"
	Category          string   `json:"category"`
	Context           []string `json:"context,omitempty"`
	IrrelevantContext []string `json:"irrelevant_context,omitempty"`
	Source            string   `json:"source"`
	Meaning           string   `json:"meaning"`
	LiteralTrap       string   `json:"literal_trap,omitempty"`
}

// Result is what a run records for a case.
type Result struct {
	ID        string   `json:"id"`
	System    string   `json:"system"`
	Variant   string   `json:"variant"`
	Direction string   `json:"direction"`
	Category  string   `json:"category"`
	Output    string   `json:"output"`
	Failures  []string `json:"failures,omitempty"`
	Error     string   `json:"error,omitempty"`
	Millis    int64    `json:"millis"`
}

type config struct {
	endpoint, model, system, variant, apiKey string
	temperature                              float64
	noThinking                               bool
	client                                   *http.Client
}

func targetLanguage(direction string) (string, error) {
	switch direction {
	case "en-th":
		return "Thai", nil
	case "th-en":
		return "English", nil
	}
	return "", fmt.Errorf("unknown direction %q", direction)
}

func contextFor(c Case, variant string) ([]string, error) {
	switch variant {
	case "none":
		return nil, nil
	case "correct":
		return c.Context, nil
	case "irrelevant":
		return c.IrrelevantContext, nil
	}
	return nil, fmt.Errorf("unknown variant %q", variant)
}

// checks are the deterministic failures. Meaning itself is left to the scorers.
func checks(c Case, output string) []string {
	var failures []string
	if strings.TrimSpace(output) == "" {
		failures = append(failures, "empty")
	}
	if c.LiteralTrap != "" && strings.Contains(output, c.LiteralTrap) {
		failures = append(failures, "literal_trap")
	}
	return failures
}

func (cfg config) translate(messages []map[string]string) (string, error) {
	client := llm.Client{Endpoint: cfg.endpoint, Model: cfg.model, APIKey: cfg.apiKey, Temperature: cfg.temperature, NoThinking: cfg.noThinking, HTTP: cfg.client}
	return client.Chat(context.Background(), messages)
}

func (cfg config) run(c Case) Result {
	result := Result{ID: c.ID, System: cfg.system, Variant: cfg.variant, Direction: c.Direction, Category: c.Category}
	target, err := targetLanguage(c.Direction)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	context, err := contextFor(c, cfg.variant)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	started := time.Now()
	output, err := cfg.translate(llm.ChatMessages(target, c.Source, context, ""))
	result.Millis = time.Since(started).Milliseconds()
	if err != nil {
		result.Error = err.Error()
		return result
	}
	result.Output = output
	result.Failures = checks(c, output)
	return result
}

func runSuite(cfg config, in io.Reader, out io.Writer) (int, error) {
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)
	encoder := json.NewEncoder(out)
	encoder.SetEscapeHTML(false)
	count := 0
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) == "" {
			continue
		}
		var c Case
		if err := json.Unmarshal(scanner.Bytes(), &c); err != nil {
			return count, fmt.Errorf("line %d: %w", count+1, err)
		}
		if err := encoder.Encode(cfg.run(c)); err != nil {
			return count, err
		}
		count++
	}
	return count, scanner.Err()
}

// recheck runs the guards over saved outputs, so a guard can be measured against scored results.
func recheck(cases, results string, out io.Writer) error {
	byID := map[string]Case{}
	in, err := os.ReadFile(cases)
	if err != nil {
		return err
	}
	for _, line := range strings.Split(string(in), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var c Case
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			return err
		}
		byID[c.ID] = c
	}
	saved, err := os.ReadFile(results)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(out)
	for _, line := range strings.Split(string(saved), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var r Result
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			return err
		}
		failed := guards.Check(guards.Request{Source: byID[r.ID].Source, Output: r.Output, TargetThai: r.Direction == "en-th"})
		if err := encoder.Encode(map[string]any{"id": r.ID, "variant": r.Variant, "direction": r.Direction, "guards": failed}); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	suite := flag.String("suite", "", "suite file, one JSON case per line")
	output := flag.String("out", "", "results file")
	endpoint := flag.String("endpoint", "http://127.0.0.1:8080", "OpenAI-compatible base URL")
	model := flag.String("model", "default", "model name sent to the endpoint")
	system := flag.String("system", "", "name recorded on each result")
	variant := flag.String("variant", "none", "context variant: none, correct or irrelevant")
	keyFile := flag.String("api-key-file", "", "file holding the API key, never printed")
	temperature := flag.Float64("temperature", 0, "sampling temperature")
	noThinking := flag.Bool("no-thinking", false, "turn off reasoning, for DeepSeek")
	recheckFile := flag.String("recheck", "", "saved results to re-run the guards over, instead of calling a model")
	flag.Parse()
	if *recheckFile != "" {
		destination, err := os.Create(*output)
		if err != nil {
			log.Fatal(err)
		}
		defer destination.Close()
		if err := recheck(*suite, *recheckFile, destination); err != nil {
			log.Fatal(err)
		}
		return
	}
	if *suite == "" || *output == "" || *system == "" {
		log.Fatal("--suite, --out and --system are required")
	}
	in, err := os.Open(*suite)
	if err != nil {
		log.Fatal(err)
	}
	defer in.Close()
	out, err := os.Create(*output)
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	cfg := config{endpoint: strings.TrimRight(*endpoint, "/"), model: *model, system: *system, variant: *variant,
		temperature: *temperature, noThinking: *noThinking, client: &http.Client{Timeout: 5 * time.Minute}}
	if *keyFile != "" {
		key, err := os.ReadFile(*keyFile)
		if err != nil {
			log.Fatal("cannot read the API key file")
		}
		cfg.apiKey = strings.TrimSpace(string(key))
	}
	count, err := runSuite(cfg, in, out)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d results written to %s\n", count, *output)
}
