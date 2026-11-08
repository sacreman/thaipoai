// Command suite runs translation test cases against any OpenAI-compatible endpoint
// and writes one JSON result per case. A failed request is recorded, never skipped,
// so failures stay in the denominator.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
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
	endpoint, model, system, variant string
	client                           *http.Client
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

// lumenPrompt renders Lumen's documented format. It must match lumen_messages in
// train/examples.py, because that is what the model is trained on.
func lumenPrompt(target, source string, context []string) string {
	guidelines := ""
	if len(context) > 0 {
		guidelines = "Use this conversation only as context and do not translate it:\n" + strings.Join(context, "\n")
	}
	return fmt.Sprintf("Please translate the following text into %s.\n#guidelines\n%s\n#src_text\n%s", target, guidelines, source)
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

func (cfg config) translate(prompt string) (string, error) {
	body, err := json.Marshal(map[string]any{
		"model":       cfg.model,
		"messages":    []map[string]string{{"role": "user", "content": prompt}},
		"temperature": 0,
		"max_tokens":  256,
	})
	if err != nil {
		return "", err
	}
	resp, err := cfg.client.Post(cfg.endpoint+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status %d", resp.StatusCode)
	}
	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return "", err
	}
	if len(decoded.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}
	return strings.TrimSpace(decoded.Choices[0].Message.Content), nil
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
	output, err := cfg.translate(lumenPrompt(target, c.Source, context))
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

func main() {
	suite := flag.String("suite", "", "suite file, one JSON case per line")
	output := flag.String("out", "", "results file")
	endpoint := flag.String("endpoint", "http://127.0.0.1:8080", "OpenAI-compatible base URL")
	model := flag.String("model", "default", "model name sent to the endpoint")
	system := flag.String("system", "", "name recorded on each result")
	variant := flag.String("variant", "none", "context variant: none, correct or irrelevant")
	flag.Parse()
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
		client: &http.Client{Timeout: 5 * time.Minute}}
	count, err := runSuite(cfg, in, out)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d results written to %s\n", count, *output)
}
