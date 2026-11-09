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
	Notes             string   `json:"notes,omitempty"` // extra system text, chat format only
}

// Result is what a run records for a case.
type Result struct {
	ID        string   `json:"id"`
	System    string   `json:"system"`
	Variant   string   `json:"variant"`
	Format    string   `json:"format,omitempty"`
	Direction string   `json:"direction"`
	Category  string   `json:"category"`
	Output    string   `json:"output"`
	Failures  []string `json:"failures,omitempty"`
	Error     string   `json:"error,omitempty"`
	Millis    int64    `json:"millis"`
}

type config struct {
	endpoint, model, system, variant, format string
	apiKey                                   string
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

// lumenPrompt renders Lumen's documented format. With no context, and in the system
// format, it must match lumen_messages in train/examples.py, which the model is trained on.
func lumenPrompt(target, source string, context []string) string {
	guidelines := ""
	if len(context) > 0 {
		guidelines = "Use this conversation only as context and do not translate it:\n" + strings.Join(context, "\n")
	}
	return fmt.Sprintf("Please translate the following text into %s.\n#guidelines\n%s\n#src_text\n%s", target, guidelines, source)
}

// hyPrompt is Hy-MT2's documented format, using its background-information form for history.
// It must match render in train/examples.py.
func hyPrompt(target, source string, context []string) string {
	if len(context) == 0 {
		return fmt.Sprintf("Translate the following text into %s. Note that you should only output the translated result without any additional explanation:\n\n%s", target, source)
	}
	return fmt.Sprintf("[Background Information]\n%s\n\nPlease translate the following text into %s, taking the provided background information into consideration.\n\n[Source Text]\n%s", strings.Join(context, "\n"), target, source)
}

// qingqiuPrompt is Qingqiu-MT's documented format. It must match qwen_messages in train/examples.py.
func qingqiuPrompt(target, source string, context []string) string {
	from := "English"
	if target == "English" {
		from = "Thai"
	}
	instruction := "Translate the meaning naturally and match the sender's tone."
	if len(context) > 0 {
		instruction += " Use this conversation only as context and do not translate it:\n" + strings.Join(context, "\n")
	}
	return fmt.Sprintf("Translate the following %s text into %s. Strictly follow the instruction below. Output only the translation.\n\ninstruction:\n%s\n\ntext:\n%s", from, target, instruction, source)
}

// indexPrompt is Index-Translate's documented Chinese format, with history as a soft constraint.
func indexPrompt(target, source string, context []string) string {
	name := "泰语"
	if target == "English" {
		name = "英语"
	}
	if len(context) == 0 {
		return fmt.Sprintf("请将以下文本翻译为%s，直接输出翻译结果，不要进行任何解释。\n\n%s", name, source)
	}
	return fmt.Sprintf("请将以下文本翻译成%s，并且严格遵循所有约束要求。\n\n【源文】\n%s\n\n【约束要求】\n1. 【注意】以下是聊天中之前的消息，仅供理解语境，切勿翻译：\n%s\n\n只输出译文，不要有任何额外说明。", name, source, strings.Join(context, "\n"))
}

// chindaPrompt is ChindaMT's documented format, with history in its free-form Rules block.
func chindaPrompt(target, source string, context []string) string {
	from, tag := "English", "EN"
	if target == "English" {
		from, tag = "Thai", "TH"
	}
	rules := ""
	if len(context) > 0 {
		rules = "Rules:\n- Return only the translated text\n- Use these earlier messages only to understand the message. Never translate them:\n" + strings.Join(context, "\n") + "\n"
	}
	return fmt.Sprintf("Translate %s to %s.\n%s\n%s: %s", from, target, rules, tag, source)
}

// messagesFor renders one way of supplying chat history. With no history every format
// is the same plain Lumen prompt.
func messagesFor(format, target, source string, context []string) ([]map[string]string, error) {
	user := func(content string) map[string]string { return map[string]string{"role": "user", "content": content} }
	plain := lumenPrompt(target, source, nil)
	switch format {
	case "hy":
		return []map[string]string{user(hyPrompt(target, source, context))}, nil
	case "qingqiu":
		return []map[string]string{user(qingqiuPrompt(target, source, context))}, nil
	case "index":
		return []map[string]string{user(indexPrompt(target, source, context))}, nil
	case "chinda":
		return []map[string]string{user(chindaPrompt(target, source, context))}, nil
	}
	if len(context) == 0 && format != "chat" {
		switch format {
		case "guidelines", "reference", "system", "before":
			return []map[string]string{user(plain)}, nil
		}
	}
	history := strings.Join(context, "\n")
	switch format {
	case "chat":
		return llm.ChatMessages(target, source, context, ""), nil
	case "guidelines":
		return []map[string]string{user(lumenPrompt(target, source, context))}, nil
	case "reference":
		guidelines := "This message is part of a chat. Translate only the text after #src_text. " +
			"Never translate the earlier messages below, which are for reference only.\nEarlier messages:\n" + history
		return []map[string]string{user(fmt.Sprintf("Please translate the following text into %s.\n#guidelines\n%s\n#src_text\n%s", target, guidelines, source))}, nil
	case "system":
		return []map[string]string{{"role": "system", "content": llm.ReferenceOnly + history}, user(plain)}, nil
	case "before":
		return []map[string]string{user(llm.ReferenceOnly + history + "\n\n" + plain)}, nil
	}
	return nil, fmt.Errorf("unknown format %q", format)
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
	result := Result{ID: c.ID, System: cfg.system, Variant: cfg.variant, Format: cfg.format, Direction: c.Direction, Category: c.Category}
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
	messages, err := messagesFor(cfg.format, target, c.Source, context)
	if err != nil {
		result.Error = err.Error()
		return result
	}
	if cfg.format == "chat" && c.Notes != "" {
		messages = llm.ChatMessages(target, c.Source, context, c.Notes)
	}
	started := time.Now()
	output, err := cfg.translate(messages)
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
	format := flag.String("format", "chat", "how history is sent: system, guidelines, reference, before, chat, hy, qingqiu, index or chinda")
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
		format: *format, temperature: *temperature, noThinking: *noThinking, client: &http.Client{Timeout: 5 * time.Minute}}
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
