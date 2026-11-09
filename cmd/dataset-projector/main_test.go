package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const secretText = "PRIVATE-MESSAGE-TEXT"

func row(model, source, target string, change func(map[string]any)) map[string]any {
	r := map[string]any{"model": model, "source_lang": source, "target_lang": target, "source_text": secretText, "target_text": secretText}
	if change != nil {
		change(r)
	}
	return r
}

func writeDay(t *testing.T, root, shard, chat, date string, rows []map[string]any, damage bool) {
	t.Helper()
	dir := filepath.Join(root, "archive", shard, chat)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	for _, r := range rows {
		line, _ := json.Marshal(r)
		zw.Write(append(line, '\n'))
	}
	zw.Close()
	sum := sha256.Sum256(buf.Bytes())
	stored := buf.Bytes()
	if damage {
		stored = append([]byte{}, stored...)
		stored[len(stored)-1] ^= 0xff
	}
	if err := os.WriteFile(filepath.Join(dir, date+".translations.jsonl.gz"), stored, 0o600); err != nil {
		t.Fatal(err)
	}
	m, _ := json.Marshal(map[string]string{"chat_id": chat, "date": date, "translations_sha256": hex.EncodeToString(sum[:])})
	if err := os.WriteFile(filepath.Join(dir, date+".manifest.json"), m, 0o600); err != nil {
		t.Fatal(err)
	}
}

func set(key string, value any) func(map[string]any) {
	return func(r map[string]any) { r[key] = value }
}

func fixture(t *testing.T) *Report {
	t.Helper()
	root := t.TempDir()
	writeDay(t, root, "shard-00", "Uperson", "2026-09-14", []map[string]any{
		row("deepseek-flash", "en", "th", nil),
		row("deepseek-flash", "th", "en", nil),
		row("claude-haiku-5-5", "en", "th", nil),
		row("deepseek-flash", "en", "th", set("voice", true)),
		row("deepseek-flash", "en", "th", set("native", true)),
		row("deepseek-flash", "en", "th", set("untranslated", true)),
		row("deepseek-flash", "en", "th", set("pack", "excluded-pack")),
		row("deepseek-flash", "ja", "th", set("chat_lang", "ja")),
	}, false)
	writeDay(t, root, "shard-01", "Cgroup", "2026-10-02", []map[string]any{row("deepseek-flash", "en", "th", nil)}, false)
	writeDay(t, root, "shard-02", "Udamaged", "2026-10-03", []map[string]any{row("deepseek-flash", "en", "th", nil)}, true)
	report, err := count(context.Background(), dirStore{root: root}, packSet("excluded-pack"), 4)
	if err != nil {
		t.Fatal(err)
	}
	return report
}

func TestFunnelCountsEachFilterInOrder(t *testing.T) {
	r := fixture(t)
	if r.Rows != 9 || r.Typed != 8 || r.Literal != 6 || r.EnglishThai != 5 || r.PackAllowed != 4 {
		t.Fatalf("funnel %+v", r)
	}
}

func TestUsableRowsAreSplitByDirectionChatAndLabel(t *testing.T) {
	r := fixture(t)
	if r.EnglishToThai != 3 || r.ThaiToEnglish != 1 || r.OneToOneUsable != 3 || r.GroupUsable != 1 || r.DeepSeekLabels != 3 || r.RelabelNeeded != 1 {
		t.Fatalf("usable %+v", r)
	}
}

func TestDamagedDaysAreCountedAndExcluded(t *testing.T) {
	r := fixture(t)
	if r.CorruptDays != 1 || r.ChatDays != 3 || r.Chats != 2 {
		t.Fatalf("days %+v", r)
	}
}

func TestReportCarriesNoTextOrChatIDs(t *testing.T) {
	encoded, err := json.Marshal(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	for _, leak := range []string{secretText, "Uperson", "Cgroup"} {
		if strings.Contains(string(encoded), leak) {
			t.Fatalf("report contains %q", leak)
		}
	}
}

func TestMonthsAndModelsAreGrouped(t *testing.T) {
	r := fixture(t)
	if r.ByMonth["2026-09"] != 8 || r.ByMonth["2026-10"] != 1 || r.ByModel["claude"] != 1 || r.ByPack["excluded-pack"] != 1 {
		t.Fatalf("groups %+v", r)
	}
}

func TestLabelSources(t *testing.T) {
	for model, want := range map[string]string{"deepseek-flash": "deepseek", "claude-opus-5-5": "claude", "gemini-3.1-flash-lite": "gemini", "": "unknown", "gpt-5": "other"} {
		if got := labelSource(model); got != want {
			t.Fatalf("%s: got %s", model, got)
		}
	}
}
