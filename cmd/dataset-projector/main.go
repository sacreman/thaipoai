// Command dataset-projector replays ThaiPo's event archive. This first version only counts:
// it decodes metadata fields, never message text, and reports aggregate totals. Every
// chat-day is checked against the sha256 in its manifest before it is counted.
package main

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// translation is deliberately missing source_text and target_text, so they are never decoded.
type translation struct {
	Model        string `json:"model"`
	SourceLang   string `json:"source_lang"`
	TargetLang   string `json:"target_lang"`
	ChatLang     string `json:"chat_lang"`
	Pack         string `json:"pack"`
	Native       bool   `json:"native"`
	Untranslated bool   `json:"untranslated"`
	Voice        bool   `json:"voice"`
	Image        bool   `json:"image"`
	Video        bool   `json:"video"`
	Pdf          bool   `json:"pdf"`
}

type manifest struct {
	ChatID             string `json:"chat_id"`
	Date               string `json:"date"`
	TranslationsSHA256 string `json:"translations_sha256"`
}

// Report holds totals only. It has no field that can carry text or a chat ID.
type Report struct {
	ChatDays       int            `json:"chat_days"`
	Chats          int            `json:"chats"`
	CorruptDays    int            `json:"corrupt_days"`
	UnreadableDays int            `json:"unreadable_days"`
	Rows           int            `json:"rows"`
	ByMonth        map[string]int `json:"by_month"`
	ByModel        map[string]int `json:"by_model"`
	ByPack         map[string]int `json:"by_pack"`
	ByChatType     map[string]int `json:"by_chat_type"`
	ByLanguagePair map[string]int `json:"by_language_pair"`
	Typed          int            `json:"after_typed_only"`
	Literal        int            `json:"after_literal_only"`
	EnglishThai    int            `json:"after_english_thai"`
	PackAllowed    int            `json:"after_pack_filter"`
	EnglishToThai  int            `json:"usable_english_to_thai"`
	ThaiToEnglish  int            `json:"usable_thai_to_english"`
	OneToOneUsable int            `json:"usable_one_to_one"`
	GroupUsable    int            `json:"usable_group"`
	DeepSeekLabels int            `json:"usable_deepseek_labels"`
	RelabelNeeded  int            `json:"usable_needs_relabel"`
	seen           map[string]bool
}

func newReport() *Report {
	return &Report{
		ByMonth: map[string]int{}, ByModel: map[string]int{}, ByPack: map[string]int{},
		ByChatType: map[string]int{}, ByLanguagePair: map[string]int{}, seen: map[string]bool{},
	}
}

func labelSource(model string) string {
	switch {
	case model == "":
		return "unknown"
	case strings.HasPrefix(model, "deepseek"):
		return "deepseek"
	case strings.HasPrefix(model, "claude"), strings.Contains(model, "haiku"), strings.Contains(model, "sonnet"), strings.Contains(model, "opus"):
		return "claude"
	case strings.HasPrefix(model, "gemini"):
		return "gemini"
	}
	return "other"
}

func chatType(chatID string) string {
	switch {
	case strings.HasPrefix(chatID, "U"):
		return "one_to_one"
	case strings.HasPrefix(chatID, "C"), strings.HasPrefix(chatID, "R"):
		return "group"
	}
	return "other"
}

func (r *Report) add(m manifest, t translation, excluded map[string]bool) {
	r.Rows++
	r.ByMonth[monthOf(m.Date)]++
	r.ByModel[labelSource(t.Model)]++
	kind := chatType(m.ChatID)
	r.ByChatType[kind]++
	pack := t.Pack
	if pack == "" {
		pack = "none"
	}
	r.ByPack[pack]++
	r.ByLanguagePair[pairName(t)]++

	if t.Voice || t.Image || t.Video || t.Pdf {
		return
	}
	r.Typed++
	if t.Native || t.Untranslated {
		return
	}
	r.Literal++
	direction := englishThaiDirection(t)
	if direction == "" {
		return
	}
	r.EnglishThai++
	if excluded[t.Pack] {
		return
	}
	r.PackAllowed++
	if direction == "en-th" {
		r.EnglishToThai++
	} else {
		r.ThaiToEnglish++
	}
	if kind == "one_to_one" {
		r.OneToOneUsable++
	} else if kind == "group" {
		r.GroupUsable++
	}
	if labelSource(t.Model) == "deepseek" {
		r.DeepSeekLabels++
	} else {
		r.RelabelNeeded++
	}
}

func monthOf(date string) string {
	if len(date) >= 7 {
		return date[:7]
	}
	return "unknown"
}

func pairName(t translation) string {
	source, target := t.SourceLang, t.TargetLang
	if source == "" {
		source = "?"
	}
	if target == "" {
		target = "?"
	}
	return source + ">" + target
}

func englishThaiDirection(t translation) string {
	if t.ChatLang != "" {
		return ""
	}
	switch pairName(t) {
	case "en>th":
		return "en-th"
	case "th>en":
		return "th-en"
	}
	return ""
}

func (r *Report) merge(day *dayResult) {
	r.ChatDays++
	if day.corrupt {
		r.CorruptDays++
		return
	}
	if day.unreadable {
		r.UnreadableDays++
		return
	}
	r.seen[day.manifest.ChatID] = true
	r.Chats = len(r.seen)
}

type dayResult struct {
	manifest            manifest
	rows                []translation
	corrupt, unreadable bool
}

type store interface {
	manifests(ctx context.Context) ([]string, error)
	get(ctx context.Context, key string) ([]byte, error)
}

func processDay(ctx context.Context, s store, key string) *dayResult {
	day := &dayResult{}
	raw, err := s.get(ctx, key)
	if err != nil || json.Unmarshal(raw, &day.manifest) != nil {
		day.unreadable = true
		return day
	}
	translationsKey := strings.TrimSuffix(key, ".manifest.json") + ".translations.jsonl.gz"
	stored, err := s.get(ctx, translationsKey)
	if err != nil {
		day.unreadable = true
		return day
	}
	sum := sha256.Sum256(stored)
	if hex.EncodeToString(sum[:]) != day.manifest.TranslationsSHA256 {
		day.corrupt = true
		return day
	}
	zr, err := gzip.NewReader(bytes.NewReader(stored))
	if err != nil {
		day.corrupt = true
		return day
	}
	defer zr.Close()
	decoder := json.NewDecoder(zr)
	for {
		var t translation
		if err := decoder.Decode(&t); err == io.EOF {
			return day
		} else if err != nil {
			day.rows = nil
			day.corrupt = true
			return day
		}
		day.rows = append(day.rows, t)
	}
}

func count(ctx context.Context, s store, excluded map[string]bool, workers int) (*Report, error) {
	keys, err := s.manifests(ctx)
	if err != nil {
		return nil, err
	}
	report := newReport()
	var mu sync.Mutex
	var wg sync.WaitGroup
	jobs := make(chan string)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for key := range jobs {
				day := processDay(ctx, s, key)
				mu.Lock()
				report.merge(day)
				for _, t := range day.rows {
					report.add(day.manifest, t, excluded)
				}
				mu.Unlock()
			}
		}()
	}
	for _, key := range keys {
		jobs <- key
	}
	close(jobs)
	wg.Wait()
	return report, nil
}

type dirStore struct{ root string }

func (d dirStore) manifests(context.Context) ([]string, error) {
	var keys []string
	err := filepath.WalkDir(filepath.Join(d.root, "archive"), func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !entry.IsDir() && strings.HasSuffix(path, ".manifest.json") {
			keys = append(keys, path)
		}
		return nil
	})
	return keys, err
}

func (d dirStore) get(_ context.Context, key string) ([]byte, error) { return os.ReadFile(key) }

type s3Store struct {
	client *s3.Client
	bucket string
}

func (s s3Store) manifests(ctx context.Context) ([]string, error) {
	var keys []string
	pages := s3.NewListObjectsV2Paginator(s.client, &s3.ListObjectsV2Input{Bucket: aws.String(s.bucket), Prefix: aws.String("archive/")})
	for pages.HasMorePages() {
		page, err := pages.NextPage(ctx)
		if err != nil {
			return nil, err
		}
		for _, object := range page.Contents {
			if strings.HasSuffix(aws.ToString(object.Key), ".manifest.json") {
				keys = append(keys, aws.ToString(object.Key))
			}
		}
	}
	return keys, nil
}

func (s s3Store) get(ctx context.Context, key string) ([]byte, error) {
	out, err := s.client.GetObject(ctx, &s3.GetObjectInput{Bucket: aws.String(s.bucket), Key: aws.String(key)})
	if err != nil {
		return nil, err
	}
	defer out.Body.Close()
	return io.ReadAll(out.Body)
}

func packSet(list string) map[string]bool {
	set := map[string]bool{}
	for _, name := range strings.Split(list, ",") {
		if name = strings.TrimSpace(name); name != "" {
			set[name] = true
		}
	}
	return set
}

func main() {
	dir := flag.String("dir", "", "local directory holding an archive/ tree")
	bucket := flag.String("bucket", "", "S3 bucket, using the standard AWS environment variables")
	exclude := flag.String("exclude-packs", "", "comma-separated industry packs to leave out")
	output := flag.String("out", "", "report file")
	workers := flag.Int("workers", 16, "parallel chat-days")
	flag.Parse()
	if (*dir == "") == (*bucket == "") || *output == "" {
		log.Fatal("give exactly one of --dir or --bucket, and --out")
	}
	ctx := context.Background()
	var source store = dirStore{root: *dir}
	if *bucket != "" {
		cfg, err := config.LoadDefaultConfig(ctx)
		if err != nil {
			log.Fatal("cannot load AWS configuration")
		}
		source = s3Store{client: s3.NewFromConfig(cfg), bucket: *bucket}
	}
	report, err := count(ctx, source, packSet(*exclude), *workers)
	if err != nil {
		log.Fatalf("listing the archive failed: %T", err)
	}
	encoded, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*output, encoded, 0o600); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%d chat-days, %d rows, %d usable\n", report.ChatDays, report.Rows, report.PackAllowed)
}
