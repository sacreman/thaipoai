// Command adapter-sync is a sidecar that keeps a model server holding the adapters it should.
// It downloads each desired adapter, verifies its sha256, loads it, confirms it serves a test
// translation and publishes the result for this replica.
//
// The model server keeps its adapters in memory and exposes no hash, so each adapter is
// registered under a name that contains its hash. The server's own list then says exactly which
// weights it holds, and this sidecar needs no memory across a restart of either side.
package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sacreman/thaipoai/internal/guards"
	"github.com/sacreman/thaipoai/internal/llm"
)

const (
	maxArchiveBytes = 4 << 30
	testSentence    = "See you tomorrow at the market."
	listTimeout     = 15 * time.Second
	workTimeout     = 5 * time.Minute
)

var (
	validName = regexp.MustCompile(`^[a-z0-9][a-z0-9-]{0,40}$`)
	validHash = regexp.MustCompile(`^[0-9a-f]{64}$`)
)

type adapter struct {
	Name     string `json:"name"`
	URL      string `json:"url"`
	SHA256   string `json:"sha256"`
	Required bool   `json:"required"` // this replica takes no traffic without it
	Attempt  int    `json:"attempt"`
}

// registered is the name the model server knows this exact set of weights by.
func (a adapter) registered() string { return a.Name + "-" + a.SHA256[:12] }

// tryKey separates the retry state of one attempt from the next, so a fixed artefact is judged
// afresh instead of inheriting the last attempt's error.
func (a adapter) tryKey() string { return fmt.Sprintf("%s/%d", a.registered(), a.Attempt) }

type adapterStatus struct {
	Name       string `json:"name"`
	Registered string `json:"registered"`
	Required   bool   `json:"required"`
	Attempt    int    `json:"attempt"`
	State      string `json:"state"` // ready or failed
	Error      string `json:"error,omitempty"`
}

type status struct {
	Ready     bool            `json:"ready"`
	BackendUp bool            `json:"backend_up"`
	Error     string          `json:"error,omitempty"`
	Adapters  []adapterStatus `json:"adapters"`
}

type syncer struct {
	desiredFile string
	dir         string // where this sidecar stores adapters
	serverDir   string // where the model server sees that same directory
	backend     string
	baseURL     string
	http        *http.Client
	retry       time.Duration

	desired   []adapter
	haveFile  bool
	confirmed map[string]bool
	retryAt   map[string]time.Time
	lastError map[string]string

	mu      sync.Mutex
	current status
}

func newSyncer(desiredFile, dir, serverDir, backend string, retry time.Duration) *syncer {
	return &syncer{
		desiredFile: desiredFile, dir: dir, serverDir: serverDir, backend: strings.TrimRight(backend, "/"), retry: retry,
		http: &http.Client{}, confirmed: map[string]bool{}, retryAt: map[string]time.Time{}, lastError: map[string]string{},
	}
}

func readDesired(file string) ([]adapter, error) {
	raw, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	var parsed struct {
		Adapters []adapter `json:"adapters"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}
	for _, a := range parsed.Adapters {
		target, err := url.Parse(a.URL)
		if !validName.MatchString(a.Name) || !validHash.MatchString(a.SHA256) || err != nil || (target.Scheme != "http" && target.Scheme != "https") {
			return nil, fmt.Errorf("invalid adapter entry %q", a.Name)
		}
	}
	return parsed.Adapters, nil
}

// list returns the adapters the model server holds. The base model is not an adapter.
func (s *syncer) list(ctx context.Context) (map[string]bool, error) {
	ctx, cancel := context.WithTimeout(ctx, listTimeout)
	defer cancel()
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, s.backend+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	response, err := s.http.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("models: status %d", response.StatusCode)
	}
	var decoded struct {
		Data []struct {
			ID     string  `json:"id"`
			Parent *string `json:"parent"`
		} `json:"data"`
	}
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	loaded := map[string]bool{}
	for _, model := range decoded.Data {
		if model.Parent != nil {
			loaded[model.ID] = true
		}
	}
	return loaded, nil
}

func (s *syncer) post(ctx context.Context, endpoint string, payload map[string]string) error {
	ctx, cancel := context.WithTimeout(ctx, workTimeout)
	defer cancel()
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, s.backend+endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := s.http.Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	io.Copy(io.Discard, response.Body)
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: status %d", endpoint, response.StatusCode)
	}
	return nil
}

// extract unpacks a flat archive and takes only JSON and safetensors, so no pickle file can
// ever reach the model server.
func extract(archive, destination string) error {
	file, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer file.Close()
	unzipped, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	remaining := int64(maxArchiveBytes)
	reader := tar.NewReader(unzipped)
	for {
		header, err := reader.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		name := path.Clean(header.Name)
		if name == "." && header.Typeflag == tar.TypeDir {
			continue
		}
		if header.Typeflag != tar.TypeReg || name == ".." || strings.Contains(name, "/") {
			return fmt.Errorf("unsafe archive entry %q", header.Name)
		}
		if strings.HasPrefix(name, ".") || (!strings.HasSuffix(name, ".json") && !strings.HasSuffix(name, ".safetensors")) {
			continue
		}
		out, err := os.OpenFile(filepath.Join(destination, name), os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0o644)
		if err != nil {
			return err
		}
		written, err := io.Copy(out, io.LimitReader(reader, remaining+1))
		out.Close()
		if err != nil {
			return err
		}
		if remaining -= written; remaining < 0 {
			return errors.New("archive too large")
		}
	}
}

// fetch downloads and verifies an adapter. The rename makes a half-written adapter impossible.
func (s *syncer) fetch(ctx context.Context, a adapter) error {
	final := filepath.Join(s.dir, a.registered())
	if info, err := os.Stat(final); err == nil && info.IsDir() {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, workTimeout)
	defer cancel()
	work, err := os.MkdirTemp(s.dir, ".tmp-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(work)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, a.URL, nil)
	if err != nil {
		return err
	}
	response, err := s.http.Do(request)
	if err != nil {
		return errors.New("download failed") // the error text would carry the URL, which may be signed
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("download: status %d", response.StatusCode)
	}
	archive := filepath.Join(work, "adapter.tar.gz")
	out, err := os.Create(archive)
	if err != nil {
		return err
	}
	hash := sha256.New()
	written, err := io.Copy(io.MultiWriter(out, hash), io.LimitReader(response.Body, maxArchiveBytes+1))
	out.Close()
	if err != nil {
		return errors.New("download failed")
	}
	if written > maxArchiveBytes {
		return errors.New("archive too large")
	}
	if hex.EncodeToString(hash.Sum(nil)) != a.SHA256 {
		return errors.New("checksum mismatch")
	}
	unpacked := filepath.Join(work, "adapter")
	if err := os.Mkdir(unpacked, 0o755); err != nil {
		return err
	}
	if err := extract(archive, unpacked); err != nil {
		return err
	}
	for _, needed := range []string{"adapter_config.json", "adapter_model.safetensors"} {
		if _, err := os.Stat(filepath.Join(unpacked, needed)); err != nil {
			return fmt.Errorf("archive has no %s", needed)
		}
	}
	return os.Rename(unpacked, final)
}

// confirm asks the model server for a real translation through this adapter.
func (s *syncer) confirm(ctx context.Context, a adapter) error {
	ctx, cancel := context.WithTimeout(ctx, workTimeout)
	defer cancel()
	client := llm.Client{Endpoint: s.backend, Model: a.registered(), HTTP: s.http}
	output, err := client.Chat(ctx, llm.ChatMessages("Thai", testSentence, nil, ""))
	if err != nil {
		return fmt.Errorf("test translation: %w", err)
	}
	for _, failure := range guards.Check(guards.Request{Source: testSentence, Output: output, TargetThai: true}) {
		if failure != "gender_hedge" { // a style fault, not a broken adapter
			return fmt.Errorf("test translation failed guard %s", failure)
		}
	}
	return nil
}

func (s *syncer) ensure(ctx context.Context, a adapter, loaded map[string]bool) error {
	registered := a.registered()
	if !loaded[registered] {
		if err := s.fetch(ctx, a); err != nil {
			return err
		}
		lora := map[string]string{"lora_name": registered, "lora_path": filepath.Join(s.serverDir, registered)}
		if err := s.post(ctx, "/v1/load_lora_adapter", lora); err != nil {
			return err
		}
	}
	if !s.confirmed[registered] {
		if err := s.confirm(ctx, a); err != nil {
			return err
		}
		s.confirmed[registered] = true
	}
	return nil
}

// removeUnwanted unloads adapters nobody wants and deletes their files, and any left-over
// work directory from a crash.
func (s *syncer) removeUnwanted(ctx context.Context, loaded map[string]bool, wanted map[string]bool) {
	for registered := range loaded {
		if !wanted[registered] {
			if err := s.post(ctx, "/v1/unload_lora_adapter", map[string]string{"lora_name": registered}); err != nil {
				log.Printf("unload failed: %v", err)
			}
		}
	}
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		log.Printf("cannot read adapter directory: %v", err)
		return
	}
	for _, entry := range entries {
		if !wanted[entry.Name()] {
			os.RemoveAll(filepath.Join(s.dir, entry.Name()))
		}
	}
}

func (s *syncer) sync(ctx context.Context) status {
	var result status
	if desired, err := readDesired(s.desiredFile); err != nil {
		result.Error = "desired adapters unreadable, keeping the last good list"
	} else {
		s.desired, s.haveFile = desired, true
	}
	loaded, err := s.list(ctx)
	if err != nil {
		result.Error = "model server unreachable"
		s.confirmed = map[string]bool{}
		for _, a := range s.desired {
			result.Adapters = append(result.Adapters, adapterStatus{Name: a.Name, Registered: a.registered(), Required: a.Required, Attempt: a.Attempt, State: "failed", Error: "model server unreachable"})
		}
		return result
	}
	result.BackendUp = true
	for registered := range s.confirmed {
		if !loaded[registered] {
			delete(s.confirmed, registered) // the server restarted or dropped it
		}
	}
	if !s.haveFile {
		return result // not knowing what is wanted is no reason to remove anything
	}
	result.Ready = true
	// Required adapters go first and are published before any candidate work begins, so a
	// candidate with a slow or hung download cannot keep a healthy replica out of service.
	ordered := append([]adapter(nil), s.desired...)
	sort.SliceStable(ordered, func(i, j int) bool { return ordered[i].Required && !ordered[j].Required })
	required := 0
	for _, a := range ordered {
		if a.Required {
			required++
		}
	}
	if required == 0 && len(ordered) > 0 {
		s.publish(result)
	}
	wanted, tries := map[string]bool{}, map[string]bool{}
	for i, a := range ordered {
		registered, key := a.registered(), a.tryKey()
		wanted[registered], tries[key] = true, true
		var failure error
		if s.confirmed[registered] && loaded[registered] {
			delete(s.lastError, key)
		} else if time.Now().Before(s.retryAt[key]) {
			failure = errors.New(s.lastError[key])
		} else if failure = s.ensure(ctx, a, loaded); failure != nil {
			s.retryAt[key] = time.Now().Add(s.retry)
			s.lastError[key] = failure.Error()
		} else {
			delete(s.lastError, key)
		}
		state := adapterStatus{Name: a.Name, Registered: registered, Required: a.Required, Attempt: a.Attempt, State: "ready"}
		if failure != nil {
			state.State, state.Error = "failed", failure.Error()
			result.Ready = result.Ready && !a.Required
		}
		result.Adapters = append(result.Adapters, state)
		if i+1 == required && len(ordered) > required {
			s.publish(result)
		}
	}
	for key := range s.retryAt {
		if !tries[key] {
			delete(s.retryAt, key)
			delete(s.lastError, key)
		}
	}
	s.removeUnwanted(ctx, loaded, wanted)
	return result
}

func (s *syncer) reconcile(ctx context.Context) { s.publish(s.sync(ctx)) }

func (s *syncer) publish(next status) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if next.Ready != s.current.Ready {
		log.Printf("ready=%v", next.Ready)
	}
	s.current = next
}

func (s *syncer) snapshot() status {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.current
}

// routes serves the replica's state. Kubernetes readiness uses /readyz, so a Service sends
// traffic only to replicas that hold their required adapters.
func (s *syncer) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) { fmt.Fprintln(w, "ok") })
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, _ *http.Request) {
		if !s.snapshot().Ready {
			http.Error(w, "not ready", http.StatusServiceUnavailable)
			return
		}
		fmt.Fprintln(w, "ready")
	})
	mux.HandleFunc("/state", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(s.snapshot())
	})
	return mux
}

func main() {
	listen := flag.String("listen", ":8090", "address to serve state on")
	desired := flag.String("desired", "/etc/adapter-sync/desired.json", "file listing the desired adapters")
	dir := flag.String("dir", "/adapters", "directory to store adapters in")
	serverDir := flag.String("server-dir", "", "where the model server sees that directory, if different")
	backend := flag.String("backend", "http://127.0.0.1:8000", "model server")
	interval := flag.Duration("interval", 5*time.Second, "time between checks")
	retry := flag.Duration("retry", 60*time.Second, "time to wait before retrying a failed adapter")
	flag.Parse()
	if *serverDir == "" {
		*serverDir = *dir
	}
	s := newSyncer(*desired, *dir, *serverDir, *backend, *retry)
	go func() { log.Fatal(http.ListenAndServe(*listen, s.routes())) }()
	for {
		s.reconcile(context.Background())
		time.Sleep(*interval)
	}
}
