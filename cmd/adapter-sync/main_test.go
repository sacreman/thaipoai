package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func archiveOf(t *testing.T, files map[string]string) []byte {
	t.Helper()
	var buffer bytes.Buffer
	zipped := gzip.NewWriter(&buffer)
	writer := tar.NewWriter(zipped)
	for name, body := range files {
		if err := writer.WriteHeader(&tar.Header{Name: name, Typeflag: tar.TypeReg, Mode: 0o644, Size: int64(len(body))}); err != nil {
			t.Fatal(err)
		}
		writer.Write([]byte(body))
	}
	writer.Close()
	zipped.Close()
	return buffer.Bytes()
}

func goodArchive(t *testing.T) []byte {
	return archiveOf(t, map[string]string{"adapter_config.json": "{}", "adapter_model.safetensors": "weights", "evil.bin": "pickle"})
}

func hashOf(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

// fakeServer behaves like vLLM: adapters live in memory and it reports no hash.
type fakeServer struct {
	mu        sync.Mutex
	adapters  map[string]string
	loads     int
	reply     string
	failLoad  bool
	downloads int
	archives  map[string][]byte
	down      bool
}

func (f *fakeServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.down {
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}
	switch {
	case strings.HasPrefix(r.URL.Path, "/store/"):
		f.downloads++
		w.Write(f.archives[strings.TrimPrefix(r.URL.Path, "/store/")])
	case r.URL.Path == "/v1/models":
		data := []map[string]any{{"id": "base", "parent": nil}}
		for name := range f.adapters {
			data = append(data, map[string]any{"id": name, "parent": "base"})
		}
		json.NewEncoder(w).Encode(map[string]any{"data": data})
	case r.URL.Path == "/v1/load_lora_adapter":
		var body map[string]string
		json.NewDecoder(r.Body).Decode(&body)
		if f.failLoad {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		f.loads++
		f.adapters[body["lora_name"]] = body["lora_path"]
	case r.URL.Path == "/v1/unload_lora_adapter":
		var body map[string]string
		json.NewDecoder(r.Body).Decode(&body)
		delete(f.adapters, body["lora_name"])
	case r.URL.Path == "/v1/chat/completions":
		var body struct {
			Model string `json:"model"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if _, ok := f.adapters[body.Model]; !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": f.reply}}}})
	default:
		w.WriteHeader(http.StatusNotFound)
	}
}

type rig struct {
	fake    *fakeServer
	ts      *httptest.Server
	s       *syncer
	desired string
}

func newRig(t *testing.T) *rig {
	t.Helper()
	fake := &fakeServer{adapters: map[string]string{}, archives: map[string][]byte{}, reply: "เจอกันพรุ่งนี้ที่ตลาด"}
	ts := httptest.NewServer(fake)
	t.Cleanup(ts.Close)
	desired := filepath.Join(t.TempDir(), "desired.json")
	s := newSyncer(desired, t.TempDir(), "/server/adapters", ts.URL, 0)
	return &rig{fake: fake, ts: ts, s: s, desired: desired}
}

func (r *rig) want(t *testing.T, adapters ...adapter) {
	t.Helper()
	raw, _ := json.Marshal(map[string]any{"adapters": adapters})
	if err := os.WriteFile(r.desired, raw, 0o644); err != nil {
		t.Fatal(err)
	}
}

func (r *rig) adapter(t *testing.T, name string, required bool, body []byte) adapter {
	t.Helper()
	r.fake.archives[name] = body
	return adapter{Name: name, URL: r.ts.URL + "/store/" + name, SHA256: hashOf(body), Required: required}
}

func TestAnAdapterIsVerifiedLoadedAndConfirmed(t *testing.T) {
	r := newRig(t)
	a := r.adapter(t, "stable", true, goodArchive(t))
	r.want(t, a)
	r.s.reconcile(context.Background())
	got := r.s.snapshot()
	if !got.Ready || got.Adapters[0].State != "ready" {
		t.Fatalf("%+v", got)
	}
	if path := r.fake.adapters[a.registered()]; path != "/server/adapters/"+a.registered() {
		t.Fatalf("loaded from %q", path)
	}
	if _, err := os.Stat(filepath.Join(r.s.dir, a.registered(), "evil.bin")); err == nil {
		t.Fatal("a non-weights file was extracted")
	}
	r.s.reconcile(context.Background())
	if r.fake.loads != 1 || r.fake.downloads != 1 {
		t.Fatalf("loads %d downloads %d", r.fake.loads, r.fake.downloads)
	}
}

func TestAChecksumMismatchIsNeverLoaded(t *testing.T) {
	r := newRig(t)
	a := r.adapter(t, "stable", true, goodArchive(t))
	a.SHA256 = strings.Repeat("0", 64)
	r.want(t, a)
	r.s.reconcile(context.Background())
	got := r.s.snapshot()
	if got.Ready || got.Adapters[0].Error != "checksum mismatch" || r.fake.loads != 0 {
		t.Fatalf("%+v loads %d", got, r.fake.loads)
	}
}

func TestABadCandidateDoesNotTakeTheReplicaOutOfService(t *testing.T) {
	r := newRig(t)
	stable := r.adapter(t, "stable", true, goodArchive(t))
	candidate := r.adapter(t, "candidate", false, goodArchive(t))
	candidate.SHA256 = strings.Repeat("0", 64)
	r.want(t, stable, candidate)
	r.s.reconcile(context.Background())
	got := r.s.snapshot()
	if !got.Ready || got.Adapters[1].State != "failed" {
		t.Fatalf("%+v", got)
	}
}

func TestAnAdapterThatTranslatesBadlyIsNotReady(t *testing.T) {
	r := newRig(t)
	r.fake.reply = "See you tomorrow at the market."
	r.want(t, r.adapter(t, "stable", true, goodArchive(t)))
	r.s.reconcile(context.Background())
	got := r.s.snapshot()
	if got.Ready || !strings.Contains(got.Adapters[0].Error, "guard") {
		t.Fatalf("%+v", got)
	}
}

func TestAFailedLoadIsNotReady(t *testing.T) {
	r := newRig(t)
	r.fake.failLoad = true
	r.want(t, r.adapter(t, "stable", true, goodArchive(t)))
	r.s.reconcile(context.Background())
	if got := r.s.snapshot(); got.Ready || got.Adapters[0].State != "failed" {
		t.Fatalf("%+v", got)
	}
}

func TestAModelServerRestartIsHealedWithoutADownload(t *testing.T) {
	r := newRig(t)
	r.want(t, r.adapter(t, "stable", true, goodArchive(t)))
	r.s.reconcile(context.Background())
	r.fake.adapters = map[string]string{}
	r.s.reconcile(context.Background())
	if got := r.s.snapshot(); !got.Ready || len(r.fake.adapters) != 1 || r.fake.loads != 2 || r.fake.downloads != 1 {
		t.Fatalf("%+v loads %d downloads %d", got, r.fake.loads, r.fake.downloads)
	}
}

func TestASidecarRestartRechecksWhatTheServerHolds(t *testing.T) {
	r := newRig(t)
	r.want(t, r.adapter(t, "stable", true, goodArchive(t)))
	r.s.reconcile(context.Background())
	restarted := newSyncer(r.desired, r.s.dir, r.s.serverDir, r.ts.URL, 0)
	restarted.reconcile(context.Background())
	if got := restarted.snapshot(); !got.Ready || r.fake.loads != 1 || r.fake.downloads != 1 {
		t.Fatalf("%+v loads %d downloads %d", got, r.fake.loads, r.fake.downloads)
	}
}

func TestAnUnreachableServerIsNotReady(t *testing.T) {
	r := newRig(t)
	r.want(t, r.adapter(t, "stable", true, goodArchive(t)))
	r.s.reconcile(context.Background())
	r.fake.down = true
	r.s.reconcile(context.Background())
	if got := r.s.snapshot(); got.Ready || got.BackendUp {
		t.Fatalf("%+v", got)
	}
}

func TestAnAdapterNoLongerWantedIsRemoved(t *testing.T) {
	r := newRig(t)
	old := r.adapter(t, "old", false, goodArchive(t))
	r.want(t, old)
	r.s.reconcile(context.Background())
	r.want(t)
	r.s.reconcile(context.Background())
	if len(r.fake.adapters) != 0 {
		t.Fatalf("still loaded: %v", r.fake.adapters)
	}
	if _, err := os.Stat(filepath.Join(r.s.dir, old.registered())); err == nil {
		t.Fatal("files were not removed")
	}
}

func TestChangedWeightsUnderTheSameNameAreANewAdapter(t *testing.T) {
	r := newRig(t)
	first := r.adapter(t, "stable", true, goodArchive(t))
	r.want(t, first)
	r.s.reconcile(context.Background())
	second := r.adapter(t, "stable", true, archiveOf(t, map[string]string{"adapter_config.json": "{}", "adapter_model.safetensors": "new weights"}))
	r.want(t, second)
	r.s.reconcile(context.Background())
	if _, old := r.fake.adapters[first.registered()]; old || r.fake.adapters[second.registered()] == "" {
		t.Fatalf("%v", r.fake.adapters)
	}
}

func TestABrokenDesiredFileKeepsTheLastGoodList(t *testing.T) {
	r := newRig(t)
	r.want(t, r.adapter(t, "stable", true, goodArchive(t)))
	r.s.reconcile(context.Background())
	os.WriteFile(r.desired, []byte("{"), 0o644)
	r.s.reconcile(context.Background())
	if got := r.s.snapshot(); !got.Ready || got.Error == "" || len(r.fake.adapters) != 1 {
		t.Fatalf("%+v", got)
	}
}

func TestANewAttemptIsNotHeldBackByTheLastAttemptsError(t *testing.T) {
	r := newRig(t)
	r.s.retry = time.Hour
	body := goodArchive(t)
	r.fake.archives["cand"] = []byte("damaged")
	a := adapter{Name: "cand", URL: r.ts.URL + "/store/cand", SHA256: hashOf(body)}
	r.want(t, a)
	r.s.reconcile(context.Background())
	if got := r.s.snapshot(); got.Adapters[0].Error != "checksum mismatch" {
		t.Fatalf("%+v", got)
	}
	r.fake.archives["cand"] = body
	a.Attempt = 1
	r.want(t, a)
	r.s.reconcile(context.Background())
	if got := r.s.snapshot(); got.Adapters[0].State != "ready" || got.Adapters[0].Attempt != 1 {
		t.Fatalf("%+v", got)
	}
}

func TestStartingWithNoListRemovesNothing(t *testing.T) {
	r := newRig(t)
	r.fake.adapters["held-000000000000"] = "/x"
	os.WriteFile(filepath.Join(r.s.dir, "keep"), []byte("x"), 0o644)
	r.s.reconcile(context.Background())
	if got := r.s.snapshot(); got.Ready || len(r.fake.adapters) != 1 {
		t.Fatalf("%+v %v", got, r.fake.adapters)
	}
	if _, err := os.Stat(filepath.Join(r.s.dir, "keep")); err != nil {
		t.Fatal("files were removed")
	}
}

func TestReadinessIsPublishedBeforeASlowCandidate(t *testing.T) {
	r := newRig(t)
	stable := r.adapter(t, "stable", true, goodArchive(t))
	entered, release := make(chan struct{}), make(chan struct{})
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		close(entered)
		<-release
	}))
	defer slow.Close()
	candidate := adapter{Name: "candidate", URL: slow.URL + "/x", SHA256: strings.Repeat("0", 64)}
	r.want(t, candidate, stable)
	done := make(chan struct{})
	go func() {
		r.s.reconcile(context.Background())
		close(done)
	}()
	<-entered
	ready := r.s.snapshot().Ready
	close(release)
	<-done
	if !ready {
		t.Fatal("a hung candidate kept the replica unready")
	}
}

func TestLinksInAnArchiveAreRefused(t *testing.T) {
	var buffer bytes.Buffer
	zipped := gzip.NewWriter(&buffer)
	writer := tar.NewWriter(zipped)
	writer.WriteHeader(&tar.Header{Name: "adapter_config.json", Typeflag: tar.TypeSymlink, Linkname: "/etc/passwd"})
	writer.Close()
	zipped.Close()
	r := newRig(t)
	r.want(t, r.adapter(t, "stable", true, buffer.Bytes()))
	r.s.reconcile(context.Background())
	if got := r.s.snapshot(); got.Ready || r.fake.loads != 0 {
		t.Fatalf("%+v", got)
	}
}

func TestAFailureWaitsBeforeRetrying(t *testing.T) {
	r := newRig(t)
	r.s.retry = time.Hour
	a := r.adapter(t, "stable", true, goodArchive(t))
	a.SHA256 = strings.Repeat("0", 64)
	r.want(t, a)
	r.s.reconcile(context.Background())
	r.s.reconcile(context.Background())
	if r.fake.downloads != 1 {
		t.Fatalf("downloads %d", r.fake.downloads)
	}
	if got := r.s.snapshot(); got.Adapters[0].Error != "checksum mismatch" {
		t.Fatalf("%+v", got)
	}
}

func TestUnsafeArchivesAreRefused(t *testing.T) {
	for name, files := range map[string]map[string]string{
		"traversal": {"../adapter_config.json": "{}"},
		"nested":    {"sub/adapter_config.json": "{}"},
		"no config": {"adapter_model.safetensors": "weights"},
	} {
		t.Run(name, func(t *testing.T) {
			r := newRig(t)
			r.want(t, r.adapter(t, "stable", true, archiveOf(t, files)))
			r.s.reconcile(context.Background())
			if got := r.s.snapshot(); got.Ready || r.fake.loads != 0 {
				t.Fatalf("%+v", got)
			}
		})
	}
}

func TestBadEntriesInTheDesiredFileAreRefused(t *testing.T) {
	good := strings.Repeat("a", 64)
	for _, a := range []adapter{
		{Name: "../x", URL: "http://x/y", SHA256: good},
		{Name: "ok", URL: "file:///etc/passwd", SHA256: good},
		{Name: "ok", URL: "http://x/y", SHA256: "short"},
	} {
		raw, _ := json.Marshal(map[string]any{"adapters": []adapter{a}})
		file := filepath.Join(t.TempDir(), "desired.json")
		os.WriteFile(file, raw, 0o644)
		if _, err := readDesired(file); err == nil {
			t.Fatalf("accepted %+v", a)
		}
	}
}

func TestReadinessFollowsTheState(t *testing.T) {
	r := newRig(t)
	ts := httptest.NewServer(r.s.routes())
	defer ts.Close()
	code := func(url string) int {
		response, err := http.Get(url)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		return response.StatusCode
	}
	if code(ts.URL+"/readyz") != http.StatusServiceUnavailable || code(ts.URL+"/healthz") != http.StatusOK {
		t.Fatal("ready before the first sync")
	}
	r.want(t, r.adapter(t, "stable", true, goodArchive(t)))
	r.s.reconcile(context.Background())
	if code(ts.URL+"/readyz") != http.StatusOK || code(ts.URL+"/state") != http.StatusOK {
		t.Fatal("not ready after syncing")
	}
}
