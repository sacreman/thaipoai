package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/sacreman/thaipoai/internal/llm"
	"github.com/sacreman/thaipoai/internal/rollout"
	"github.com/sacreman/thaipoai/internal/rollout/api"
)

var (
	stable    = rollout.Adapter{Name: "stable", URL: "http://s/a", SHA256: strings.Repeat("a", 64)}
	candidate = rollout.Adapter{Name: "cand", URL: "http://s/c", SHA256: strings.Repeat("c", 64)}
)

type world struct {
	t       *testing.T
	r       *reconciler
	clock   time.Time
	replica string
	broken  map[string]bool // adapters that give bad translations
	down    bool            // the model server is unreachable
	perChat time.Duration   // how long each translation takes
	attempt int             // the attempt the replicas report
	stats   map[string]any
	chats   int
}

func newWorld(t *testing.T) *world {
	t.Helper()
	scheme := runtime.NewScheme()
	clientgoscheme.AddToScheme(scheme)
	api.AddToScheme(scheme)
	running := func(name, app, ip string) *corev1.Pod {
		return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default", Labels: map[string]string{"app": app}}, Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: ip}}
	}
	ar := &api.AdapterRollout{
		ObjectMeta: metav1.ObjectMeta{Name: "translate", Namespace: "default"},
		Spec:       api.Spec{Adapter: candidate, Canary: api.CanarySpec{Percent: 10, MinSamples: 100, TimeoutMinutes: 30}},
		Status:     api.Status{Stable: &stable},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(ar, running("m1", "model", "1.1.1.1"), running("g1", "gateway", "2.2.2.2")).WithStatusSubresource(&api.AdapterRollout{}).Build()
	w := &world{t: t, clock: time.Unix(1000000, 0), replica: "ready", broken: map[string]bool{}}
	w.stats = healthyStats()
	w.r = &reconciler{
		Client: c, reader: c, namespace: "default", base: "base", verifyTimeout: time.Minute, poll: time.Second,
		golden: []testCase{{Direction: "en-th", Source: "a"}, {Direction: "en-th", Source: "b"}, {Direction: "en-th", Source: "c"}},
		replay: []testCase{{Direction: "en-th", Source: "a"}},
		now:    func() time.Time { return w.clock },
		chat: func(_ context.Context, model string, _ []llm.Message) (string, error) {
			w.chats++
			w.clock = w.clock.Add(w.perChat)
			if w.down {
				return "", errors.New("connection refused")
			}
			if w.broken[model] {
				return "not thai", nil
			}
			return "ไทย", nil
		},
		fetch: func(_ context.Context, url string, into any) error {
			var body any
			switch {
			case strings.HasSuffix(url, "/state"):
				body = map[string]any{"adapters": []map[string]any{{"registered": candidate.Registered(), "state": w.replica, "error": w.replicaError(), "attempt": w.attempt}}}
			default:
				body = w.stats
			}
			raw, _ := json.Marshal(body)
			return json.Unmarshal(raw, into)
		},
	}
	return w
}

func (w *world) replicaError() string {
	if w.replica == "mismatch" {
		return "checksum mismatch"
	}
	return ""
}

func healthyStats() map[string]any {
	return map[string]any{"rollout_id": candidate.SHA256 + "/0", "adapters": map[string]rollout.Stats{
		stable.Registered():    {Requests: 1000, GuardFailures: 10, SourceChars: 10000, OutputChars: 12000},
		candidate.Registered(): {Requests: 200, GuardFailures: 2, SourceChars: 2000, OutputChars: 2400},
	}}
}

func (w *world) get() *api.AdapterRollout {
	w.t.Helper()
	var ar api.AdapterRollout
	if err := w.r.Get(context.Background(), types.NamespacedName{Name: "translate", Namespace: "default"}, &ar); err != nil {
		w.t.Fatal(err)
	}
	return &ar
}

func (w *world) reconcile() ctrl.Result {
	w.t.Helper()
	result, err := w.r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Name: "translate", Namespace: "default"}})
	if err != nil {
		w.t.Fatal(err)
	}
	return result
}

// until reconciles until the phase is reached, or fails after too many tries.
func (w *world) until(want rollout.Phase) *api.AdapterRollout {
	w.t.Helper()
	for i := 0; i < 20; i++ {
		w.reconcile()
		if ar := w.get(); ar.Status.Phase == want {
			return ar
		}
	}
	w.t.Fatalf("never reached %s, stuck at %+v", want, w.get().Status)
	return nil
}

func (w *world) routing() rollout.Routing {
	w.t.Helper()
	var cm corev1.ConfigMap
	if err := w.r.Get(context.Background(), types.NamespacedName{Name: "rollout-translate", Namespace: "default"}, &cm); err != nil {
		w.t.Fatal(err)
	}
	var route rollout.Routing
	json.Unmarshal([]byte(cm.Data["routing.json"]), &route)
	return route
}

func TestAGoodCandidateIsPromotedAndBecomesStable(t *testing.T) {
	w := newWorld(t)
	ar := w.until(rollout.Promoted)
	if ar.Status.Stable.Name != "cand" || len(ar.Status.History) != 1 || ar.Status.History[0].Phase != rollout.Promoted {
		t.Fatalf("%+v", ar.Status)
	}
	w.reconcile()
	if route := w.routing(); route.Stable != candidate.Registered() || route.Candidate != "" {
		t.Fatalf("%+v", route)
	}
}

func TestACanaryGetsTrafficOnlyInCanary(t *testing.T) {
	w := newWorld(t)
	w.stats = map[string]any{"rollout_id": "other", "adapters": map[string]rollout.Stats{}} // no samples yet
	w.until(rollout.Shadow)
	w.reconcile()
	if route := w.routing(); route.Candidate != "" {
		t.Fatalf("traffic before canary: %+v", route)
	}
	w.until(rollout.Canary)
	w.reconcile()
	if route := w.routing(); route.Candidate != candidate.Registered() || route.CandidatePercent != 10 {
		t.Fatalf("%+v", route)
	}
}

func TestAChecksumMismatchBlocksWithoutTraffic(t *testing.T) {
	w := newWorld(t)
	w.replica = "mismatch"
	ar := w.until(rollout.Blocked)
	if ar.Status.Stable.Name != "stable" || w.chats != 0 {
		t.Fatalf("%+v chats %d", ar.Status, w.chats)
	}
}

func TestALoadFailureOnAReplicaIsInconclusive(t *testing.T) {
	w := newWorld(t)
	w.replica = "failed"
	w.reconcile()
	w.reconcile()
	w.clock = w.clock.Add(2 * time.Minute)
	ar := w.until(rollout.Inconclusive)
	if ar.Status.Stable.Name != "stable" {
		t.Fatalf("%+v", ar.Status)
	}
}

func TestABadCandidateIsBlockedAtTheGoldenGate(t *testing.T) {
	w := newWorld(t)
	w.broken[candidate.Registered()] = true
	ar := w.until(rollout.Blocked)
	if ar.Status.Golden == nil || ar.Status.Golden.Worse != 3 || ar.Status.Shadow != nil {
		t.Fatalf("%+v", ar.Status)
	}
}

func TestAModelOutageDuringGoldenIsInconclusive(t *testing.T) {
	w := newWorld(t)
	w.until(rollout.Golden)
	w.down = true
	w.until(rollout.Inconclusive)
}

func TestBadCanaryNumbersRollBack(t *testing.T) {
	w := newWorld(t)
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "adapters": map[string]rollout.Stats{
		stable.Registered():    {Requests: 1000, GuardFailures: 10, SourceChars: 10000, OutputChars: 12000},
		candidate.Registered(): {Requests: 200, GuardFailures: 2, SourceChars: 2000, OutputChars: 7000},
	}}
	ar := w.until(rollout.RolledBack)
	if ar.Status.Stable.Name != "stable" {
		t.Fatalf("%+v", ar.Status)
	}
	w.reconcile()
	if route := w.routing(); route.Stable != stable.Registered() || route.Candidate != "" {
		t.Fatalf("%+v", route)
	}
}

func TestACanaryOutageIsInconclusiveNotARollback(t *testing.T) {
	w := newWorld(t)
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "adapters": map[string]rollout.Stats{
		stable.Registered():    {Requests: 1000, SourceChars: 10000, OutputChars: 12000},
		candidate.Registered(): {Requests: 10, GuardFailures: 10, SourceChars: 100, OutputChars: 900, InfraErrors: 90},
	}}
	w.until(rollout.Inconclusive)
}

func TestTooLittleTrafficIsInconclusive(t *testing.T) {
	w := newWorld(t)
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "adapters": map[string]rollout.Stats{}}
	w.until(rollout.Canary)
	w.reconcile()
	w.clock = w.clock.Add(time.Hour)
	w.until(rollout.Inconclusive)
}

func TestGatewaysOnOldRoutingAreIgnored(t *testing.T) {
	w := newWorld(t)
	stale := healthyStats()
	stale["rollout_id"] = "previous/0"
	w.stats = stale
	w.until(rollout.Canary)
	w.reconcile()
	if ar := w.get(); ar.Status.Phase != rollout.Canary || ar.Status.Canary.Candidate.Requests != 0 {
		t.Fatalf("%+v", ar.Status)
	}
}

func TestAControllerRestartMidCanaryResumes(t *testing.T) {
	w := newWorld(t)
	good := w.stats
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "adapters": map[string]rollout.Stats{}}
	w.until(rollout.Canary)
	w.reconcile()
	before := w.routing()
	restarted := *w.r // a new process has nothing but what is in the API server
	w.r = &restarted
	w.stats = good
	if w.routing() != before {
		t.Fatal("routing changed while the controller was down")
	}
	w.until(rollout.Promoted)
}

func TestReapplyingIsIdempotentAfterACrashBeforeApply(t *testing.T) {
	w := newWorld(t)
	ar := w.until(rollout.Promoted)
	var cm corev1.ConfigMap
	key := client.ObjectKey{Name: "rollout-translate", Namespace: "default"}
	if err := w.r.Get(context.Background(), key, &cm); err == nil {
		w.r.Delete(context.Background(), &cm) // the crash happened before apply ran
	}
	w.reconcile()
	if route := w.routing(); route.Stable != candidate.Registered() {
		t.Fatalf("%+v after %s", route, ar.Status.Phase)
	}
}

func TestANewSpecStartsAgainAndKeepsStable(t *testing.T) {
	w := newWorld(t)
	w.broken[candidate.Registered()] = true
	w.until(rollout.Blocked)
	delete(w.broken, candidate.Registered())
	ar := w.get()
	ar.Spec.Attempt = 1
	if err := w.r.Update(context.Background(), ar); err != nil {
		t.Fatal(err)
	}
	w.attempt = 1
	w.stats["rollout_id"] = candidate.SHA256 + "/1"
	got := w.until(rollout.Promoted)
	if len(got.Status.History) != 2 || got.Status.ObservedAttempt != 1 {
		t.Fatalf("%+v", got.Status)
	}
}

func TestTheFirstRolloutComparesAgainstTheBaseModel(t *testing.T) {
	w := newWorld(t)
	ar := w.get()
	ar.Status.Stable = nil
	if err := w.r.Status().Update(context.Background(), ar); err != nil {
		t.Fatal(err)
	}
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "adapters": map[string]rollout.Stats{
		"base":                 {Requests: 1000, SourceChars: 10000, OutputChars: 12000},
		candidate.Registered(): {Requests: 200, SourceChars: 2000, OutputChars: 2400},
	}}
	got := w.until(rollout.Promoted)
	if got.Status.Stable.Name != "cand" {
		t.Fatalf("%+v", got.Status)
	}
}

func TestASlowReplayDoesNotUseUpTheCanaryTimeout(t *testing.T) {
	w := newWorld(t)
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "adapters": map[string]rollout.Stats{}}
	w.perChat = time.Hour
	w.until(rollout.Canary)
	w.reconcile()
	if ar := w.get(); ar.Status.Phase != rollout.Canary {
		t.Fatalf("%+v", ar.Status)
	}
}

func TestDeletingARolloutMidCanaryReturnsTrafficToStable(t *testing.T) {
	w := newWorld(t)
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "adapters": map[string]rollout.Stats{}}
	w.until(rollout.Canary)
	w.reconcile()
	if w.routing().Candidate == "" {
		t.Fatal("no canary to end")
	}
	if err := w.r.Delete(context.Background(), w.get()); err != nil {
		t.Fatal(err)
	}
	w.reconcile()
	if route := w.routing(); route.Candidate != "" || route.Stable != stable.Registered() {
		t.Fatalf("%+v", route)
	}
	var left api.AdapterRollout
	if err := w.r.Get(context.Background(), client.ObjectKey{Name: "translate", Namespace: "default"}, &left); err == nil {
		t.Fatal("finalizer was not removed")
	}
}

func TestARecreatedRolloutCarriesOnFromWhatIsServed(t *testing.T) {
	w := newWorld(t)
	w.until(rollout.Promoted)
	w.reconcile()
	if err := w.r.Delete(context.Background(), w.get()); err != nil {
		t.Fatal(err)
	}
	w.reconcile()
	again := &api.AdapterRollout{ObjectMeta: metav1.ObjectMeta{Name: "translate", Namespace: "default"}, Spec: api.Spec{Adapter: stable}}
	if err := w.r.Create(context.Background(), again); err != nil {
		t.Fatal(err)
	}
	w.reconcile()
	w.reconcile()
	if got := w.get(); got.Status.Stable == nil || got.Status.Stable.Name != "cand" {
		t.Fatalf("%+v", got.Status)
	}
}

func TestAPromotionKeepsTheOldStableUntilTheNextRollout(t *testing.T) {
	w := newWorld(t)
	w.until(rollout.Promoted)
	w.reconcile()
	read := func() string {
		var cm corev1.ConfigMap
		w.r.Get(context.Background(), client.ObjectKey{Name: "rollout-translate", Namespace: "default"}, &cm)
		return cm.Data["desired.json"]
	}
	if !strings.Contains(read(), stable.SHA256) {
		t.Fatalf("old stable dropped: %s", read())
	}
	ar := w.get()
	ar.Spec.Attempt = 1
	if err := w.r.Update(context.Background(), ar); err != nil {
		t.Fatal(err)
	}
	w.reconcile()
	w.reconcile()
	if strings.Contains(read(), stable.SHA256) {
		t.Fatalf("old stable kept too long: %s", read())
	}
}

func TestAReplicaStateFromAnotherAttemptIsIgnored(t *testing.T) {
	w := newWorld(t)
	w.replica = "mismatch"
	ar := w.get()
	ar.Spec.Attempt = 3 // the fake replica reports attempt 0, the stale result of an earlier try
	if err := w.r.Update(context.Background(), ar); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		w.reconcile()
	}
	if got := w.get(); got.Status.Phase != rollout.Verifying {
		t.Fatalf("%+v", got.Status)
	}
}
