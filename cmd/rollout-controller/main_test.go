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
	stable    = rollout.Model{Name: "stable", SHA256: strings.Repeat("a", 64)}
	candidate = rollout.Model{Name: "cand", SHA256: strings.Repeat("c", 64)}
)

type world struct {
	t       *testing.T
	r       *reconciler
	clock   time.Time
	missing bool            // the replica does not serve the candidate
	broken  map[string]bool // models that give bad translations
	down    bool            // the model server is unreachable
	perChat time.Duration   // how long each translation takes
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
	mr := &api.ModelRollout{
		ObjectMeta: metav1.ObjectMeta{Name: "translate", Namespace: "default"},
		Spec:       api.Spec{Model: candidate, Canary: api.CanarySpec{Percent: 10, MinSamples: 100, TimeoutMinutes: 30}},
		Status:     api.Status{Stable: &stable},
	}
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(mr, running("m1", "model", "1.1.1.1"), running("g1", "gateway", "2.2.2.2")).WithStatusSubresource(&api.ModelRollout{}).Build()
	w := &world{t: t, clock: time.Unix(1000000, 0), broken: map[string]bool{}}
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
			var body any = w.stats
			if strings.Contains(url, "/v1/models/") {
				if w.missing {
					return errors.New("status 404")
				}
				body = map[string]string{"id": candidate.Registered()}
			}
			raw, _ := json.Marshal(body)
			return json.Unmarshal(raw, into)
		},
	}
	return w
}

func healthyStats() map[string]any {
	return map[string]any{"rollout_id": candidate.SHA256 + "/0", "models": map[string]rollout.Stats{
		stable.Registered():    {Requests: 1000, GuardFailures: 10, SourceChars: 10000, OutputChars: 12000},
		candidate.Registered(): {Requests: 200, GuardFailures: 2, SourceChars: 2000, OutputChars: 2400},
	}}
}

func (w *world) get() *api.ModelRollout {
	w.t.Helper()
	var mr api.ModelRollout
	if err := w.r.Get(context.Background(), types.NamespacedName{Name: "translate", Namespace: "default"}, &mr); err != nil {
		w.t.Fatal(err)
	}
	return &mr
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
func (w *world) until(want rollout.Phase) *api.ModelRollout {
	w.t.Helper()
	for i := 0; i < 20; i++ {
		w.reconcile()
		if mr := w.get(); mr.Status.Phase == want {
			return mr
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
	mr := w.until(rollout.Promoted)
	if mr.Status.Stable.Name != "cand" || len(mr.Status.History) != 1 || mr.Status.History[0].Phase != rollout.Promoted {
		t.Fatalf("%+v", mr.Status)
	}
	w.reconcile()
	if route := w.routing(); route.Stable != candidate.Registered() || route.Candidate != "" {
		t.Fatalf("%+v", route)
	}
}

func TestACanaryGetsTrafficOnlyInCanary(t *testing.T) {
	w := newWorld(t)
	w.stats = map[string]any{"rollout_id": "other", "models": map[string]rollout.Stats{}} // no samples yet
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

func TestACandidateAReplicaDoesNotServeIsInconclusiveWithoutTraffic(t *testing.T) {
	w := newWorld(t)
	w.missing = true
	w.reconcile()
	w.reconcile()
	w.clock = w.clock.Add(2 * time.Minute)
	mr := w.until(rollout.Inconclusive)
	if mr.Status.Stable.Name != "stable" || w.chats != 0 {
		t.Fatalf("%+v chats %d", mr.Status, w.chats)
	}
}

func TestABadCandidateIsBlockedAtTheGoldenGate(t *testing.T) {
	w := newWorld(t)
	w.broken[candidate.Registered()] = true
	mr := w.until(rollout.Blocked)
	if mr.Status.Golden == nil || mr.Status.Golden.Worse != 3 || mr.Status.Shadow != nil {
		t.Fatalf("%+v", mr.Status)
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
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "models": map[string]rollout.Stats{
		stable.Registered():    {Requests: 1000, GuardFailures: 10, SourceChars: 10000, OutputChars: 12000},
		candidate.Registered(): {Requests: 200, GuardFailures: 2, SourceChars: 2000, OutputChars: 7000},
	}}
	mr := w.until(rollout.RolledBack)
	if mr.Status.Stable.Name != "stable" {
		t.Fatalf("%+v", mr.Status)
	}
	w.reconcile()
	if route := w.routing(); route.Stable != stable.Registered() || route.Candidate != "" {
		t.Fatalf("%+v", route)
	}
}

func TestACanaryOutageIsInconclusiveNotARollback(t *testing.T) {
	w := newWorld(t)
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "models": map[string]rollout.Stats{
		stable.Registered():    {Requests: 1000, SourceChars: 10000, OutputChars: 12000},
		candidate.Registered(): {Requests: 10, GuardFailures: 10, SourceChars: 100, OutputChars: 900, InfraErrors: 90},
	}}
	w.until(rollout.Inconclusive)
}

func TestTooLittleTrafficIsInconclusive(t *testing.T) {
	w := newWorld(t)
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "models": map[string]rollout.Stats{}}
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
	if mr := w.get(); mr.Status.Phase != rollout.Canary || mr.Status.Canary.Candidate.Requests != 0 {
		t.Fatalf("%+v", mr.Status)
	}
}

func TestAControllerRestartMidCanaryResumes(t *testing.T) {
	w := newWorld(t)
	good := w.stats
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "models": map[string]rollout.Stats{}}
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
	mr := w.until(rollout.Promoted)
	var cm corev1.ConfigMap
	key := client.ObjectKey{Name: "rollout-translate", Namespace: "default"}
	if err := w.r.Get(context.Background(), key, &cm); err == nil {
		w.r.Delete(context.Background(), &cm) // the crash happened before apply ran
	}
	w.reconcile()
	if route := w.routing(); route.Stable != candidate.Registered() {
		t.Fatalf("%+v after %s", route, mr.Status.Phase)
	}
}

func TestANewSpecStartsAgainAndKeepsStable(t *testing.T) {
	w := newWorld(t)
	w.broken[candidate.Registered()] = true
	w.until(rollout.Blocked)
	delete(w.broken, candidate.Registered())
	mr := w.get()
	mr.Spec.Attempt = 1
	if err := w.r.Update(context.Background(), mr); err != nil {
		t.Fatal(err)
	}
	w.stats["rollout_id"] = candidate.SHA256 + "/1"
	got := w.until(rollout.Promoted)
	if len(got.Status.History) != 2 || got.Status.ObservedAttempt != 1 {
		t.Fatalf("%+v", got.Status)
	}
}

func TestTheFirstRolloutComparesAgainstTheBaseModel(t *testing.T) {
	w := newWorld(t)
	mr := w.get()
	mr.Status.Stable = nil
	if err := w.r.Status().Update(context.Background(), mr); err != nil {
		t.Fatal(err)
	}
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "models": map[string]rollout.Stats{
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
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "models": map[string]rollout.Stats{}}
	w.perChat = time.Hour
	w.until(rollout.Canary)
	w.reconcile()
	if mr := w.get(); mr.Status.Phase != rollout.Canary {
		t.Fatalf("%+v", mr.Status)
	}
}

func TestDeletingARolloutMidCanaryReturnsTrafficToStable(t *testing.T) {
	w := newWorld(t)
	w.stats = map[string]any{"rollout_id": candidate.SHA256 + "/0", "models": map[string]rollout.Stats{}}
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
	var left api.ModelRollout
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
	again := &api.ModelRollout{ObjectMeta: metav1.ObjectMeta{Name: "translate", Namespace: "default"}, Spec: api.Spec{Model: stable}}
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
	mr := w.get()
	mr.Spec.Attempt = 1
	if err := w.r.Update(context.Background(), mr); err != nil {
		t.Fatal(err)
	}
	w.reconcile()
	w.reconcile()
	if strings.Contains(read(), stable.SHA256) {
		t.Fatalf("old stable kept too long: %s", read())
	}
}
