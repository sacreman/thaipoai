// Command rollout-controller moves a target to a new model through verify, golden suite, paired
// shadow and canary. The decisions are pure functions in internal/rollout. This file only reads
// the world and writes the result, and every step can be repeated after a crash.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	"github.com/sacreman/thaipoai/internal/guards"
	"github.com/sacreman/thaipoai/internal/llm"
	"github.com/sacreman/thaipoai/internal/rollout"
	"github.com/sacreman/thaipoai/internal/rollout/api"
)

const (
	finalizer       = "thaipoai.dev/publish-stable"
	maxHistory      = 10
	modelPort       = 8000
	gatewayPort     = 8088
	defaultPercent  = 10
	defaultSamples  = 100
	defaultMinutes  = 60
	transitionDelay = 10 * time.Millisecond
)

// testCase is a line of a suite. Suites are synthetic, and no live message is ever replayed.
type testCase struct {
	Direction     string   `json:"direction"`
	Source        string   `json:"source"`
	Context       []string `json:"context"`
	SpeakerGender string   `json:"speaker_gender"`
}

type reconciler struct {
	client.Client
	reader        client.Reader // reads the API server directly, so a phase never reruns on a stale copy
	namespace     string
	base          string
	golden        []testCase
	replay        []testCase
	verifyTimeout time.Duration
	poll          time.Duration
	now           func() time.Time
	chat          func(ctx context.Context, model string, messages []llm.Message) (string, error)
	fetch         func(ctx context.Context, url string, into any) error
}

func canaryOf(spec api.Spec) (percent int, samples int64, timeout time.Duration) {
	percent, samples, minutes := spec.Canary.Percent, int64(spec.Canary.MinSamples), spec.Canary.TimeoutMinutes
	if percent == 0 {
		percent = defaultPercent
	}
	if samples == 0 {
		samples = defaultSamples
	}
	if minutes == 0 {
		minutes = defaultMinutes
	}
	return percent, samples, time.Duration(minutes) * time.Minute
}

func configMapName(mr *api.ModelRollout) string { return "rollout-" + mr.Name }

// recoverStable reads the stable model from the published list. It lets a rollout object that
// was deleted and created again carry on from what is really being served.
func (r *reconciler) recoverStable(ctx context.Context, mr *api.ModelRollout) (*rollout.Model, error) {
	var configMap corev1.ConfigMap
	if err := r.reader.Get(ctx, client.ObjectKey{Name: configMapName(mr), Namespace: mr.Namespace}, &configMap); err != nil {
		return nil, client.IgnoreNotFound(err)
	}
	var published struct {
		Models []rollout.Desired `json:"models"`
	}
	if err := json.Unmarshal([]byte(configMap.Data["desired.json"]), &published); err != nil {
		return nil, err
	}
	for _, entry := range published.Models {
		if entry.Required {
			model := entry.Model
			return &model, nil
		}
	}
	return nil, nil
}

// apply publishes what the current status says should be deployed.
func (r *reconciler) apply(ctx context.Context, mr *api.ModelRollout) error {
	percent, _, _ := canaryOf(mr.Spec)
	desired, routing := rollout.Plan(rollout.PlanInput{
		Phase: mr.Status.Phase, Stable: mr.Status.Stable, Previous: mr.Status.Previous, Candidate: mr.Spec.Model,
		Percent: percent, Base: r.base, ID: mr.Spec.ID(),
	})
	desiredJSON, err := json.Marshal(map[string]any{"models": desired})
	if err != nil {
		return err
	}
	routingJSON, err := json.Marshal(routing)
	if err != nil {
		return err
	}
	configMap := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: configMapName(mr), Namespace: mr.Namespace}}
	// No owner reference: if the rollout object is deleted, serving must keep its last weights.
	_, err = controllerutil.CreateOrUpdate(ctx, r.Client, configMap, func() error {
		configMap.Data = map[string]string{"desired.json": string(desiredJSON), "routing.json": string(routingJSON)}
		return nil
	})
	return err
}

func (r *reconciler) running(ctx context.Context, app string) ([]corev1.Pod, error) {
	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(r.namespace), client.MatchingLabels{"app": app}); err != nil {
		return nil, err
	}
	var running []corev1.Pod
	for _, pod := range pods.Items {
		if pod.Status.Phase == corev1.PodRunning && pod.DeletionTimestamp == nil && pod.Status.PodIP != "" {
			running = append(running, pod)
		}
	}
	return running, nil
}

// serving counts the model replicas that answer for this exact name.
func (r *reconciler) serving(ctx context.Context, registered string) (ready, total int, err error) {
	pods, err := r.running(ctx, "model")
	if err != nil {
		return 0, 0, err
	}
	for _, pod := range pods {
		var served struct {
			ID string `json:"id"`
		}
		url := fmt.Sprintf("http://%s:%d/v1/models/%s", pod.Status.PodIP, modelPort, registered)
		if r.fetch(ctx, url, &served) == nil && served.ID == registered {
			ready++
		}
	}
	return ready, len(pods), nil
}

func (r *reconciler) passes(ctx context.Context, model string, c testCase) (bool, error) {
	target := "Thai"
	if c.Direction == "th-en" {
		target = "English"
	}
	output, err := r.chat(ctx, model, llm.ChatMessages(target, c.Source, c.Context, llm.SpeakerNotes(c.SpeakerGender, "")))
	if err != nil {
		return false, err
	}
	failed := guards.Check(guards.Request{Source: c.Source, Output: output, TargetThai: target == "Thai", SpeakerGender: c.SpeakerGender})
	return len(guards.Hard(failed)) == 0, nil
}

// compare translates every case with both models. It is deterministic, so after a crash the
// phase simply runs it again.
func (r *reconciler) compare(ctx context.Context, mr *api.ModelRollout, cases []testCase) rollout.Tally {
	stable := r.base
	if mr.Status.Stable != nil {
		stable = mr.Status.Stable.Registered()
	}
	pairs := make([]rollout.Pair, 0, len(cases))
	for _, c := range cases {
		stablePass, err1 := r.passes(ctx, stable, c)
		candidatePass, err2 := r.passes(ctx, mr.Spec.Model.Registered(), c)
		pairs = append(pairs, rollout.Pair{StablePass: stablePass, CandidatePass: candidatePass, Error: err1 != nil || err2 != nil})
	}
	return rollout.Count(pairs)
}

// counts sums per-model numbers from gateways already on this rollout's routing. A gateway
// that restarted has simply lost its share, so the canary waits for more traffic.
func (r *reconciler) counts(ctx context.Context, mr *api.ModelRollout) (api.CanaryResult, error) {
	pods, err := r.running(ctx, "gateway")
	if err != nil {
		return api.CanaryResult{}, err
	}
	stable := r.base
	if mr.Status.Stable != nil {
		stable = mr.Status.Stable.Registered()
	}
	var result api.CanaryResult
	for _, pod := range pods {
		var stats struct {
			RolloutID string                   `json:"rollout_id"`
			Models    map[string]rollout.Stats `json:"models"`
		}
		if err := r.fetch(ctx, fmt.Sprintf("http://%s:%d/stats", pod.Status.PodIP, gatewayPort), &stats); err != nil || stats.RolloutID != mr.Spec.ID() {
			continue
		}
		result.Stable.Add(stats.Models[stable])
		result.Candidate.Add(stats.Models[mr.Spec.Model.Registered()])
	}
	return result, nil
}

func (r *reconciler) step(ctx context.Context, mr *api.ModelRollout) (rollout.Decision, string, error) {
	status := &mr.Status
	elapsed := r.now().Sub(status.PhaseStartedAt.Time)
	switch status.Phase {
	case rollout.Verifying:
		ready, total, err := r.serving(ctx, mr.Spec.Model.Registered())
		if err != nil {
			return rollout.Wait, "", err
		}
		decision, reason := rollout.EvaluateVerify(ready, total, elapsed, r.verifyTimeout)
		return decision, reason, nil
	case rollout.Golden:
		tally := r.compare(ctx, mr, r.golden)
		status.Golden = &tally
		decision, reason := rollout.EvaluateGolden(tally)
		return decision, reason, nil
	case rollout.Shadow:
		tally := r.compare(ctx, mr, r.replay)
		status.Shadow = &tally
		decision, reason := rollout.EvaluateShadow(tally)
		return decision, reason, nil
	case rollout.Canary:
		counts, err := r.counts(ctx, mr)
		if err != nil {
			return rollout.Wait, "", err
		}
		status.Canary = &counts
		_, samples, timeout := canaryOf(mr.Spec)
		decision, reason := rollout.EvaluateCanary(counts.Stable, counts.Candidate, samples, elapsed, timeout)
		return decision, reason, nil
	}
	return rollout.Wait, "", nil
}

// write saves status. A conflict means this reconcile read an old copy, so it is safely retried.
func (r *reconciler) write(ctx context.Context, mr *api.ModelRollout, after time.Duration) (ctrl.Result, error) {
	if err := r.Status().Update(ctx, mr); err != nil {
		if apierrors.IsConflict(err) {
			return ctrl.Result{RequeueAfter: r.poll}, nil
		}
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: after}, nil
}

func (r *reconciler) Reconcile(ctx context.Context, request ctrl.Request) (ctrl.Result, error) {
	var mr api.ModelRollout
	if err := r.reader.Get(ctx, request.NamespacedName, &mr); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	status := &mr.Status
	if !mr.DeletionTimestamp.IsZero() {
		return r.finalize(ctx, &mr)
	}
	if controllerutil.AddFinalizer(&mr, finalizer) {
		if err := r.Update(ctx, &mr); err != nil {
			return ctrl.Result{RequeueAfter: r.poll}, client.IgnoreNotFound(err)
		}
		return ctrl.Result{RequeueAfter: transitionDelay}, nil
	}
	if status.Observed != mr.Spec.Model.SHA256 || status.ObservedAttempt != mr.Spec.Attempt {
		if status.Observed == "" && status.Stable == nil {
			recovered, err := r.recoverStable(ctx, &mr)
			if err != nil {
				return ctrl.Result{}, err
			}
			status.Stable = recovered
		}
		now := metav1.NewTime(r.now())
		status.Observed, status.ObservedAttempt = mr.Spec.Model.SHA256, mr.Spec.Attempt
		status.Phase, status.Reason, status.PhaseStartedAt = rollout.Verifying, "", &now
		status.Previous, status.Golden, status.Shadow, status.Canary = nil, nil, nil, nil
		return r.write(ctx, &mr, transitionDelay)
	}
	if err := r.apply(ctx, &mr); err != nil {
		return ctrl.Result{}, err
	}
	if rollout.Terminal(status.Phase) {
		return ctrl.Result{}, nil
	}
	phase := status.Phase
	decision, reason, err := r.step(ctx, &mr)
	if err != nil {
		return ctrl.Result{}, err
	}
	now := metav1.NewTime(r.now()) // read after step: a replay can take longer than the next phase's timeout
	next := rollout.Next(phase, decision)
	if next == phase {
		if phase == rollout.Canary {
			return r.write(ctx, &mr, r.poll) // keeps the live counts visible on the object
		}
		return ctrl.Result{RequeueAfter: r.poll}, nil
	}
	if next == rollout.Promoted {
		model := mr.Spec.Model
		status.Previous, status.Stable = status.Stable, &model // one update, so it cannot be half done
	}
	if rollout.Terminal(next) {
		status.History = append(status.History, api.Outcome{Model: mr.Spec.Model.Registered(), Phase: next, Reason: reason, At: now})
		if extra := len(status.History) - maxHistory; extra > 0 {
			status.History = status.History[extra:]
		}
	}
	status.Phase, status.Reason, status.PhaseStartedAt = next, reason, &now
	return r.write(ctx, &mr, transitionDelay)
}

// finalize returns all traffic to stable when a rollout is deleted, so a canary cannot outlive
// the object that was governing it.
func (r *reconciler) finalize(ctx context.Context, mr *api.ModelRollout) (ctrl.Result, error) {
	if mr.Status.Observed != "" {
		ended := mr.DeepCopyObject().(*api.ModelRollout)
		ended.Status.Phase = rollout.Inconclusive
		if err := r.apply(ctx, ended); err != nil {
			return ctrl.Result{}, err
		}
	}
	if controllerutil.RemoveFinalizer(mr, finalizer) {
		return ctrl.Result{}, client.IgnoreNotFound(r.Update(ctx, mr))
	}
	return ctrl.Result{}, nil
}

func readCases(file string) ([]testCase, error) {
	raw, err := os.ReadFile(file)
	if err != nil {
		return nil, err
	}
	var cases []testCase
	for _, line := range strings.Split(string(raw), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var c testCase
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			return nil, err
		}
		cases = append(cases, c)
	}
	return cases, nil
}

func fetcher(httpClient *http.Client) func(context.Context, string, any) error {
	return func(ctx context.Context, url string, into any) error {
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		response, err := httpClient.Do(request)
		if err != nil {
			return err
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			return fmt.Errorf("status %d", response.StatusCode)
		}
		return json.NewDecoder(response.Body).Decode(into)
	}
}

func main() {
	namespace := flag.String("namespace", "default", "namespace to work in")
	base := flag.String("base-model", "base", "base model, used as stable until the first promotion")
	modelURL := flag.String("model-url", "http://model:8000", "model server Service")
	goldenFile := flag.String("golden", "/suites/golden.jsonl", "must-pass cases")
	replayFile := flag.String("replay", "/suites/replay.jsonl", "paired replay corpus")
	verifyTimeout := flag.Duration("verify-timeout", 10*time.Minute, "time allowed for every replica to hold the candidate")
	poll := flag.Duration("poll", 3*time.Second, "time between checks")
	flag.Parse()
	ctrl.SetLogger(zap.New())
	golden, err := readCases(*goldenFile)
	if err != nil {
		ctrl.Log.Error(err, "cannot read golden suite")
		os.Exit(1)
	}
	replay, err := readCases(*replayFile)
	if err != nil {
		ctrl.Log.Error(err, "cannot read replay corpus")
		os.Exit(1)
	}
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		panic(err)
	}
	if err := api.AddToScheme(scheme); err != nil {
		panic(err)
	}
	// ponytail: one replica and no leader election. A restart resumes from status; two replicas need election.
	manager, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme:  scheme,
		Metrics: metricsserver.Options{BindAddress: "0"},
		Cache:   cache.Options{DefaultNamespaces: map[string]cache.Config{*namespace: {}}},
	})
	if err != nil {
		ctrl.Log.Error(err, "cannot start manager")
		os.Exit(1)
	}
	modelClient := &http.Client{Timeout: 5 * time.Minute}
	r := &reconciler{
		Client: manager.GetClient(), reader: manager.GetAPIReader(), namespace: *namespace, base: *base, golden: golden, replay: replay,
		verifyTimeout: *verifyTimeout, poll: *poll, now: time.Now,
		chat: func(ctx context.Context, model string, messages []llm.Message) (string, error) {
			return llm.Client{Endpoint: strings.TrimRight(*modelURL, "/"), Model: model, HTTP: modelClient}.Chat(ctx, messages)
		},
		fetch: fetcher(&http.Client{Timeout: 5 * time.Second}),
	}
	// Status writes do not wake the controller. It schedules its own next look.
	err = ctrl.NewControllerManagedBy(manager).
		For(&api.ModelRollout{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Complete(r)
	if err != nil {
		ctrl.Log.Error(err, "cannot build controller")
		os.Exit(1)
	}
	if err := manager.Start(ctrl.SetupSignalHandler()); err != nil {
		ctrl.Log.Error(err, "manager stopped")
		os.Exit(1)
	}
}
