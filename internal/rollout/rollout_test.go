package rollout

import (
	"math"
	"strings"
	"testing"
	"time"
)

var candidate = Model{Name: "cand", SHA256: strings.Repeat("c", 64)}
var stable = Model{Name: "stable", SHA256: strings.Repeat("a", 64)}

func TestAnInfrastructureFaultIsNeverARollback(t *testing.T) {
	for _, phase := range []Phase{Verifying, Golden, Shadow, Canary} {
		if got := Next(phase, GiveUp); got != Inconclusive {
			t.Fatalf("%s gave %s", phase, got)
		}
	}
}

func TestOnlyCanaryCanRollBack(t *testing.T) {
	for _, phase := range []Phase{Verifying, Golden, Shadow} {
		if got := Next(phase, Rollback); got != phase {
			t.Fatalf("%s moved to %s", phase, got)
		}
	}
	if Next(Canary, Rollback) != RolledBack {
		t.Fatal("canary did not roll back")
	}
}

func TestACandidateWithLiveTrafficCannotBeBlocked(t *testing.T) {
	if got := Next(Canary, Block); got != Canary {
		t.Fatalf("got %s", got)
	}
}

func TestTheHappyPath(t *testing.T) {
	phase := Verifying
	for _, want := range []Phase{Golden, Shadow, Canary, Promoted} {
		if phase = Next(phase, Advance); phase != want {
			t.Fatalf("got %s want %s", phase, want)
		}
	}
}

func TestTerminalPhasesStayTerminal(t *testing.T) {
	for _, phase := range []Phase{Promoted, Blocked, RolledBack, Inconclusive} {
		for _, decision := range []Decision{Wait, Advance, Block, Rollback, GiveUp} {
			if Next(phase, decision) != phase {
				t.Fatalf("%s moved on %d", phase, decision)
			}
		}
	}
}

func TestWaitNeverMoves(t *testing.T) {
	for _, phase := range []Phase{Verifying, Golden, Shadow, Canary} {
		if Next(phase, Wait) != phase {
			t.Fatalf("%s moved", phase)
		}
	}
}

func plan(phase Phase, stable *Model) ([]Desired, Routing) {
	return Plan(PlanInput{Phase: phase, Stable: stable, Candidate: candidate, Percent: 10, Base: "base", ID: "id"})
}

func TestPlanBeforeCanarySendsNoTrafficToTheCandidate(t *testing.T) {
	for _, phase := range []Phase{Verifying, Golden, Shadow} {
		desired, routing := plan(phase, &stable)
		if len(desired) != 2 || !desired[0].Required || desired[1].Required {
			t.Fatalf("%s desired %+v", phase, desired)
		}
		if routing.Stable != stable.Registered() || routing.Candidate != "" || routing.CandidatePercent != 0 {
			t.Fatalf("%s routing %+v", phase, routing)
		}
	}
}

func TestPlanDuringCanarySplitsTraffic(t *testing.T) {
	_, routing := plan(Canary, &stable)
	if routing.Candidate != candidate.Registered() || routing.CandidatePercent != 10 || routing.ID != "id" {
		t.Fatalf("%+v", routing)
	}
}

func TestPlanAfterAFailureReturnsToStableOnly(t *testing.T) {
	for _, phase := range []Phase{Blocked, RolledBack, Inconclusive} {
		desired, routing := plan(phase, &stable)
		if len(desired) != 1 || desired[0].Name != "stable" || routing.Candidate != "" || routing.ID != "" {
			t.Fatalf("%s %+v %+v", phase, desired, routing)
		}
	}
}

func TestPlanWithNoStableUsesTheBaseModel(t *testing.T) {
	desired, routing := plan(Shadow, nil)
	if len(desired) != 1 || desired[0].Required || routing.Stable != "base" {
		t.Fatalf("%+v %+v", desired, routing)
	}
}

func TestAPromotionKeepsThePreviousStableLoaded(t *testing.T) {
	previous := stable
	desired, routing := Plan(PlanInput{Phase: Promoted, Stable: &candidate, Previous: &previous, Base: "base"})
	if len(desired) != 2 || !desired[0].Required || desired[1].Name != "stable" || desired[1].Required {
		t.Fatalf("%+v", desired)
	}
	if routing.Stable != candidate.Registered() {
		t.Fatalf("%+v", routing)
	}
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestMcNemarP(t *testing.T) {
	cases := []struct {
		worse, better int
		want          float64
	}{{0, 0, 1}, {1, 0, 0.5}, {5, 0, 1.0 / 32}, {7, 0, 1.0 / 128}, {0, 5, 1}, {3, 3, 0.65625}}
	for _, c := range cases {
		if got := McNemarP(c.worse, c.better); !near(got, c.want) {
			t.Fatalf("%d,%d: got %v want %v", c.worse, c.better, got, c.want)
		}
	}
}

func TestFisherP(t *testing.T) {
	// The tea-tasting table, whose one-sided value is 17/70.
	if got := FisherP(3, 4, 1, 4); !near(got, 17.0/70) {
		t.Fatalf("got %v", got)
	}
	if got := FisherP(0, 10, 5, 10); !near(got, 1) {
		t.Fatalf("got %v", got)
	}
	if got := FisherP(1, 0, 0, 10); got != 1 {
		t.Fatalf("got %v", got)
	}
}

func TestVerify(t *testing.T) {
	const timeout = time.Minute
	cases := []struct {
		name         string
		ready, total int
		elapsed      time.Duration
		want         Decision
	}{
		{"all serving", 3, 3, 0, Advance},
		{"no replicas yet", 0, 0, 0, Wait},
		{"no replicas ever", 0, 0, 2 * timeout, GiveUp},
		{"one missing, still time", 2, 3, 0, Wait},
		{"one missing, out of time", 2, 3, 2 * timeout, GiveUp},
		{"none serving, still time", 0, 3, 0, Wait},
		{"none serving, out of time", 0, 3, 2 * timeout, GiveUp},
	}
	for _, c := range cases {
		if got, _ := EvaluateVerify(c.ready, c.total, c.elapsed, timeout); got != c.want {
			t.Fatalf("%s: got %d", c.name, got)
		}
	}
}

func TestCount(t *testing.T) {
	got := Count([]Pair{{StablePass: true}, {CandidatePass: true}, {StablePass: true, CandidatePass: true}, {}, {Error: true, StablePass: true}})
	if got != (Tally{Cases: 5, Worse: 1, Better: 1, Errors: 1}) {
		t.Fatalf("%+v", got)
	}
}

func TestGolden(t *testing.T) {
	cases := []struct {
		tally Tally
		want  Decision
	}{
		{Tally{Cases: 40}, Advance},
		{Tally{Cases: 40, Worse: 1}, Advance},
		{Tally{Cases: 40, Worse: 2}, Block},
		{Tally{Cases: 40, Worse: 30, Errors: 3}, GiveUp},
		{Tally{}, GiveUp},
	}
	for _, c := range cases {
		if got, _ := EvaluateGolden(c.tally); got != c.want {
			t.Fatalf("%+v: got %d", c.tally, got)
		}
	}
}

func TestShadowLooksOnlyAtDiscordantPairs(t *testing.T) {
	// 7 worse and none better is significant however large the corpus is.
	if got, _ := EvaluateShadow(Tally{Cases: 1000, Worse: 7}); got != Block {
		t.Fatalf("got %d", got)
	}
	// As many better as worse is no evidence of harm.
	if got, _ := EvaluateShadow(Tally{Cases: 1000, Worse: 20, Better: 20}); got != Advance {
		t.Fatalf("got %d", got)
	}
	if got, _ := EvaluateShadow(Tally{Cases: 100, Worse: 50, Errors: 6}); got != GiveUp {
		t.Fatalf("got %d", got)
	}
}

func canaryStats(requests, failures, source, output, infra int64) Stats {
	return Stats{Requests: requests, GuardFailures: failures, SourceChars: source, OutputChars: output, InfraErrors: infra}
}

func TestCanary(t *testing.T) {
	const timeout = time.Hour
	healthy := canaryStats(1000, 10, 10000, 12000, 0)
	cases := []struct {
		name      string
		candidate Stats
		elapsed   time.Duration
		want      Decision
	}{
		{"healthy", canaryStats(200, 2, 2000, 2400, 0), 0, Advance},
		{"too few samples", canaryStats(50, 0, 500, 600, 0), 0, Wait},
		{"too few samples, out of time", canaryStats(50, 0, 500, 600, 0), 2 * timeout, GiveUp},
		{"guard failures", canaryStats(200, 30, 2000, 2400, 0), 0, Rollback},
		{"output bloat", canaryStats(200, 2, 2000, 6000, 0), 0, Rollback},
		{"outage wins over bad numbers", canaryStats(200, 30, 2000, 6000, 100), 0, GiveUp},
		{"a few errors are tolerated", canaryStats(200, 2, 2000, 2400, 5), 0, Advance},
		{"too many errors to promote", canaryStats(200, 2, 2000, 2400, 30), 0, GiveUp},
	}
	for _, c := range cases {
		if got, reason := EvaluateCanary(healthy, c.candidate, 100, c.elapsed, timeout); got != c.want {
			t.Fatalf("%s: got %d (%s)", c.name, got, reason)
		}
	}
}
