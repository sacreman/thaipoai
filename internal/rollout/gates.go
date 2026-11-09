package rollout

import (
	"fmt"
	"math"
	"strings"
	"time"
)

const (
	// GoldenMaxRegressions is how many must-pass cases may newly fail.
	GoldenMaxRegressions = 1
	// ShadowAlpha is the one-sided McNemar level for the replay corpus.
	ShadowAlpha = 0.01
	// CanaryAlpha is the one-sided Fisher level for live guard failures.
	CanaryAlpha = 0.01
	// CanaryMaxLengthRatio is how much longer candidate output may run, relative to stable.
	CanaryMaxLengthRatio = 1.5
	// MaxErrorShare is the share of infrastructure errors that makes a gate meaningless.
	MaxErrorShare = 0.05
	// CanaryMaxInfraShare is higher than MaxErrorShare because a lost replica is expected to
	// cost some candidate requests while it recovers.
	CanaryMaxInfraShare = 0.2
	// canaryMinInfraSamples stops a handful of early errors from ending a rollout.
	canaryMinInfraSamples = 20
	checksumMismatch      = "checksum mismatch"
)

func logChoose(n, k int) float64 {
	a, _ := math.Lgamma(float64(n + 1))
	b, _ := math.Lgamma(float64(k + 1))
	c, _ := math.Lgamma(float64(n - k + 1))
	return a - b - c
}

// McNemarP is the exact one-sided p-value that the candidate is worse. worse counts cases only
// the candidate failed, better counts cases only stable failed. Cases both passed or both failed
// say nothing about the difference, which is why an overall failure rate is not used.
func McNemarP(worse, better int) float64 {
	n := worse + better
	if n == 0 {
		return 1
	}
	p := 0.0
	for k := worse; k <= n; k++ {
		p += math.Exp(logChoose(n, k) - float64(n)*math.Ln2)
	}
	return math.Min(p, 1)
}

// FisherP is the exact one-sided p-value that group 1 fails more often than group 2. Live
// traffic is not paired, so McNemar cannot be used on it.
func FisherP(failed1, total1, failed2, total2 int) float64 {
	failed := failed1 + failed2
	total := total1 + total2
	if total1 == 0 || total2 == 0 {
		return 1
	}
	p := 0.0
	for k := failed1; k <= min(failed, total1); k++ {
		if failed-k > total2 {
			continue
		}
		p += math.Exp(logChoose(total1, k) + logChoose(total2, failed-k) - logChoose(total, failed))
	}
	return math.Min(p, 1)
}

// Replica is what one adapter-sync reported about the candidate.
type Replica struct {
	Ready bool
	Error string
}

// EvaluateVerify decides whether every replica holds the candidate. A bad checksum on every
// replica means the artefact is not what it claims to be, so it blocks. On only some replicas it
// may be a damaged download, so like any other local failure it waits and then gives up without
// blaming the candidate.
func EvaluateVerify(replicas []Replica, elapsed, timeout time.Duration) (Decision, string) {
	ready, mismatched := 0, 0
	for _, replica := range replicas {
		if strings.Contains(replica.Error, checksumMismatch) {
			mismatched++
		}
		if replica.Ready {
			ready++
		}
	}
	if len(replicas) > 0 && mismatched == len(replicas) {
		return Block, checksumMismatch
	}
	if len(replicas) > 0 && ready == len(replicas) {
		return Advance, fmt.Sprintf("%d of %d replicas verified", ready, len(replicas))
	}
	if elapsed > timeout {
		return GiveUp, fmt.Sprintf("only %d of %d replicas verified in time", ready, len(replicas))
	}
	return Wait, ""
}

// Pair is one test case translated by both adapters.
type Pair struct {
	StablePass, CandidatePass, Error bool
}

// Tally counts the pairs that matter.
type Tally struct {
	Cases  int `json:"cases"`
	Worse  int `json:"worse"`
	Better int `json:"better"`
	Errors int `json:"errors"`
}

// Count reduces pairs to a tally.
func Count(pairs []Pair) Tally {
	tally := Tally{Cases: len(pairs)}
	for _, pair := range pairs {
		switch {
		case pair.Error:
			tally.Errors++
		case pair.StablePass && !pair.CandidatePass:
			tally.Worse++
		case !pair.StablePass && pair.CandidatePass:
			tally.Better++
		}
	}
	return tally
}

func tooManyErrors(t Tally) bool {
	return t.Cases == 0 || float64(t.Errors)/float64(t.Cases) > MaxErrorShare
}

// EvaluateGolden blocks a candidate that newly fails must-pass cases.
func EvaluateGolden(t Tally) (Decision, string) {
	if tooManyErrors(t) {
		return GiveUp, fmt.Sprintf("%d of %d golden cases hit errors", t.Errors, t.Cases)
	}
	reason := fmt.Sprintf("%d golden regressions, %d allowed", t.Worse, GoldenMaxRegressions)
	if t.Worse > GoldenMaxRegressions {
		return Block, reason
	}
	return Advance, reason
}

// EvaluateShadow blocks a candidate that is worse on the replay corpus, tested on discordant
// pairs only.
func EvaluateShadow(t Tally) (Decision, string) {
	if tooManyErrors(t) {
		return GiveUp, fmt.Sprintf("%d of %d replay cases hit errors", t.Errors, t.Cases)
	}
	p := McNemarP(t.Worse, t.Better)
	reason := fmt.Sprintf("worse %d, better %d, p=%.4f", t.Worse, t.Better, p)
	if p < ShadowAlpha {
		return Block, reason
	}
	return Advance, reason
}

// Stats are the per-adapter counts from the gateway. They are counts only, so no live message
// is ever read or sent to a judge.
type Stats struct {
	Requests      int64 `json:"requests"`
	GuardFailures int64 `json:"guard_failures"`
	InfraErrors   int64 `json:"infra_errors"`
	SourceChars   int64 `json:"source_chars"`
	OutputChars   int64 `json:"output_chars"`
	Millis        int64 `json:"millis"`
}

// Add sums the counts from several gateways.
func (s *Stats) Add(other Stats) {
	s.Requests += other.Requests
	s.GuardFailures += other.GuardFailures
	s.InfraErrors += other.InfraErrors
	s.SourceChars += other.SourceChars
	s.OutputChars += other.OutputChars
	s.Millis += other.Millis
}

func lengthRatio(s Stats) float64 {
	if s.SourceChars == 0 {
		return 0
	}
	return float64(s.OutputChars) / float64(s.SourceChars)
}

// EvaluateCanary judges live traffic. Latency is reported but not gated, because both adapters
// share one queue and a slow candidate slows stable with it. Output length does not have that
// problem. Infrastructure is checked first so an outage can never look like a bad adapter.
func EvaluateCanary(stable, candidate Stats, minSamples int64, elapsed, timeout time.Duration) (Decision, string) {
	attempts := candidate.Requests + candidate.InfraErrors
	if attempts >= canaryMinInfraSamples && float64(candidate.InfraErrors)/float64(attempts) > CanaryMaxInfraShare {
		return GiveUp, fmt.Sprintf("%d of %d candidate requests hit errors", candidate.InfraErrors, attempts)
	}
	if candidate.Requests < minSamples || stable.Requests < minSamples {
		if elapsed > timeout {
			return GiveUp, fmt.Sprintf("not enough traffic: %d candidate, %d stable, %d needed", candidate.Requests, stable.Requests, minSamples)
		}
		return Wait, ""
	}
	p := FisherP(int(candidate.GuardFailures), int(candidate.Requests), int(stable.GuardFailures), int(stable.Requests))
	if p < CanaryAlpha {
		return Rollback, fmt.Sprintf("guard failures: candidate %d of %d, stable %d of %d, p=%.4f", candidate.GuardFailures, candidate.Requests, stable.GuardFailures, stable.Requests, p)
	}
	if ratio := lengthRatio(candidate) / math.Max(lengthRatio(stable), 1e-9); ratio > CanaryMaxLengthRatio {
		return Rollback, fmt.Sprintf("output is %.2f times stable's length", ratio)
	}
	if float64(candidate.InfraErrors)/float64(attempts) > MaxErrorShare {
		return GiveUp, fmt.Sprintf("%d of %d candidate requests hit errors, too many to promote", candidate.InfraErrors, attempts)
	}
	return Advance, fmt.Sprintf("%d candidate requests passed, p=%.4f", candidate.Requests, p)
}
