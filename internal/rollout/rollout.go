// Package rollout holds the decisions of a model rollout as pure functions: which phase comes
// next, what each phase should have deployed, and whether a gate has passed. Everything that
// touches the network lives in cmd/rollout-controller. See docs/rollout-states.md.
package rollout

// Phase is where a rollout has got to.
type Phase string

const (
	Verifying    Phase = "Verifying"
	Golden       Phase = "Golden"
	Shadow       Phase = "Shadow"
	Canary       Phase = "Canary"
	Promoted     Phase = "Promoted"
	Blocked      Phase = "Blocked"
	RolledBack   Phase = "RolledBack"
	Inconclusive Phase = "Inconclusive"
)

// Terminal reports whether a rollout has finished.
func Terminal(p Phase) bool {
	return p == Promoted || p == Blocked || p == RolledBack || p == Inconclusive
}

// Decision is what a gate concluded.
type Decision int

const (
	Wait     Decision = iota
	Advance           // the gate passed
	Block             // the candidate is bad and never received traffic
	Rollback          // the candidate is bad and had traffic
	GiveUp            // the infrastructure failed, so nothing was learned about the candidate
)

type edge struct {
	from     Phase
	decision Decision
}

// transitions is the whole state machine. Anything not listed leaves the phase unchanged, which
// is how an infrastructure fault is kept from ever becoming a rollback.
var transitions = map[edge]Phase{
	{Verifying, Advance}: Golden,
	{Verifying, GiveUp}:  Inconclusive,
	{Golden, Advance}:    Shadow,
	{Golden, Block}:      Blocked,
	{Golden, GiveUp}:     Inconclusive,
	{Shadow, Advance}:    Canary,
	{Shadow, Block}:      Blocked,
	{Shadow, GiveUp}:     Inconclusive,
	{Canary, Advance}:    Promoted,
	{Canary, Rollback}:   RolledBack,
	{Canary, GiveUp}:     Inconclusive,
}

// Next returns the phase that follows a decision.
func Next(p Phase, d Decision) Phase {
	if next, ok := transitions[edge{p, d}]; ok {
		return next
	}
	return p
}

// Model identifies one exact set of weights.
type Model struct {
	Name   string `json:"name"`
	SHA256 string `json:"sha256"`
}

// Registered is the name the model server knows these weights by. The digest in it means a
// replica that serves this name holds these weights.
func (m Model) Registered() string { return m.Name + "-" + m.SHA256[:12] }

// Desired is one entry in the published list of what should be served.
type Desired struct {
	Model
	Required bool `json:"required"`
}

// Routing is what the gateway reads. The gateway keeps the last one it read, so traffic holds
// steady while the controller is down.
type Routing struct {
	ID               string `json:"id"`
	Stable           string `json:"stable"`
	Candidate        string `json:"candidate,omitempty"`
	CandidatePercent int    `json:"candidate_percent,omitempty"`
}

// PlanInput is the durable state a plan is derived from.
type PlanInput struct {
	Phase     Phase
	Stable    *Model
	Previous  *Model // the last stable, kept until the next rollout starts
	Candidate Model
	Percent   int
	Base      string
	ID        string
}

// Plan derives what should be deployed from the durable status alone. The controller reapplies
// it on every reconcile, so a crash between writing status and applying it cannot lose anything.
//
// The previous stable stays in the list after a promotion. Replicas and gateways pick up changes at
// different moments, and a gateway still sending traffic to it must not find it gone.
//
// Routing carries the rollout's ID only during Canary, so each canary counts from zero.
func Plan(in PlanInput) ([]Desired, Routing) {
	desired := []Desired{}
	routing := Routing{Stable: in.Base}
	if in.Stable != nil {
		desired = append(desired, Desired{Model: *in.Stable, Required: true})
		routing.Stable = in.Stable.Registered()
	}
	if in.Previous != nil {
		desired = append(desired, Desired{Model: *in.Previous})
	}
	if Terminal(in.Phase) {
		return desired, routing
	}
	desired = append(desired, Desired{Model: in.Candidate})
	if in.Phase == Canary {
		routing.ID, routing.Candidate, routing.CandidatePercent = in.ID, in.Candidate.Registered(), in.Percent
	}
	return desired, routing
}
