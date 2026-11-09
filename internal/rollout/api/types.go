// Package api is the ModelRollout resource.
package api

import (
	"encoding/json"
	"strconv"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"github.com/sacreman/thaipoai/internal/rollout"
)

// GroupVersion identifies the resource.
var GroupVersion = schema.GroupVersion{Group: "thaipoai.dev", Version: "v1alpha1"}

// CanarySpec tunes the live phase. The CRD fills in the defaults.
type CanarySpec struct {
	Percent        int `json:"percent,omitempty"`
	MinSamples     int `json:"minSamples,omitempty"`
	TimeoutMinutes int `json:"timeoutMinutes,omitempty"`
}

// Spec is one rollout. The object's name is the target.
type Spec struct {
	Model   rollout.Model `json:"model"`
	Attempt int           `json:"attempt,omitempty"`
	Canary  CanarySpec    `json:"canary,omitempty"`
}

// ID names this attempt in the gateway's counts.
func (s Spec) ID() string { return s.Model.SHA256 + "/" + strconv.Itoa(s.Attempt) }

// CanaryResult is the last count read from the gateways.
type CanaryResult struct {
	Stable    rollout.Stats `json:"stable"`
	Candidate rollout.Stats `json:"candidate"`
}

// Outcome is one finished rollout.
type Outcome struct {
	Model  string        `json:"model"`
	Phase  rollout.Phase `json:"phase"`
	Reason string        `json:"reason,omitempty"`
	At     metav1.Time   `json:"at"`
}

// Status is written only by the controller, through the status subresource.
type Status struct {
	Observed        string         `json:"observed,omitempty"` // the model sha256 this status is about
	ObservedAttempt int            `json:"observedAttempt"`
	Phase           rollout.Phase  `json:"phase,omitempty"`
	Reason          string         `json:"reason,omitempty"`
	Stable          *rollout.Model `json:"stable,omitempty"`
	Previous        *rollout.Model `json:"previous,omitempty"`
	PhaseStartedAt  *metav1.Time   `json:"phaseStartedAt,omitempty"`
	Golden          *rollout.Tally `json:"golden,omitempty"`
	Shadow          *rollout.Tally `json:"shadow,omitempty"`
	Canary          *CanaryResult  `json:"canary,omitempty"`
	History         []Outcome      `json:"history,omitempty"`
}

// ModelRollout moves one target to a new model.
type ModelRollout struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              Spec   `json:"spec"`
	Status            Status `json:"status,omitempty"`
}

// ModelRolloutList is a list of rollouts.
type ModelRolloutList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ModelRollout `json:"items"`
}

// Known limit: deep copies go through JSON instead of generated code. One object, reconciled every few seconds.
func roundTrip[T any](in *T) *T {
	raw, err := json.Marshal(in)
	if err != nil {
		panic(err)
	}
	out := new(T)
	if err := json.Unmarshal(raw, out); err != nil {
		panic(err)
	}
	return out
}

// DeepCopyObject implements runtime.Object.
func (mr *ModelRollout) DeepCopyObject() runtime.Object { return roundTrip(mr) }

// DeepCopyObject implements runtime.Object.
func (list *ModelRolloutList) DeepCopyObject() runtime.Object { return roundTrip(list) }

// AddToScheme registers the resource.
func AddToScheme(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(GroupVersion, &ModelRollout{}, &ModelRolloutList{})
	metav1.AddToGroupVersion(scheme, GroupVersion)
	return nil
}
