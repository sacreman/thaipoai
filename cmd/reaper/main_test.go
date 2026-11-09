package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

var (
	testNow = time.Unix(1_800_000_000, 0)
	past    = testNow.Add(-time.Hour)
	future  = testNow.Add(time.Hour)
)

func labels(run string, deadline time.Time) map[string]string {
	return map[string]string{runLabel: run, deadlineLabel: strconv.FormatInt(deadline.Unix(), 10)}
}

// fakeGCP plays the metadata server, the GKE API and the Compute API. A deleted resource
// leaves the next listing, as it does in real life, once the delete is answered.
type fakeGCP struct {
	mu           sync.Mutex
	t            *testing.T
	clusters     []cluster
	instances    []item
	disks        []item
	deletes      []string
	failDelete   map[string]int // path to status, for the first attempt on that path
	gone         map[string]bool
	failList     map[string]bool
	missingZones []string
	pages        int
}

func (f *fakeGCP) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if strings.HasPrefix(r.URL.Path, "/computeMetadata/") {
		if r.Header.Get("Metadata-Flavor") != "Google" {
			f.t.Errorf("metadata request without the flavor header")
			w.WriteHeader(http.StatusForbidden)
			return
		}
		if strings.HasSuffix(r.URL.Path, "/project-id") {
			fmt.Fprint(w, "test-project\n")
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"access_token": "secret-token", "expires_in": 3599})
		return
	}
	if r.Header.Get("Authorization") != "Bearer secret-token" {
		f.t.Errorf("%s %s without the metadata token", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusUnauthorized)
		return
	}
	if r.Method == http.MethodDelete {
		f.deletes = append(f.deletes, r.URL.Path)
		if status, ok := f.failDelete[r.URL.Path]; ok {
			delete(f.failDelete, r.URL.Path)
			w.WriteHeader(status)
			return
		}
		if f.gone[r.URL.Path] {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		f.remove(r.URL.Path)
		json.NewEncoder(w).Encode(map[string]string{"name": "operation-1"})
		return
	}
	switch {
	case f.failList[r.URL.Path]:
		w.WriteHeader(http.StatusInternalServerError)
	case r.URL.Path == "/v1/projects/test-project/locations/-/clusters":
		json.NewEncoder(w).Encode(map[string]any{"clusters": f.clusters, "missingZones": f.missingZones})
	case r.URL.Path == "/compute/v1/projects/test-project/aggregated/instances":
		f.serveAggregated(w, r, "instances", f.instances)
	case r.URL.Path == "/compute/v1/projects/test-project/aggregated/disks":
		f.serveAggregated(w, r, "disks", f.disks)
	default:
		f.t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusNotFound)
	}
}

func (f *fakeGCP) remove(path string) {
	name := path[strings.LastIndex(path, "/")+1:]
	f.clusters = slices.DeleteFunc(f.clusters, func(c cluster) bool { return c.Name == name })
	drop := func(i item) bool { return i.Name == name }
	f.instances, f.disks = slices.DeleteFunc(f.instances, drop), slices.DeleteFunc(f.disks, drop)
}

// serveAggregated answers one resource per page, so any test proves paging is followed.
func (f *fakeGCP) serveAggregated(w http.ResponseWriter, r *http.Request, kind string, items []item) {
	f.pages++
	index, _ := strconv.Atoi(r.URL.Query().Get("pageToken"))
	response := map[string]any{"items": map[string]any{"zones/empty-zone": map[string]any{"warning": map[string]string{"code": "NO_RESULTS_ON_PAGE"}}}}
	if index < len(items) {
		listed := items[index]
		entry := map[string]any{"name": listed.Name, "labels": listed.Labels, "users": listed.Users}
		response["items"].(map[string]any)[listed.Scope] = map[string]any{kind: []any{entry}}
	}
	if index+1 < len(items) {
		response["nextPageToken"] = strconv.Itoa(index + 1)
	}
	json.NewEncoder(w).Encode(response)
}

func runPass(t *testing.T, f *fakeGCP) error {
	t.Helper()
	f.t = t
	server := httptest.NewServer(f)
	t.Cleanup(server.Close)
	r := newReaper(server.URL, server.URL, server.URL)
	r.now = func() time.Time { return testNow }
	return r.pass(context.Background())
}

func wantDeletes(t *testing.T, f *fakeGCP, want ...string) {
	t.Helper()
	got := slices.Clone(f.deletes)
	slices.Sort(got)
	slices.Sort(want)
	if !slices.Equal(got, want) {
		t.Fatalf("deleted %v, want %v", got, want)
	}
}

const (
	clusterPath  = "/v1/projects/test-project/locations/us-central1-a/clusters/"
	instancePath = "/compute/v1/projects/test-project/zones/us-central1-a/instances/"
	diskPath     = "/compute/v1/projects/test-project/zones/us-central1-a/disks/"
	zone         = "zones/us-central1-a"
)

func TestAClusterPastItsDeadlineIsDeleted(t *testing.T) {
	f := &fakeGCP{clusters: []cluster{{Name: "c1", Location: "us-central1-a", ResourceLabels: labels("r1", past)}}}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f, clusterPath+"c1")
}

func TestAClusterBeforeItsDeadlineIsKept(t *testing.T) {
	f := &fakeGCP{clusters: []cluster{{Name: "c1", Location: "us-central1-a", ResourceLabels: labels("r1", future)}}}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f)
}

func TestAResourceExactlyAtItsDeadlineIsKept(t *testing.T) {
	f := &fakeGCP{clusters: []cluster{{Name: "c1", Location: "us-central1-a", ResourceLabels: labels("r1", testNow)}}}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f)
}

func TestResourcesWithoutOurLabelAreNeverTouched(t *testing.T) {
	unlabelled := map[string]string{"team": "someone-else", deadlineLabel: "1"}
	f := &fakeGCP{
		clusters:  []cluster{{Name: "theirs", Location: "us-central1-a", ResourceLabels: unlabelled}, {Name: "bare", Location: "us-central1-a"}},
		instances: []item{{Name: "theirs-vm", Scope: zone, Labels: unlabelled}, {Name: "bare-vm", Scope: zone}},
		disks:     []item{{Name: "theirs-disk", Scope: zone, Labels: unlabelled}, {Name: "bare-disk", Scope: zone}},
	}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f)
}

func TestAnUnreadableDeadlineIsNeverDeletedAndFailsThePass(t *testing.T) {
	for _, deadline := range []string{"", "soon", "0", "-5"} {
		f := &fakeGCP{clusters: []cluster{{Name: "c1", Location: "us-central1-a", ResourceLabels: map[string]string{runLabel: "r1", deadlineLabel: deadline}}}}
		err := runPass(t, f)
		if err == nil || !strings.Contains(err.Error(), "no readable deadline") {
			t.Fatalf("deadline %q: got %v", deadline, err)
		}
		wantDeletes(t, f)
	}
}

func TestAResourceAlreadyGoneIsNotAFailure(t *testing.T) {
	f := &fakeGCP{
		clusters: []cluster{{Name: "c1", Location: "us-central1-a", ResourceLabels: labels("r1", past)}},
		gone:     map[string]bool{clusterPath + "c1": true},
	}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f, clusterPath+"c1")
}

func TestASecondPassDeletesNothingMore(t *testing.T) {
	f := &fakeGCP{
		instances: []item{{Name: "vm1", Scope: zone, Labels: labels("r1", past)}},
		disks:     []item{{Name: "d1", Scope: zone, Labels: labels("r1", past)}},
	}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f, instancePath+"vm1", diskPath+"d1")
}

func TestAClusterAlreadyStoppingIsNotAskedAgain(t *testing.T) {
	f := &fakeGCP{clusters: []cluster{{Name: "c1", Location: "us-central1-a", Status: "STOPPING", ResourceLabels: labels("r1", past)}}}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f)
}

func TestNodesAreLeftToTheirClusterWhileItExists(t *testing.T) {
	f := &fakeGCP{
		clusters:  []cluster{{Name: "c1", Location: "us-central1-a", ResourceLabels: labels("r1", past)}},
		instances: []item{{Name: "node-1", Scope: zone, Labels: labels("r1", past)}},
		disks:     []item{{Name: "pv-1", Scope: zone, Labels: labels("r1", past)}},
	}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f, clusterPath+"c1")
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f, clusterPath+"c1", instancePath+"node-1", diskPath+"pv-1")
}

func TestAnAttachedDiskWaitsForAPassAfterItsVMHasGone(t *testing.T) {
	f := &fakeGCP{disks: []item{{Name: "d1", Scope: zone, Labels: labels("r1", past), Users: []string{"vm1"}}}}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f)
}

func TestARegionalDiskIsDeletedAtItsRegionalPath(t *testing.T) {
	f := &fakeGCP{disks: []item{{Name: "d1", Scope: "regions/us-central1", Labels: labels("r1", past)}}}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f, "/compute/v1/projects/test-project/regions/us-central1/disks/d1")
}

func TestEveryPageOfAListingIsRead(t *testing.T) {
	f := &fakeGCP{instances: []item{
		{Name: "keep", Scope: zone, Labels: labels("r1", future)},
		{Name: "bare", Scope: zone},
		{Name: "vm3", Scope: "zones/us-central1-b", Labels: labels("r2", past)},
	}}
	if err := runPass(t, f); err != nil {
		t.Fatal(err)
	}
	wantDeletes(t, f, "/compute/v1/projects/test-project/zones/us-central1-b/instances/vm3")
	if f.pages != 4 { // three instance pages and one disk page
		t.Fatalf("read %d pages, want 4", f.pages)
	}
}

func TestOneFailedDeleteDoesNotStopTheRestAndFailsThePass(t *testing.T) {
	f := &fakeGCP{
		instances:  []item{{Name: "vm1", Scope: zone, Labels: labels("r1", past)}, {Name: "vm2", Scope: zone, Labels: labels("r2", past)}},
		failDelete: map[string]int{instancePath + "vm1": http.StatusForbidden},
	}
	err := runPass(t, f)
	if err == nil || !strings.Contains(err.Error(), "instances vm1") {
		t.Fatalf("got %v", err)
	}
	wantDeletes(t, f, instancePath+"vm1", instancePath+"vm2")
}

func TestAFailedClusterListDeletesNothing(t *testing.T) {
	f := &fakeGCP{
		instances: []item{{Name: "vm1", Scope: zone, Labels: labels("r1", past)}},
		failList:  map[string]bool{"/v1/projects/test-project/locations/-/clusters": true},
	}
	if err := runPass(t, f); err == nil {
		t.Fatal("want a failed pass")
	}
	wantDeletes(t, f)
}

func TestAnIncompleteClusterListDeletesWhatItSawButNoOrphans(t *testing.T) {
	f := &fakeGCP{
		clusters:     []cluster{{Name: "c1", Location: "us-central1-a", ResourceLabels: labels("r1", past)}},
		instances:    []item{{Name: "maybe-a-live-node", Scope: zone, Labels: labels("r2", past)}},
		missingZones: []string{"us-central1-b"},
	}
	if err := runPass(t, f); err == nil {
		t.Fatal("want a failed pass")
	}
	wantDeletes(t, f, clusterPath+"c1")
}

func TestAFailedInstanceListStillClearsDisksAndFailsThePass(t *testing.T) {
	f := &fakeGCP{
		disks:    []item{{Name: "d1", Scope: zone, Labels: labels("r1", past)}},
		failList: map[string]bool{"/compute/v1/projects/test-project/aggregated/instances": true},
	}
	if err := runPass(t, f); err == nil {
		t.Fatal("want a failed pass")
	}
	wantDeletes(t, f, diskPath+"d1")
}

func TestNoMetadataServerMeansNoRequestsToTheAPIs(t *testing.T) {
	r := newReaper("http://127.0.0.1:1", "http://127.0.0.1:1", "http://127.0.0.1:1")
	r.http.Timeout = time.Second
	if err := r.pass(context.Background()); err == nil || !strings.Contains(err.Error(), "metadata") {
		t.Fatalf("got %v", err)
	}
}
