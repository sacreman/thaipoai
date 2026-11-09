// Command reaper deletes training resources that have outlived their deadline.
//
// Every per-run GKE cluster, VM and disk carries a run ID label and a deadline label in Unix
// seconds. Cloud Scheduler runs this as a one-shot job outside the training workflow, so a
// workflow that dies still cannot leave a GPU running. Anything without both labels is left alone.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const (
	runLabel      = "thaipoai-run-id"
	deadlineLabel = "thaipoai-deadline"
	passTimeout   = 4 * time.Minute
)

type reaper struct {
	metadata, container, compute string
	http                         *http.Client
	now                          func() time.Time
	token, project               string
}

func newReaper(metadata, container, compute string) *reaper {
	return &reaper{
		metadata: strings.TrimRight(metadata, "/"), container: strings.TrimRight(container, "/"), compute: strings.TrimRight(compute, "/"),
		http: &http.Client{Timeout: 30 * time.Second}, now: time.Now,
	}
}

// expiry reads our labels. ours is false for anything without the run label. A resource with
// the run label and no readable deadline is an error and is never deleted, because a deadline
// we cannot read is not a deadline that has passed.
func expiry(labels map[string]string) (run string, deadline time.Time, ours bool, err error) {
	run = labels[runLabel]
	if run == "" {
		return "", time.Time{}, false, nil
	}
	seconds, parseErr := strconv.ParseUint(labels[deadlineLabel], 10, 63)
	if parseErr != nil || seconds == 0 {
		return run, time.Time{}, true, fmt.Errorf("run %s has no readable deadline", run)
	}
	return run, time.Unix(int64(seconds), 0), true, nil
}

func (r *reaper) do(ctx context.Context, method, target string, headers map[string]string) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, method, target, nil)
	if err != nil {
		return nil, err
	}
	for key, value := range headers {
		request.Header.Set(key, value)
	}
	return r.http.Do(request)
}

func (r *reaper) metadataGet(ctx context.Context, path string) ([]byte, error) {
	response, err := r.do(ctx, http.MethodGet, r.metadata+path, map[string]string{"Metadata-Flavor": "Google"})
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<16))
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: status %d", path, response.StatusCode)
	}
	return body, nil
}

func (r *reaper) authenticate(ctx context.Context) error {
	project, err := r.metadataGet(ctx, "/computeMetadata/v1/project/project-id")
	if err != nil {
		return err
	}
	raw, err := r.metadataGet(ctx, "/computeMetadata/v1/instance/service-accounts/default/token")
	if err != nil {
		return err
	}
	var token struct {
		AccessToken string `json:"access_token"`
	}
	if err := json.Unmarshal(raw, &token); err != nil || token.AccessToken == "" {
		return errors.New("metadata server returned no token")
	}
	r.project, r.token = strings.TrimSpace(string(project)), token.AccessToken
	return nil
}

func (r *reaper) api(ctx context.Context, method, target string) (*http.Response, error) {
	return r.do(ctx, method, target, map[string]string{"Authorization": "Bearer " + r.token})
}

func (r *reaper) getJSON(ctx context.Context, target string, out any) error {
	response, err := r.api(ctx, http.MethodGet, target)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("list: status %d", response.StatusCode)
	}
	return json.NewDecoder(response.Body).Decode(out)
}

// remove asks for one deletion. A resource that is already gone is what we wanted.
func (r *reaper) remove(ctx context.Context, target string) error {
	response, err := r.api(ctx, http.MethodDelete, target)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(response.Body, 512))
	switch response.StatusCode {
	case http.StatusOK, http.StatusAccepted, http.StatusNoContent, http.StatusNotFound:
		return nil
	}
	return fmt.Errorf("delete: status %d: %s", response.StatusCode, body)
}

type cluster struct {
	Name           string            `json:"name"`
	Location       string            `json:"location"`
	Status         string            `json:"status"`
	ResourceLabels map[string]string `json:"resourceLabels"`
}

// item is a VM or a disk. Scope is the "zones/x" or "regions/y" the listing filed it under.
type item struct {
	Name   string            `json:"name"`
	Labels map[string]string `json:"labels"`
	Users  []string          `json:"users"`
	Scope  string            `json:"-"`
}

func (r *reaper) listClusters(ctx context.Context) ([]cluster, bool, error) {
	var page struct {
		Clusters     []cluster `json:"clusters"`
		MissingZones []string  `json:"missingZones"`
	}
	err := r.getJSON(ctx, r.container+"/v1/projects/"+url.PathEscape(r.project)+"/locations/-/clusters", &page)
	return page.Clusters, len(page.MissingZones) == 0, err
}

func (r *reaper) listItems(ctx context.Context, kind string) ([]item, error) {
	var all []item
	token := ""
	for {
		target := r.compute + "/compute/v1/projects/" + url.PathEscape(r.project) + "/aggregated/" + kind + "?maxResults=500"
		if token != "" {
			target += "&pageToken=" + url.QueryEscape(token)
		}
		var page struct {
			Items map[string]struct {
				Instances []item `json:"instances"`
				Disks     []item `json:"disks"`
			} `json:"items"`
			NextPageToken string `json:"nextPageToken"`
		}
		if err := r.getJSON(ctx, target, &page); err != nil {
			return nil, err
		}
		for scope, scoped := range page.Items {
			for _, found := range append(scoped.Instances, scoped.Disks...) {
				found.Scope = scope
				all = append(all, found)
			}
		}
		if token = page.NextPageToken; token == "" {
			return all, nil
		}
	}
}

// pass makes one sweep. Clusters go first, and a run's VMs and disks are left to their cluster
// while it exists: its nodes sit in managed groups that would just replace a deleted VM. Once
// the cluster is gone, a later pass clears anything it left behind.
func (r *reaper) pass(ctx context.Context) error {
	if err := r.authenticate(ctx); err != nil {
		return fmt.Errorf("metadata: %w", err)
	}
	clusters, complete, err := r.listClusters(ctx)
	if err != nil {
		return fmt.Errorf("clusters: %w", err)
	}
	now := r.now()
	var problems []error
	clusterRuns := map[string]bool{}
	for _, found := range clusters {
		run, deadline, ours, err := expiry(found.ResourceLabels)
		if err != nil {
			problems = append(problems, fmt.Errorf("cluster %s: %w", found.Name, err))
			continue
		}
		if !ours {
			continue
		}
		clusterRuns[run] = true
		if !now.After(deadline) || found.Status == "STOPPING" {
			continue
		}
		target := r.container + "/v1/projects/" + url.PathEscape(r.project) + "/locations/" + url.PathEscape(found.Location) + "/clusters/" + url.PathEscape(found.Name)
		if err := r.remove(ctx, target); err != nil {
			problems = append(problems, fmt.Errorf("cluster %s: %w", found.Name, err))
			continue
		}
		log.Printf("deleting cluster %s, run %s, deadline %s", found.Name, run, deadline.UTC().Format(time.RFC3339))
	}
	if !complete {
		// A cluster we could not see would look like it had gone, and its live nodes like orphans.
		return errors.Join(append(problems, errors.New("cluster list incomplete, leaving VMs and disks for the next pass"))...)
	}
	for _, kind := range []string{"instances", "disks"} {
		items, err := r.listItems(ctx, kind)
		if err != nil {
			problems = append(problems, fmt.Errorf("%s: %w", kind, err))
			continue
		}
		for _, found := range items {
			run, deadline, ours, err := expiry(found.Labels)
			if err != nil {
				problems = append(problems, fmt.Errorf("%s %s: %w", kind, found.Name, err))
				continue
			}
			if !ours || clusterRuns[run] || !now.After(deadline) || len(found.Users) > 0 {
				continue // an attached disk goes on a later pass, once its VM has
			}
			target := r.compute + "/compute/v1/projects/" + url.PathEscape(r.project) + "/" + found.Scope + "/" + kind + "/" + url.PathEscape(found.Name)
			if err := r.remove(ctx, target); err != nil {
				problems = append(problems, fmt.Errorf("%s %s: %w", kind, found.Name, err))
				continue
			}
			log.Printf("deleting %s %s, run %s, deadline %s", kind, found.Name, run, deadline.UTC().Format(time.RFC3339))
		}
	}
	return errors.Join(problems...)
}

func main() {
	metadata := flag.String("metadata", "http://metadata.google.internal", "metadata server")
	container := flag.String("container", "https://container.googleapis.com", "GKE API")
	compute := flag.String("compute", "https://compute.googleapis.com", "Compute Engine API")
	flag.Parse()
	ctx, cancel := context.WithTimeout(context.Background(), passTimeout)
	defer cancel()
	if err := newReaper(*metadata, *container, *compute).pass(ctx); err != nil {
		log.Fatalf("pass failed: %v", err) // a failed Cloud Run execution is what preflight looks for
	}
	log.Print("pass complete")
}
