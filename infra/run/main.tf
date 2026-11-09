# One run's GPU cluster. Everything here carries the run ID and deadline labels, because the
# reaper and teardown find resources by label and nothing else.

terraform {
  required_version = ">= 1.9"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.46"
    }
  }
  backend "gcs" {} # bucket and prefix come from terraform init, one state per run
}

variable "project_id" {
  type = string
}

variable "zone" {
  type    = string
  default = "us-central1-a"
}

variable "run_id" {
  type = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{0,30}$", var.run_id))
    error_message = "A run ID must be a valid label value and fit in a cluster name."
  }
}

variable "deadline_epoch" {
  description = "Unix seconds. The reaper deletes this run after it. The workflow sets it once."
  type        = string
  validation {
    condition     = can(regex("^[1-9][0-9]{9}$", var.deadline_epoch))
    error_message = "Use Unix seconds."
  }
}

variable "job_deadline_seconds" {
  type = number
}

variable "gpu" {
  type    = string
  default = "l4"
  validation {
    condition     = contains(["l4", "a100"], var.gpu)
    error_message = "Use l4 or a100."
  }
}

provider "google" {
  project = var.project_id
  zone    = var.zone
}

locals {
  labels = {
    thaipoai-run-id   = var.run_id
    thaipoai-deadline = var.deadline_epoch
  }
  gpu = {
    l4   = { machine_type = "g2-standard-8", accelerator = "nvidia-l4" }
    a100 = { machine_type = "a2-highgpu-1g", accelerator = "nvidia-tesla-a100" }
  }[var.gpu]
  scopes = ["https://www.googleapis.com/auth/cloud-platform"]
}

# ponytail: public control plane and public nodes, IAM-guarded, since hosted runners have no
# fixed IP. Add private nodes and a DNS endpoint if this ever outlives a single run.
resource "google_container_cluster" "run" {
  name                = "thaipoai-${var.run_id}"
  location            = var.zone
  deletion_protection = false
  network             = "thaipoai"
  subnetwork          = "thaipoai"
  resource_labels     = local.labels
  initial_node_count  = 1

  ip_allocation_policy {}

  release_channel {
    channel = "REGULAR"
  }

  # The default pool is the CPU system pool: Kueue and GKE's own pods live here.
  node_config {
    machine_type    = "e2-standard-4"
    disk_type       = "pd-balanced"
    disk_size_gb    = 50
    service_account = "thaipoai-system-node@${var.project_id}.iam.gserviceaccount.com"
    oauth_scopes    = local.scopes
    resource_labels = local.labels
  }
}

# ponytail: the GPU node's own service account reaches the bucket, with no Workload Identity,
# because only the training Job schedules here. Add it if other workloads ever share the pool.
resource "google_container_node_pool" "gpu" {
  name               = "gpu"
  cluster            = google_container_cluster.run.name
  location           = var.zone
  initial_node_count = 0

  autoscaling {
    min_node_count = 0
    max_node_count = 1
  }

  node_config {
    machine_type    = local.gpu.machine_type
    spot            = true
    disk_type       = "pd-balanced"
    disk_size_gb    = 200
    service_account = "thaipoai-gpu-node@${var.project_id}.iam.gserviceaccount.com"
    oauth_scopes    = local.scopes
    resource_labels = local.labels

    # Google's own cap per node, one hour past the Job deadline. The reaper is the real guarantee.
    max_run_duration = "${var.job_deadline_seconds + 3600}s"

    guest_accelerator {
      type  = local.gpu.accelerator
      count = 1
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }
  }
}
