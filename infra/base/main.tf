# The long-lived half: reaper, identity, network, bucket. Applied by hand, once, by the owner.
# Runs get no rights to change anything here, so a dying run cannot remove its own reaper.

terraform {
  required_version = ">= 1.9"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.46"
    }
  }
}

variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "bucket_name" {
  description = "Globally unique. Holds run state, the ledger, checkpoints and adapters."
  type        = string
}

variable "github_repository" {
  description = "owner/name of the only repository allowed to start a run."
  type        = string
  validation {
    condition     = can(regex("^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", var.github_repository))
    error_message = "Use owner/name."
  }
}

variable "reaper_image" {
  description = "Full reaper image reference. Build with deploy/Dockerfile and CMD=reaper."
  type        = string
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "apis" {
  for_each = toset([
    "artifactregistry.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "compute.googleapis.com",
    "container.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "run.googleapis.com",
    "sts.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# Networks cannot carry labels, so the network lives here and only labelled things are per-run.
resource "google_compute_network" "main" {
  name                    = "thaipoai"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "main" {
  name                     = "thaipoai"
  network                  = google_compute_network.main.id
  ip_cidr_range            = "10.10.0.0/20"
  private_ip_google_access = true
}

resource "google_storage_bucket" "runs" {
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
}

resource "google_artifact_registry_repository" "images" {
  repository_id = "thaipoai"
  location      = var.region
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# Service accounts.

resource "google_service_account" "deployer" {
  account_id = "thaipoai-deployer"
  depends_on = [google_project_service.apis]
}

resource "google_service_account" "reaper" {
  account_id = "thaipoai-reaper"
  depends_on = [google_project_service.apis]
}

resource "google_service_account" "gpu_node" {
  account_id = "thaipoai-gpu-node"
  depends_on = [google_project_service.apis]
}

resource "google_service_account" "system_node" {
  account_id = "thaipoai-system-node"
  depends_on = [google_project_service.apis]
}

locals {
  node_roles = ["roles/artifactregistry.reader", "roles/logging.logWriter", "roles/monitoring.metricWriter"]

  # Only viewer rights on the reaper. Enough for preflight to see it, not enough to harm it.
  deployer_roles = [
    "roles/artifactregistry.reader",
    "roles/cloudscheduler.viewer",
    "roles/compute.instanceAdmin.v1",
    "roles/compute.viewer",
    "roles/container.admin",
    "roles/run.viewer",
  ]
  reaper_roles = ["roles/compute.instanceAdmin.v1", "roles/container.clusterAdmin"]

  bucket_members = {
    deployer = google_service_account.deployer.member
    gpu_node = google_service_account.gpu_node.member
  }
  bucket_roles = ["roles/storage.legacyBucketReader", "roles/storage.objectAdmin"]

  node_accounts = {
    gpu_node    = google_service_account.gpu_node.name
    system_node = google_service_account.system_node.name
  }

  run_condition = join(" && ", [
    "assertion.repository == \"${var.github_repository}\"",
    "assertion.ref == \"refs/heads/main\"",
    "assertion.workflow_ref == \"${var.github_repository}/.github/workflows/train.yml@refs/heads/main\"",
    # gpu is the reviewed start. gpu-run carries the later jobs, which must not wait for a reviewer.
    "(assertion.environment == \"gpu\" || assertion.environment == \"gpu-run\")",
  ])
}

resource "google_project_iam_member" "gpu_node" {
  for_each = toset(local.node_roles)
  project  = var.project_id
  role     = each.value
  member   = google_service_account.gpu_node.member
}

resource "google_project_iam_member" "system_node" {
  for_each = toset(local.node_roles)
  project  = var.project_id
  role     = each.value
  member   = google_service_account.system_node.member
}

resource "google_project_iam_member" "deployer" {
  for_each = toset(local.deployer_roles)
  project  = var.project_id
  role     = each.value
  member   = google_service_account.deployer.member
}

resource "google_project_iam_member" "reaper" {
  for_each = toset(local.reaper_roles)
  project  = var.project_id
  role     = each.value
  member   = google_service_account.reaper.member
}

resource "google_storage_bucket_iam_member" "runs" {
  for_each = {
    for pair in setproduct(keys(local.bucket_members), local.bucket_roles) : "${pair[0]} ${pair[1]}" => pair
  }
  bucket = google_storage_bucket.runs.name
  role   = each.value[1]
  member = local.bucket_members[each.value[0]]
}

# Creating or deleting a node needs the right to act as its service account.
resource "google_service_account_iam_member" "deployer_acts_as_node" {
  for_each           = local.node_accounts
  service_account_id = each.value
  role               = "roles/iam.serviceAccountUser"
  member             = google_service_account.deployer.member
}

resource "google_service_account_iam_member" "reaper_acts_as_node" {
  for_each           = local.node_accounts
  service_account_id = each.value
  role               = "roles/iam.serviceAccountUser"
  member             = google_service_account.reaper.member
}

# GitHub OIDC. No long-lived keys anywhere.

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "thaipoai-github"
  depends_on                = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  attribute_condition                = local.run_condition
  attribute_mapping = {
    "google.subject"        = "assertion.sub"
    "attribute.repository"  = "assertion.repository"
    "attribute.ref"         = "assertion.ref"
    "attribute.environment" = "assertion.environment"
  }
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "github_is_deployer" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${var.github_repository}"
}

# The reaper: a one-shot Cloud Run job, started by Cloud Scheduler.

resource "google_cloud_run_v2_job" "reaper" {
  name                = "thaipoai-reaper"
  location            = var.region
  deletion_protection = true
  depends_on          = [google_project_service.apis]

  template {
    template {
      service_account = google_service_account.reaper.email
      timeout         = "300s"
      max_retries     = 0
      containers {
        image = var.reaper_image
      }
    }
  }
}

resource "google_cloud_run_v2_job_iam_member" "scheduler_starts_reaper" {
  name     = google_cloud_run_v2_job.reaper.name
  location = var.region
  role     = "roles/run.invoker"
  member   = google_service_account.reaper.member
}

resource "google_cloud_scheduler_job" "reaper" {
  name             = "thaipoai-reaper"
  region           = var.region
  schedule         = "*/15 * * * *"
  time_zone        = "Etc/UTC"
  attempt_deadline = "320s"
  depends_on       = [google_project_service.apis]

  retry_config {
    retry_count = 1
  }

  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.reaper.name}:run"
    oauth_token {
      service_account_email = google_service_account.reaper.email
    }
  }
}

output "workload_identity_provider" {
  value = google_iam_workload_identity_pool_provider.github.name
}

output "deployer_service_account" {
  value = google_service_account.deployer.email
}
