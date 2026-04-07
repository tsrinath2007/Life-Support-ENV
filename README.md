---
title: DepUpgradeEnv
emoji: 📦
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - dependency-management
  - security
  - devops
license: mit
---

# DepUpgradeEnv

An **OpenEnv-compliant** RL environment where an AI agent upgrades outdated,
vulnerable, and conflicting Python package dependencies — a task every software
engineering team deals with continuously.

## Why This Matters

Dependency management failures are responsible for some of the most damaging
security breaches in software history (Log4Shell, OpenSSL Heartbleed, requests
CVEs). The average production codebase has 6+ outdated dependencies with known
CVEs at any given time. An agent that learns to safely upgrade dependencies —
respecting version constraints, resolving conflicts, and validating with tests —
has immediate real-world value.

## Environment Description

The agent receives a `requirements.txt` with issues — CVEs, outdated packages,
version conflicts — and must apply upgrade actions to resolve them without
breaking the test suite. Rewards are **dense**: every resolved issue earns
partial credit immediately, not just at episode end.

## Action Space

| `action_type` | Parameters | Description |
|---|---|---|
| `upgrade` | `package`, `version` | Upgrade a package to target version |
| `pin` | `package`, `version` | Pin a package at exact version |
| `remove` | `package` | Remove a package entirely |
| `run_tests` | — | Run the test suite |
| `validate` | — | Check remaining issues (no-op grader trigger) |
| `skip` | — | Do nothing this step |

## Observation Space

```json
{
  "task_id": "easy",
  "step": 3,
  "packages": [
    {
      "name": "requests",
      "current_version": "2.20.0",
      "latest_version": "2.31.0",
      "has_cve": true,
      "cve_severity": "critical",
      "is_outdated": true,
      "is_conflicting": false,
      "conflict_reason": null,
      "locked": false
    }
  ],
  "test_results": {"test_http_requests": false, "test_web_routes": false},
  "issues_remaining": ["requests: CVE (critical)", "flask: CVE (high)"],
  "score_so_far": 0.30,
  "message": "Upgraded requests to 2.31.0"
}
```

## Tasks

### Easy — 2 CVEs + 1 outdated package
No conflicts. Every upgrade is safe. Agent just needs to identify and fix CVEs
then update the outdated package.
**Grader:** cves_resolved (40%) + packages_updated (30%) + tests_passing (30%)
**Expected score:** 0.85 – 1.00

### Medium — 2 CVEs + version conflicts
Upgrading pandas without first upgrading scipy breaks everything. Agent must
learn the correct dependency order.
**Grader:** cves_resolved (35%) + conflicts_resolved (30%) + tests_passing (35%)
**Expected score:** 0.55 – 0.75

### Hard — 2 CVEs + diamond conflict + locked package + breaking API change
A diamond dependency conflict between transformers and datasets through
tokenizers. A locked torch package that must not be touched. A pillow upgrade
that introduces a breaking API change requiring `run_tests` to reconcile.
**Grader:** cves_resolved (30%) + diamond_resolved (30%) + locked_respected (10%) + tests_passing (30%)
**Expected score:** 0.35 – 0.55

## Setup

```bash
# Local
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860

# Docker
docker build -t dep-upgrade-env .
docker run -p 7860:7860 dep-upgrade-env

# Inference
export API_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="your_token"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
python inference.py

# Validate before submitting
python validate.py
python validate.py --url http://localhost:7860
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Liveness check |
| GET | /tasks | List all tasks |
| POST | /reset | Start new episode |
| POST | /step | Take an action |
| GET | /state | Full env state |

## Baseline Scores

| Task | Score |
|---|---|
| easy | ~0.90 |
| medium | ~0.65 |
| hard | ~0.45 |
