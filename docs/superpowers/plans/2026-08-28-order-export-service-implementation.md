# Order Export Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-free, multi-module order-export HTTP service whose initial integration tests deterministically expose realistic idempotency, worker-claim, retry, cache, and cleanup defects for mca to repair.

**Architecture:** The fixture runs independently under `/private/tmp/mca-order-export-service`. `repository.py` is the only SQLite owner, `worker.py` owns asynchronous execution, `service.py` maps HTTP use cases to persistence/cache, and `cleanup.py` owns terminal-record retention. Initial intentional defects live at those boundaries, while tests use barriers and controlled time so failures do not rely on scheduling luck.

**Tech Stack:** Python 3 standard library, sqlite3, ThreadingHTTPServer, unittest, curl.

---

### Task 1: Build the runnable service skeleton and happy path

**Files:**
- Create: `/private/tmp/mca-order-export-service/app/models.py`
- Create: `/private/tmp/mca-order-export-service/app/repository.py`
- Create: `/private/tmp/mca-order-export-service/app/service.py`
- Create: `/private/tmp/mca-order-export-service/app/worker.py`
- Create: `/private/tmp/mca-order-export-service/app/cache.py`
- Create: `/private/tmp/mca-order-export-service/app/cleanup.py`
- Create: `/private/tmp/mca-order-export-service/app/server.py`
- Create: `/private/tmp/mca-order-export-service/app/__init__.py`
- Create: `/private/tmp/mca-order-export-service/scripts/seed_orders.py`
- Create: `/private/tmp/mca-order-export-service/README.md`

- [x] Define task states, SQLite schema, HTTP endpoints, CSV rendering, and `/healthz`.
- [x] Seed deterministic order data and prove `POST /exports → worker run-once → GET/download` works.

### Task 2: Create deterministic failing maintenance tests

**Files:**
- Create: `/private/tmp/mca-order-export-service/tests/test_api.py`
- Create: `/private/tmp/mca-order-export-service/tests/test_worker.py`
- Create: `/private/tmp/mca-order-export-service/tests/test_concurrency.py`
- Create: `/private/tmp/mca-order-export-service/tests/test_cleanup.py`

- [x] Cover happy-path contract plus deterministic failures: concurrent idempotency, double worker claim, retry status, cache invalidation, and running-job cleanup; the expired completed-job cleanup control test already passes.
- [x] Run `python3 -m unittest discover -s tests -v`; initial result is 3 pass / 5 deterministic failures.

### Task 3: Prepare user experience and service process

**Files:**
- Create: `/private/tmp/mca-order-export-service/PROMPT.md`
- Create: `/private/tmp/mca-order-export-service/scripts/reset_runtime.py`

- [x] Document start/reset/curl commands and a production-issue-style mca prompt without implementation answers.
- [x] Start the server at `127.0.0.1:8765`, verify health + one HTTP happy path, and save initial failure evidence in `runtime/initial-tests.log`.
