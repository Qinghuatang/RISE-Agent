# P3-E local API and Worker integration

The P3-E integration tests start a real API process and one or more real Worker
processes against an isolated PostgreSQL schema. Their Worker factory uses the
repository's deterministic `FakeLLM` and `FakeRetrieval` fixtures; no model API
or retrieval service is contacted.

## Run the independent-process acceptance tests

Start a disposable PostgreSQL server or use a dedicated test database, then
run:

```bash
export RISE_TEST_POSTGRES_URL='postgresql://USER@127.0.0.1:5432/TEST_DATABASE'
PYTHONPATH=.:src:tests pytest -q tests/test_p3e_e2e.py
```

The suite creates a unique schema per test, starts the API using
`research_agent.service.api_cli`, and loads the test Worker through the same
`RISE_WORKER_FACTORY=module:callable` contract used by deployments. It drops
only its own schema after the child processes exit.

Run the full PostgreSQL-backed regression suite with:

```bash
export RISE_TEST_POSTGRES_URL='postgresql://USER@127.0.0.1:5432/TEST_DATABASE'
PYTHONPATH=.:src:tests pytest -q
```

## Manually start a deterministic local smoke run

Install the project extras and configure a dedicated database and shared paper
directory:

```bash
pip install -e '.[agent,postgres,service,api,test]'
export RISE_DATABASE_URL='postgresql://USER@127.0.0.1:5432/RISE_LOCAL'
export RISE_PAPER_STORE_ROOT="$PWD/var/papers"
export PYTHONPATH="$PWD/src:$PWD/tests"
export RISE_WORKER_FACTORY='p3e_worker_factory:build_worker'
```

Start `research-agent-api` in one terminal and `research-agent-worker` in a
second terminal. The API applies the business-table migration and initializes
the Graph checkpoint schema. Both processes must use the same database URL and
paper-store root. The Worker claims queued tasks serially and writes Graph
checkpoints; the API reads task events and final results from PostgreSQL.

Submit a test Markdown paper from another terminal:

```bash
curl -X POST http://127.0.0.1:8000/api/tasks \
  -H 'Content-Type: application/json' \
  -d '{"markdown":"# Generic supplied seed\n\n## ABSTRACT\nA supplied abstract.\n\n## 1 METHOD\nA method with interacting units. Unit A and Unit B.\n\n## 2 FINDINGS\nA supplied finding and a supplied limitation.\n\n## REFERENCES\nReference-only content.\n"}'
```

Use the returned `task_id` with `GET /api/tasks/{task_id}`,
`GET /api/tasks/{task_id}/events`, and `GET /api/tasks/{task_id}/result`.

`p3e_worker_factory` is a deterministic test fixture, not a production provider
configuration. A deployment must supply its own importable Worker factory that
constructs the intended model and retrieval clients and returns a configured
`TaskWorker`; provider credentials stay in process configuration and never
belong in Graph State or the task event payload.
