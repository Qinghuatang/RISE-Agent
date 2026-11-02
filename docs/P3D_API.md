# P3-D FastAPI service

The API accepts work into the P3-B business database and serves events/results
from existing persistence. It never invokes LangGraph or starts a Worker.
Run the Worker separately with the P3-C `research-agent-worker` entry point.

## Run locally

Install the existing Runtime and service dependencies plus the API extra:

```bash
pip install -e '.[agent,postgres,service,api]'
export RISE_DATABASE_URL='postgresql://...'
export RISE_PAPER_STORE_ROOT='/var/lib/rise/papers'
research-agent-api
```

The default bind address is `127.0.0.1`; set `RISE_API_HOST` only when a
deliberate deployment needs a different interface. The API and Worker must
share `RISE_DATABASE_URL` and `RISE_PAPER_STORE_ROOT`. The Markdown store is
content-addressed and stores read-only files; the PostgreSQL task row and
input-artifact metadata are committed separately, so an interrupted task
creation can leave an unreferenced content file for later cleanup.

Supported settings are `RISE_API_HOST`, `RISE_API_PORT`,
`RISE_MAX_MARKDOWN_BYTES` (default 10 MiB), `RISE_SSE_POLL_INTERVAL` (default
1 second), and `RISE_SSE_HEARTBEAT_INTERVAL` (default 15 seconds). The app
applies Alembic business-table migrations during startup and initializes the
official Graph checkpointer schema for read access. It does not run a model or
retrieval provider.

## HTTP contract

| Method and path | Behavior |
| --- | --- |
| `GET /health/live` | Process liveness; no database request. |
| `GET /health/ready` | Checks PostgreSQL business and checkpoint access. |
| `POST /api/tasks` | JSON `{ "markdown": "...", "execution_config": {} }`, or multipart `markdown` text / `file` `.md` plus optional JSON `execution_config`. Returns `202` and a task ID. |
| `GET /api/tasks?limit=50&cursor=...` | Cursor pagination ordered by creation time and UUID, newest first. |
| `GET /api/tasks/{id}` | Business status, RISE outcome, timestamps, attempts and latest persisted event summary. |
| `GET /api/tasks/{id}/result` | Final Proposal, CommitSnapshot and result counters from the Graph checkpoint. Returns `409` until the task reaches a business terminal state. |
| `POST /api/tasks/{id}/cancel` | Cancels queued work or records a safe-boundary cancellation request for the Worker. |
| `GET /api/tasks/{id}/events` | Replays persisted `task_events` in sequence and continues polling while nonterminal. |

An optional `Idempotency-Key` makes repeated identical submissions return the
same task; reuse for a different Markdown/configuration digest returns `409`.
Markdown must be non-empty UTF-8, uploaded files must end in `.md`, and the
configured byte limit is enforced. Business `task_status` remains independent
of `rise_outcome`: confirmed RISE outcomes use `task_status=completed`, while
service failures have `task_status=failed` and a null RISE outcome.

SSE uses numeric task-local event sequence IDs. Clients reconnect with
`Last-Event-ID: <seq>`; the server selects rows strictly greater than that
sequence. SSE payloads are an allowlisted progress projection, so persisted
error bodies and arbitrary fields are not exposed. Events emitted by the
Worker remain distinct from the Graph checkpoint; if event projection lags,
the Worker can fill it on its next pass.
