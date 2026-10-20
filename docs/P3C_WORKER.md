# P3-C Worker

The Worker is a single-process, serial task consumer. It uses the P3-B
`TaskRepository` for short business transactions and the LangGraph PostgreSQL
checkpointer as the sole source of Research State, Research Memory, Trajectory,
Budget, Replay, and pending-node state.

## Launch contract

Install the `agent`, `postgres`, and `service` extras, then run
`research-agent-worker`. Set `RISE_WORKER_FACTORY` to an importable
`module:callable`. The callable constructs and returns one `TaskWorker`,
including the database URL, repository/session factory, long-lived
`PostgresSaver`, paper store, and a per-task runtime factory. This keeps model
and retrieval credentials in deployment configuration and out of Graph State.
The runtime factory returns `WorkerRuntime(runner, seed_context, markdown)`;
seed context and Markdown are required only when no checkpoint exists. They may
be zero-argument callables so recovery does not repeat pre-Graph Seed Context
Extraction or reopen the submitted input unnecessarily.

The worker factory must keep the saver and database engine open for the
lifetime of `run_forever()`. `SIGINT` and `SIGTERM` stop new claims and allow an
in-flight graph invocation to return at its next durable node boundary.

## Ownership and recovery

A queued task is atomically claimed through `FOR UPDATE SKIP LOCKED`. The
Worker starts a separate heartbeat thread and holds a dedicated PostgreSQL
session-level advisory lock for the task's stable Graph Thread for the entire
execution. An expired lease is only reclaimed after the advisory lock is
acquired; the repository then rechecks expiry under row lock and rotates the
lease token. Business writes require the current token. Heartbeat or database ownership loss makes the dedicated Worker process
fail-stop (exit code 75). This matters when the main thread is blocked inside
an LLM or retrieval call: the process cannot keep running that node after its
PostgreSQL lock session has disappeared. A supervisor can restart the Worker;
recovery waits for the task lease to expire and then takes the advisory lock.

The graph is interrupted before each Bootstrap, Controller, Scientific
Action, and Finalizer node. After each returned checkpoint, the Worker projects
events idempotently from the saved Trajectory and Replay records. On the next
invoke it checks cancellation at the node boundary. A running scientific call
is not forcefully interrupted. A cancellation before CommitSnapshot becomes a
business `cancelled` terminal; after CommitSnapshot, Finalizer is allowed to
finish. Existing RISE terminal outcomes map to business `completed` and retain
`rise_outcome`.

Only infrastructure failures are retried. A lost process is reclaimed after
its lease expires and the advisory lock becomes available. Recovery is bounded
by `max_recoveries` (default 3). A saved scientific `failed` or
`finalization_failed` state is reconciled as a RISE terminal and is not retried.
The external LLM or retrieval call in a node that did not commit a checkpoint
may run again after recovery; external exactly-once execution is not claimed.

## Event projection

Projection keys are derived from task UUID and Trajectory index or Replay
sequence. Repository writes are idempotent; if a checkpoint commits but an
event insert does not, the next Worker pass projects the same saved snapshot
again. `rise_action_confirmed` denotes an action represented by a persisted
Replay record. Task-attempt lifecycle events remain distinct from confirmed
scientific actions.
