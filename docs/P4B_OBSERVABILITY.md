# P4-B Structured Logging and Execution Metrics

P4-B adds a standard-library logging and telemetry layer. It does not add a
metrics database, alter `task_events`, or change Graph State, Checkpoint,
Trajectory, Replay, Budget, or task status behavior.

## Configuration

The API and Worker console entry points call `configure_logging()` once.
Configuration is read from:

- `RISE_LOG_LEVEL` (default `INFO`)
- `RISE_LOG_JSON` (default `true`; accepts true/false, 1/0, yes/no, on/off)

The formatter is installed on an existing root handler when possible, so
repeated initialization does not add duplicate handlers. Both JSON and plain
formats serialize only event names and allowlisted scalar fields; arbitrary log
messages and exception text are not emitted.

## Correlation and privacy

The allowlisted context includes `request_id`, `task_id`, `thread_id`,
`worker_id`, `attempt_count`, `recovery_count`, `graph_node`, and `action`.
Events also allow safe status/outcome/error categories, operation names,
durations, token counts, budget counts, and provider call counts. Unknown fields
are dropped. The output intentionally excludes paper text, prompts, model output,
retrieval queries, lease tokens, API keys, database URLs, and exception messages
or tracebacks.

`ContextVar` scopes are reset at request/task boundaries. API requests create a
request ID; task creation adds the returned task/thread IDs before logging
submission completion. The independent Worker creates its own scope from the
claimed database row, correlating across processes by the stable task/thread
IDs rather than trying to carry Python context across a process boundary.
LangGraph nodes add node/action context. The heartbeat thread receives an
explicit `copy_context()` snapshot and stops with the task attempt.

For SSE, `api.request.finished` means the streaming response was opened. The
separate `api.sse.stream_finished` event measures the full stream lifetime and
reports its emitted event count and completion/disconnect/error status.

## Measurement semantics

- API request and accepted task-submission durations use `time.monotonic_ns()`.
- Worker attempt duration covers lock acquisition through attempt cleanup;
  recovery and cancellation have separate lifecycle events.
- Each Graph Node and Graph invocation emits elapsed time and a snapshot summary
  of the existing outcome, budget, Trajectory, and Replay state.
- LLM counts represent calls through `complete_structured`. Token fields stay
  `null` when unavailable. Format retries are reported when response metadata
  exposes them; DeepSeek transport retries are reported when the concrete client
  exposes its attempt counter. Hidden provider retry details remain unknown.
- Retrieval counts represent provider method calls. HTTP request/retry counts
  and cache hits are included only when the provider exposes them; otherwise
  they remain `null`.
- Budget values are read from the returned Graph State; telemetry does not
  maintain a second counter.

The Worker wraps shared LLM and Retrieval dependencies after its configured
Runtime Factory returns, deduplicating wrappers by object identity. The wrapper
preserves signatures, return values, and exception propagation. A deployment
Runtime Factory that performs Seed Context Extraction before returning its
`WorkerRuntime` should wrap that extraction client with `ObservedLLMClient`
explicitly if those pre-Graph calls also need measurement. The Pilot's existing
recording client is not treated as production telemetry.

## Failure isolation

`Telemetry.record()` is best-effort and reports whether its sink accepted an
event; a sink failure does not interrupt task execution. The observed LLM and
Retrieval wrappers re-raise the original operation exception. Checkpoint,
Worker ownership, repository, and scientific failures continue through their
existing error handling; telemetry does not retry or translate them. Durable
`task_events` remain the source for user-visible progress, and are not duplicated
by these operational log events.
