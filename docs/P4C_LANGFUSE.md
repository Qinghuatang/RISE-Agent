# P4-C Optional Langfuse Tracing

The P4-C tracing adapter is opt-in and uses the official Python SDK
`langfuse==4.15.4` through the `langfuse` project extra:

```bash
pip install -e '.[agent,postgres,service,api,langfuse]'
```

The SDK metadata declares Python `>=3.10,<4.0` and Pydantic `>=2,<3`, matching
this project's Python and Pydantic bounds. It also depends on HTTPX `<1` and
OpenTelemetry `<2`; it is isolated behind an optional extra so an ordinary
install does not need the SDK.

## Enable and configure

Tracing is off unless `RISE_LANGFUSE_ENABLED=true` (also accepts `1`, `yes`, or
`on`). When enabled, provide:

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- optional `LANGFUSE_BASE_URL` for a self-hosted instance or data region

When disabled, RISE does not import the Langfuse package, construct a client, or
make a tracing request. Missing credentials or client initialization failure
leaves tracing disabled and is recorded as a safe P4-B operational event.

The API owns its client in the FastAPI lifespan and shuts it down during
lifespan cleanup. The Worker owns one client for its process and shuts it down
when `run_forever()` exits. `shutdown()` flushes the SDK buffer; failures are
reported through failure-isolated P4-B telemetry and do not change task results.

## Trace structure

Task UUIDs are converted to stable 32-character lowercase hexadecimal W3C trace
IDs. API submission observations and Worker attempts reuse that trace ID. Every
Worker attempt creates a new root observation with its attempt and recovery
counts; after process recovery, later attempts retain the same task trace ID
but receive new observation IDs. Graph nodes, LLM generations, and Retrieval
observations are children of the active attempt where context propagation is
available. Graph nodes also supply the stable task trace ID explicitly, so a
thread-pool boundary can create a separate root observation in the same trace
instead of losing correlation.

P4-B's existing LLM and Retrieval wrappers create at most one Langfuse
observation per actual wrapper call. P4-B structured log metrics remain the
operational event stream; Langfuse is an optional trace sink, not a second
counter or durable progress store. Replayed, not-yet-checkpointed calls create
new observations under the later attempt and are distinguishable by the parent
attempt metadata. No exactly-once claim is made for remote LLM/Retrieval calls.

## Privacy boundary

The implementation uses only manual `start_as_current_observation()` contexts;
it does not attach a LangGraph `CallbackHandler`, use `@observe`, or pass
`input`/`output` values. Only task/thread/worker identifiers, attempt/recovery
counts, node/action/operation, safe status/error class, provider/model, duration,
request/retry counts, and available token usage are eligible metadata. Metadata
is allowlisted and scalar values are normalized before submission. Prompt,
model inputs and outputs, paper contents, Research State/Memory/Trajectory,
retrieval result text, credentials, lease tokens, and exception messages are
never passed to the SDK.

## Validation

Unit tests inject a fake Langfuse client/exporter and deterministic
FakeLLM/FakeRetrieval dependencies. No real Langfuse server is needed. A real
PostgreSQL URL is still required for the P3 API/Worker integration tests that
exercise tracing across service processes.

The SDK integration was also smoke-tested on Python 3.10 with the pinned
4.15.4 package and its OpenTelemetry in-memory exporter: a worker-attempt Span
and nested LLM Generation retained the supplied task Trace ID and parent link,
recorded token usage, and emitted no raw observation input/output fields. This
does not replace the PostgreSQL-backed API/Worker integration test.
