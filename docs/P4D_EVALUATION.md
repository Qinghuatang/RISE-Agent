# P4-D Engineering Evaluation

P4-D adds a read-only report command. It aggregates existing `tasks`,
`task_events`, the latest LangGraph checkpoint, and optional P4-B structured
JSON logs. It creates no metric table and never copies Research State, Memory,
Trajectory, Replay, or Proposal content into the report.

## Install and run against PostgreSQL

Install the existing persistence/service extras, then point the evaluator at
the business database and (if separate) the checkpoint database:

```bash
pip install -e '.[agent,postgres,service]'
export RISE_DATABASE_URL='postgresql://USER@127.0.0.1:5432/RISE_DATABASE'
research-agent-eval \
  --logs ./var/log/api.jsonl \
  --logs ./var/log/worker.jsonl \
  --json-out ./var/reports/rise-evaluation.json \
  --markdown-out ./var/reports/rise-evaluation.md
```

`RISE_CHECKPOINT_DATABASE_URL` or `--checkpoint-database-url` can select a
separate checkpoint database; otherwise it uses the business URL. `--limit N`
restricts the report to the newest N tasks. Evaluation does not run migrations
or `PostgresSaver.setup()`. It reads checkpoint and business data only.

The log arguments must be captured JSON output from the API and Worker, for
example with `RISE_LOG_JSON=true` and `tee`. Call counts are marked as observed
only because console logs can be incomplete. JSON log `event_id` values prevent
overlapping files from counting the same emitted record twice. Older logs
without `event_id` are accepted and reported as unkeyed observations. If no log
files are supplied, LLM/Retrieval counts and token usage are `null` (unknown),
not zero.

## Metric definitions

| Metric | Definition and denominator | Source |
|---|---|---|
| Business task counts and proportions | Each `task_status` count divided by all tasks in the report | `tasks` |
| RISE outcome distribution | Each non-null outcome divided by tasks with a non-null `rise_outcome` | `tasks.rise_outcome` |
| Scientific failure | `task_status=completed` and outcome `failed` or `finalization_failed` | `tasks` |
| Infrastructure/input failure | `task_status=failed`, separated by `failure_kind` | `tasks` |
| Proposal submission success | Committed tasks with a final proposal divided by all committed tasks; absent or unreadable proposal checkpoints count as not found | checkpoint `final_proposal` + `tasks.rise_outcome` |
| Queue time | `started_at - created_at`; tasks without a start timestamp are unknown | `tasks` |
| Execution time | `finished_at - started_at`; tasks without both timestamps are unknown | `tasks` |
| Action counts | Unique Action-kind replay sequence records in each latest checkpoint; failed Action attempts remain counted as attempts and are listed separately. Tasks without readable checkpoints are reported as unknown coverage | checkpoint `replay_records` and `trajectory_events` |
| Budget use | Sum of final `steps_used` and `searches_used` over readable checkpoints; absent values are unknown | checkpoint `budget` |
| LLM/Retrieval calls | Observed `llm.call` / `retrieval.call` structured records associated with report task IDs; completeness is not assumed | supplied JSON logs |
| Token use | Sums only calls with all three token fields; partial/missing usage increments unknown count and never becomes zero | supplied LLM call log records |
| Recovery outcomes | `recovery_count` and final task status; `completed` means infrastructure recovery reached a confirmed RISE terminal outcome, including a scientific RISE failure | `tasks`, with error class/kind counts |
| Checkpoint/event consistency | Compare replay and trajectory projection keys/content, task RISE outcome, action-step count versus replay Action count, and final budget snapshot | checkpoint + `task_events` |

Latency summaries include known and unknown sample counts. Percentages with an
empty denominator and unavailable measurements are `null`.

## Deterministic FakeLLM/FakeRetrieval sample

The sample uses the repository's fixed `test_m11` seed, Markdown, FakeLLM
responses, and FakeRetrieval result. It runs the real LangGraph trajectory and
feeds its checkpoint-shaped state, replay, trajectory, and captured telemetry
through the same report builder. Task timestamps are fixed; Fake call latency
and missing Fake token usage are left unknown so the sample output is
byte-for-byte repeatable.

```bash
PYTHONPATH=src:tests python tests/evaluation_sample.py \
  --json-out /tmp/rise-evaluation-sample.json \
  --markdown-out /tmp/rise-evaluation-sample.md
```

The sample report should show one completed/committed task, a successful
proposal checkpoint, matching replay/trajectory projections, five Action
attempts, one scientific search, and observed Fake LLM/Retrieval calls. Its
Markdown is limited to aggregate engineering metrics and contains no proposal
text or paper content.

## Database integration acceptance

The evaluator itself is unit-tested with fixed records and Fake RISE execution.
The PostgreSQL-backed API/Worker/Checkpoint trace acceptance still requires
the dedicated database setup described in
[`P3E_LOCAL_INTEGRATION.md`](P3E_LOCAL_INTEGRATION.md). No mocked database run is
reported as PostgreSQL validation.
