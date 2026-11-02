"""Real PostgreSQL tests for the P3-D FastAPI and SSE surface."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.base import empty_checkpoint
from sqlalchemy import text
from sqlalchemy.engine import make_url

from research_agent.controller.budget import BudgetTracker
from research_agent.runtime.checkpointer import open_postgres_checkpointer
from research_agent.runtime.graph_runner import LangGraphTrajectoryRunner
from research_agent.runtime.graph_state import RISEGraphState
from research_agent.service.api import APISettings, _event_stream, create_app
from research_agent.service.db import create_business_engine

from test_m11 import modules, reading, seed


POSTGRES_URL = os.environ.get("RISE_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set RISE_TEST_POSTGRES_URL to run real PostgreSQL API integration tests",
)


@pytest.fixture
def api_client(tmp_path: Path):
    assert POSTGRES_URL is not None
    admin_engine = create_business_engine(POSTGRES_URL)
    schema = f"rise_p3d_{uuid4().hex}"
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    scoped_url = make_url(POSTGRES_URL).update_query_dict(
        {"options": f"-csearch_path={schema}"}
    ).render_as_string(hide_password=False)
    settings = APISettings(
        database_url=scoped_url,
        paper_store_root=tmp_path / "papers",
        max_markdown_bytes=1024,
        sse_poll_interval=0.02,
        sse_heartbeat_interval=0.05,
    )
    try:
        with TestClient(create_app(settings)) as client:
            yield client
    finally:
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin_engine.dispose()


def _create(client: TestClient, markdown: str = "# Submitted paper\n\nAbstract text.") -> dict:
    response = client.post("/api/tasks", json={"markdown": markdown})
    assert response.status_code == 202, response.text
    return response.json()


def _save_terminal_state(client: TestClient, task_id: UUID, state: RISEGraphState) -> None:
    resources = client.app.state.resources
    payload = state.evolve(run_id=str(task_id)).to_payload()
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = payload
    versions = {key: 1 for key in payload}
    checkpoint["channel_versions"] = versions
    resources.checkpointer.put(
        {"configurable": {"thread_id": str(task_id), "checkpoint_ns": ""}},
        checkpoint,
        {"source": "input", "step": -1, "writes": {}},
        versions,
    )
    task = resources.repository.claim_next_queued(worker_id="api-test-worker", lease_seconds=30)
    assert task is not None and task.id == task_id
    resources.repository.mark_completed(
        task_id=task_id,
        worker_id="api-test-worker",
        lease_token=task.lease_token,
        rise_outcome=state.rise_status,
        error_type=state.error_type,
        error_summary=state.error_message,
    )


def test_api_request_logs_task_correlation_without_input_content(
    api_client: TestClient,
    caplog: pytest.LogCaptureFixture,
):
    markdown = "# Private test manuscript\n\nDo not emit this text to logs."
    with caplog.at_level(logging.INFO, logger="research_agent.metrics"):
        response = api_client.post(
            "/api/tasks",
            json={"markdown": markdown},
            headers={"X-Request-ID": "p4b-api-test-request"},
        )

    assert response.status_code == 202
    assert response.headers["X-Request-ID"] == "p4b-api-test-request"
    task_id = response.json()["task_id"]
    structured = [
        record
        for record in caplog.records
        if getattr(record, "rise_event", None)
        in {"api.task.submitted", "api.request.finished"}
    ]
    assert {record.rise_event for record in structured} == {
        "api.task.submitted",
        "api.request.finished",
    }
    for record in structured:
        fields = record.rise_fields
        assert fields["task_id"] == task_id
        assert fields["thread_id"] == task_id
        assert fields["request_id"] == "p4b-api-test-request"
        assert markdown not in str(fields)
        assert "Do not emit" not in record.getMessage()


def test_health_submit_text_and_upload_and_input_validation(api_client: TestClient):
    assert api_client.get("/health/live").json() == {"status": "alive"}
    ready = api_client.get("/health/ready")
    assert ready.status_code == 200 and ready.json() == {"status": "ready"}

    text_submission = _create(api_client)
    assert text_submission["task_status"] == "queued"
    assert text_submission["rise_outcome"] is None
    task = api_client.get(f"/api/tasks/{text_submission['task_id']}")
    assert task.status_code == 200
    assert task.json()["progress"]["latest_event_type"] == "task_queued"
    artifact_rows = api_client.app.state.resources.repository.list_artifacts(
        UUID(text_submission["task_id"])
    )
    assert len(artifact_rows) == 1
    assert artifact_rows[0].artifact_type == "input_markdown"

    uploaded = api_client.post(
        "/api/tasks",
        files={"file": ("paper.md", b"# Upload\n\nBody", "text/markdown")},
    )
    assert uploaded.status_code == 202, uploaded.text

    for files, expected in [
        ({"file": ("paper.txt", b"content", "text/plain")}, 422),
        ({"file": ("bad.md", b"\xff", "text/markdown")}, 422),
        ({"file": ("large.md", b"x" * 1025, "text/markdown")}, 413),
    ]:
        assert api_client.post("/api/tasks", files=files).status_code == expected
    assert api_client.post("/api/tasks", json={"markdown": "   "}).status_code == 422
    assert api_client.post("/api/tasks", content="{}", headers={"content-type": "text/plain"}).status_code == 415


def test_idempotency_listing_cursor_and_safe_cancel(api_client: TestClient):
    headers = {"Idempotency-Key": "same-submission"}
    first = api_client.post("/api/tasks", json={"markdown": "# Same"}, headers=headers)
    repeated = api_client.post("/api/tasks", json={"markdown": "# Same"}, headers=headers)
    assert first.status_code == repeated.status_code == 202
    assert first.json()["task_id"] == repeated.json()["task_id"]
    assert repeated.json()["idempotent_replay"] is True
    conflict = api_client.post("/api/tasks", json={"markdown": "# Different"}, headers=headers)
    assert conflict.status_code == 409

    second = _create(api_client, "# Second")
    third = _create(api_client, "# Third")
    page_one = api_client.get("/api/tasks?limit=2")
    assert page_one.status_code == 200
    assert len(page_one.json()["items"]) == 2
    cursor = page_one.json()["next_cursor"]
    assert cursor
    page_two = api_client.get("/api/tasks", params={"limit": 2, "cursor": cursor})
    all_ids = {item["task_id"] for item in page_one.json()["items"] + page_two.json()["items"]}
    assert all_ids == {first.json()["task_id"], second["task_id"], third["task_id"]}
    assert api_client.get("/api/tasks?cursor=invalid").status_code == 422

    cancel = api_client.post(f"/api/tasks/{second['task_id']}/cancel")
    assert cancel.status_code == 200
    assert cancel.json()["task_status"] == "cancelled"
    assert cancel.json()["rise_outcome"] is None
    assert api_client.post(f"/api/tasks/{second['task_id']}/cancel").json()["task_status"] == "cancelled"

    api_client.post(f"/api/tasks/{first.json()['task_id']}/cancel")
    third_id = UUID(third["task_id"])
    repository = api_client.app.state.resources.repository
    claimed = repository.claim_next_queued(worker_id="api-cancel-test-worker", lease_seconds=30)
    assert claimed is not None and claimed.id == third_id
    assert api_client.get(f"/api/tasks/{third_id}").json()["task_status"] == "running"
    requested = api_client.post(f"/api/tasks/{third_id}/cancel")
    assert requested.status_code == 200
    assert requested.json()["task_status"] == "cancel_requested"
    repository.mark_cancelled(
        task_id=third_id,
        worker_id="api-cancel-test-worker",
        lease_token=claimed.lease_token,
    )
    assert api_client.get(f"/api/tasks/{third_id}").json()["task_status"] == "cancelled"
    assert api_client.get(f"/api/tasks/{uuid4()}").status_code == 404


def test_result_reads_checkpoint_and_keeps_task_status_separate_from_rise_outcome(
    api_client: TestClient,
):
    base = LangGraphTrajectoryRunner(
        modules(), budget=BudgetTracker(max_steps=5, max_searches=1)
    ).run(seed(), full_paper_reading_context=reading())
    assert base.rise_status == "committed"
    variants = [
        base,
        base.evolve(rise_status="stopped", commit_snapshot=None, final_proposal=None),
        base.evolve(
            rise_status="failed",
            commit_snapshot=None,
            final_proposal=None,
            error_type="SyntheticScientificError",
            error_message="do not expose this prompt or paper text",
        ),
        base.evolve(
            rise_status="finalization_failed",
            final_proposal=None,
            error_type="SyntheticFinalizerError",
            error_message="do not expose this error body",
        ),
    ]
    for expected in variants:
        accepted = _create(api_client, f"# Result {expected.rise_status}")
        task_id = UUID(accepted["task_id"])
        _save_terminal_state(api_client, task_id, expected)
        result = api_client.get(f"/api/tasks/{task_id}/result")
        assert result.status_code == 200, result.text
        payload = result.json()
        assert payload["task_status"] == "completed"
        assert payload["rise_outcome"] == expected.rise_status
        assert payload["replay_record_count"] == len(expected.replay_records)
        if expected.rise_status == "committed":
            assert payload["final_proposal"] is not None
        else:
            assert payload["final_proposal"] is None
        if expected.rise_status == "finalization_failed":
            assert payload["commit_snapshot"] is not None
        assert "do not expose this prompt or paper text" not in result.text
        assert "do not expose this error body" not in result.text

    queued = _create(api_client, "# Waiting")
    assert api_client.get(f"/api/tasks/{queued['task_id']}/result").status_code == 409
    cancelled = api_client.post(f"/api/tasks/{queued['task_id']}/cancel")
    cancelled_result = api_client.get(f"/api/tasks/{queued['task_id']}/result")
    assert cancelled.json()["task_status"] == "cancelled"
    assert cancelled_result.status_code == 200
    assert cancelled_result.json()["rise_outcome"] is None

    failed = _create(api_client, "# Infrastructure failure")
    api_client.app.state.resources.repository.fail_before_start(
        task_id=UUID(failed["task_id"]),
        failure_kind="infrastructure",
        error_type="DatabaseUnavailable",
        error_summary="api_key=do-not-return-this-value",
    )
    failed_result = api_client.get(f"/api/tasks/{failed['task_id']}/result")
    assert failed_result.status_code == 200
    assert failed_result.json()["task_status"] == "failed"
    assert failed_result.json()["rise_outcome"] is None
    assert "do-not-return-this-value" not in failed_result.text


def test_result_reader_accepts_a_real_langgraph_checkpoint(api_client: TestClient):
    markdown = "\n".join(reading().parsed.lines) + "\n"
    accepted = _create(api_client, markdown)
    task_id = UUID(accepted["task_id"])
    resources = api_client.app.state.resources
    with open_postgres_checkpointer(resources.settings.database_url) as writer:
        runner = LangGraphTrajectoryRunner(
            modules(),
            budget=BudgetTracker(max_steps=5, max_searches=1),
            checkpointer=writer,
            paper_store=resources.paper_store,
        )
        state = runner.start_persistent(seed(), thread_id=str(task_id), markdown=markdown)
    assert state.rise_status == "committed"
    assert state.paper_input.locator == resources.repository.get_task(task_id).input_locator
    claimed = resources.repository.claim_next_queued(worker_id="api-test-worker", lease_seconds=30)
    assert claimed is not None and claimed.id == task_id
    resources.repository.mark_completed(
        task_id=task_id,
        worker_id="api-test-worker",
        lease_token=claimed.lease_token,
        rise_outcome=state.rise_status,
    )

    response = api_client.get(f"/api/tasks/{task_id}/result")
    assert response.status_code == 200, response.text
    assert response.json()["rise_outcome"] == "committed"
    assert response.json()["final_proposal"] is not None


def test_sse_sends_idle_heartbeat_and_closes_after_terminal_event(api_client: TestClient):
    accepted = _create(api_client)
    task_id = UUID(accepted["task_id"])
    resources = api_client.app.state.resources
    repository = resources.repository
    claimed = repository.claim_next_queued(worker_id="heartbeat-test-worker", lease_seconds=30)
    assert claimed is not None and claimed.id == task_id

    class ConnectedRequest:
        async def is_disconnected(self) -> bool:
            return False

    async def collect() -> list[str]:
        stream = _event_stream(ConnectedRequest(), resources, task_id, 0)
        observed = [await anext(stream)]
        while not any(line == ": keep-alive\n\n" for line in observed):
            observed.append(await anext(stream))
        repository.request_cancel(task_id)
        repository.mark_cancelled(
            task_id=task_id,
            worker_id="heartbeat-test-worker",
            lease_token=claimed.lease_token,
        )
        observed.append(await anext(stream))
        observed.append(await anext(stream))
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        await stream.aclose()
        return observed

    observed = asyncio.run(collect())
    assert observed[0].startswith("id: 1")
    assert ": keep-alive\n\n" in observed
    assert any("event: task_cancelled\n" in block for block in observed)


def test_sse_reconnects_by_sequence_without_duplicates_or_sensitive_payloads(
    api_client: TestClient,
):
    task = _create(api_client)
    task_id = UUID(task["task_id"])
    repository = api_client.app.state.resources.repository
    repository.append_event(
        task_id=task_id,
        event_type="rise_action_confirmed",
        event_key=f"test:{task_id}:action",
        action="VERIFY_RELATED_WORK",
        payload={
            "step": 1,
            "kind": "action",
            "budget": {"steps_used": 1, "searches_used": 0},
            "error": "private prompt and full paper must not be sent",
        },
    )
    api_client.post(f"/api/tasks/{task_id}/cancel")

    full = api_client.get(f"/api/tasks/{task_id}/events")
    assert full.status_code == 200
    ids = [int(line[4:]) for line in full.text.splitlines() if line.startswith("id: ")]
    assert ids == list(range(1, len(ids) + 1))
    assert "private prompt and full paper" not in full.text
    assert "payload" in full.text and "VERIFY_RELATED_WORK" in full.text

    resumed = api_client.get(
        f"/api/tasks/{task_id}/events", headers={"Last-Event-ID": "1"}
    )
    resumed_ids = [int(line[4:]) for line in resumed.text.splitlines() if line.startswith("id: ")]
    assert resumed_ids == ids[1:]
    assert api_client.get(
        f"/api/tasks/{task_id}/events", headers={"Last-Event-ID": "bad"}
    ).status_code == 400
    assert api_client.get(f"/api/tasks/{uuid4()}/events").status_code == 404
