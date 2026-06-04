"""Tests for cron registry helpers."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from openharness.services.cron import (
    delete_cron_job,
    get_cron_job,
    load_cron_jobs,
    mark_job_run,
    next_run_time,
    set_job_enabled,
    upsert_cron_job,
    validate_cron_expression,
    validate_timezone,
)
from openharness.services.cron_scheduler import _notify_job_result


@pytest.fixture(autouse=True)
def _tmp_cron_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect cron registry to a temp directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(
        "openharness.services.cron.get_cron_registry_path",
        lambda: data_dir / "cron_jobs.json",
    )


class TestValidation:
    def test_valid_expressions(self) -> None:
        assert validate_cron_expression("* * * * *")
        assert validate_cron_expression("*/5 * * * *")
        assert validate_cron_expression("0 9 * * 1-5")
        assert validate_cron_expression("0 0 1 1 *")

    def test_invalid_expressions(self) -> None:
        assert not validate_cron_expression("")
        assert not validate_cron_expression("every 5 minutes")
        assert not validate_cron_expression("60 * * * *")
        assert not validate_cron_expression("* * * *")  # only 4 fields

    def test_next_run_time(self) -> None:
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        nxt = next_run_time("0 * * * *", base)
        assert nxt == datetime(2026, 1, 1, 1, 0, 0, tzinfo=timezone.utc)

    def test_next_run_time_with_timezone_returns_utc(self) -> None:
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        nxt = next_run_time("0 18 * * *", base, tz="Asia/Hong_Kong")
        assert nxt == datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_validate_timezone(self) -> None:
        assert validate_timezone("Asia/Hong_Kong")
        assert not validate_timezone("Asia/HongKong")


class TestCRUD:
    def test_empty_load(self) -> None:
        assert load_cron_jobs() == []

    def test_upsert_and_load(self) -> None:
        upsert_cron_job({"name": "test-job", "schedule": "*/5 * * * *", "command": "echo hi"})
        jobs = load_cron_jobs()
        assert len(jobs) == 1
        assert jobs[0]["name"] == "test-job"
        assert jobs[0]["enabled"] is True
        assert "next_run" in jobs[0]
        assert "created_at" in jobs[0]

    def test_upsert_preserves_notify_target(self) -> None:
        notify = {"type": "feishu_dm", "user_open_id": "ou_test"}
        upsert_cron_job({"name": "test-job", "schedule": "*/5 * * * *", "command": "echo hi", "notify": notify})
        job = get_cron_job("test-job")
        assert job is not None
        assert job["notify"] == notify

    def test_upsert_preserves_feishu_chat_notify_target(self) -> None:
        notify = {"type": "feishu_chat", "chat_id": "oc_test"}
        upsert_cron_job({"name": "test-job", "schedule": "*/5 * * * *", "command": "echo hi", "notify": notify})
        job = get_cron_job("test-job")
        assert job is not None
        assert job["notify"] == notify

    def test_upsert_preserves_agent_turn_payload(self) -> None:
        payload = {"kind": "agent_turn", "message": "check GitHub", "deliver": True, "channel": "feishu", "to": "ou_test"}
        upsert_cron_job({"name": "test-job", "schedule": "0 18 * * *", "timezone": "Asia/Hong_Kong", "payload": payload})
        job = get_cron_job("test-job")
        assert job is not None
        assert job["payload"] == payload
        assert job["timezone"] == "Asia/Hong_Kong"

    def test_upsert_replaces(self) -> None:
        upsert_cron_job({"name": "j1", "schedule": "* * * * *", "command": "echo 1"})
        upsert_cron_job({"name": "j1", "schedule": "0 * * * *", "command": "echo 2"})
        jobs = load_cron_jobs()
        assert len(jobs) == 1
        assert jobs[0]["command"] == "echo 2"

    def test_delete(self) -> None:
        upsert_cron_job({"name": "j1", "schedule": "* * * * *", "command": "echo 1"})
        assert delete_cron_job("j1") is True
        assert load_cron_jobs() == []

    def test_delete_missing(self) -> None:
        assert delete_cron_job("nope") is False

    def test_get_job(self) -> None:
        upsert_cron_job({"name": "j1", "schedule": "* * * * *", "command": "echo 1"})
        job = get_cron_job("j1")
        assert job is not None
        assert job["name"] == "j1"

    def test_get_missing(self) -> None:
        assert get_cron_job("nope") is None

    def test_sorted_output(self) -> None:
        upsert_cron_job({"name": "z-job", "schedule": "* * * * *", "command": "z"})
        upsert_cron_job({"name": "a-job", "schedule": "* * * * *", "command": "a"})
        jobs = load_cron_jobs()
        assert [j["name"] for j in jobs] == ["a-job", "z-job"]


class TestToggle:
    def test_enable_disable(self) -> None:
        upsert_cron_job({"name": "j1", "schedule": "* * * * *", "command": "echo 1"})
        assert set_job_enabled("j1", False) is True
        job = get_cron_job("j1")
        assert job is not None
        assert job["enabled"] is False

        assert set_job_enabled("j1", True) is True
        job = get_cron_job("j1")
        assert job is not None
        assert job["enabled"] is True

    def test_toggle_missing(self) -> None:
        assert set_job_enabled("nope", True) is False


class TestMarkRun:
    def test_mark_success(self) -> None:
        upsert_cron_job({"name": "j1", "schedule": "*/5 * * * *", "command": "echo ok"})
        mark_job_run("j1", success=True)
        job = get_cron_job("j1")
        assert job is not None
        assert job["last_status"] == "success"
        assert "last_run" in job

    def test_mark_failure(self) -> None:
        upsert_cron_job({"name": "j1", "schedule": "*/5 * * * *", "command": "false"})
        mark_job_run("j1", success=False)
        job = get_cron_job("j1")
        assert job is not None
        assert job["last_status"] == "failed"

    def test_mark_missing_is_noop(self) -> None:
        # Should not raise
        mark_job_run("nope", success=True)


class TestCorruptData:
    def test_corrupt_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        bad_file = tmp_path / "data" / "cron_jobs.json"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text("{not valid json", encoding="utf-8")
        monkeypatch.setattr(
            "openharness.services.cron.get_cron_registry_path",
            lambda: bad_file,
        )
        assert load_cron_jobs() == []

    def test_non_list_json(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        bad_file = tmp_path / "data" / "cron_jobs.json"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        monkeypatch.setattr(
            "openharness.services.cron.get_cron_registry_path",
            lambda: bad_file,
        )
        assert load_cron_jobs() == []


@pytest.mark.asyncio
async def test_notify_job_result_sends_feishu_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[dict[str, str | None]] = []

    async def _fake_send_feishu_chat(*, chat_id: str, content: str, workspace: str | None = None) -> None:
        sent.append({"chat_id": chat_id, "content": content, "workspace": workspace})

    monkeypatch.setattr("ohmo.gateway.notify.send_feishu_chat", _fake_send_feishu_chat)

    entry = {
        "status": "success",
        "returncode": 0,
        "started_at": "2026-06-04T00:00:00Z",
        "ended_at": "2026-06-04T00:00:01Z",
        "stdout": "CRON_OK",
    }
    await _notify_job_result(
        {
            "name": "nightly",
            "notify": {
                "type": "feishu_chat",
                "chat_id": "oc_test",
                "workspace": "/repo",
            },
        },
        entry,
    )

    assert len(sent) == 1
    assert sent[0]["chat_id"] == "oc_test"
    assert sent[0]["workspace"] == "/repo"
    assert "Cron job finished: nightly" in str(sent[0]["content"])
    assert "Status: success (rc=0)" in str(sent[0]["content"])
    assert "CRON_OK" in str(sent[0]["content"])
    assert entry["notification_status"] == "sent"


@pytest.mark.asyncio
async def test_notify_job_result_feishu_chat_requires_chat_id() -> None:
    entry = {"status": "success", "returncode": 0}

    await _notify_job_result(
        {
            "name": "nightly",
            "notify": {"type": "feishu_chat"},
        },
        entry,
    )

    assert entry["notification_status"] == "failed"
    assert entry["notification_error"] == "missing notify.chat_id"
