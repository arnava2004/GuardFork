from __future__ import annotations

import json
from pathlib import Path

import pytest


def _event(task_id: str, *, command: str | None = "echo hi", content: str = "",
           invalid: list | None = None, issue: str | None = None, extra: dict | None = None):
    event = {
        "type": "reviewed_turn",
        "task_id": task_id,
        "trial": 0,
        "candidate_id": "c",
        "original": {
            "reasoning": "",
            "content": content,
            "tool_call": {"name": "execute", "command": command} if command else None,
        },
        "raw_original": {"invalid_tool_calls": invalid or []},
        "candidate_issue": issue,
    }
    event.update(extra or {})
    return event


@pytest.fixture
def make_trial(tmp_path: Path):
    """Write a fake Harbor trial dir; returns the job dir."""

    def _make(job: str, trial: str, task_id: str, reward: float | None, events: list[dict],
              wall_s: float = 60.0, write_teacher: bool = True) -> Path:
        job_dir = tmp_path / job
        trial_dir = job_dir / trial
        trial_dir.mkdir(parents=True)
        (trial_dir / "config.json").write_text("{}")
        result = {
            "task_name": task_id,
            "started_at": "2026-10-24T00:00:00+00:00",
            "finished_at": f"2026-10-24T00:{int(wall_s // 60):02d}:{int(wall_s % 60):02d}+00:00",
            "exception_info": None if reward is not None else {"type": "Boom"},
        }
        if reward is not None:
            result["verifier_result"] = {"rewards": {"reward": reward}}
        (trial_dir / "result.json").write_text(json.dumps(result))
        if write_teacher:
            tdir = trial_dir / "agent" / "teacher" / "trial"
            tdir.mkdir(parents=True)
            (tdir / "trajectory.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events))
        return job_dir

    return _make


@pytest.fixture
def event():
    return _event
