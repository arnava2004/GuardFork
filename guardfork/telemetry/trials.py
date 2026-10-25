"""Read Harness-Zero / Harbor trial artifacts into flat records.

Layout (Harness-Zero `TrialStore` under Harbor)::

    <job>/<trial>/result.json                         Harbor verifier result
    <job>/<trial>/agent/teacher/trial/trajectory.jsonl one `reviewed_turn` event per live turn
    <job>/<trial>/agent/teacher/trial/result.json      TrialStore result (local runs)

`trajectory.jsonl` is written by the review middleware even under PassthroughReviewer.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_TURNS_DEFAULT = 40
UNPARSED_MARKUP = ("<tool_call>", "</tool_call>", "<function=")
SUBAGENT_RE = re.compile(r"(?:^|[\s;&|(])agent\s+[\"']")


@dataclass
class TurnRecord:
    index: int
    command: str | None
    parse_failure: bool
    parse_reasons: list[str] = field(default_factory=list)
    subagent_call: bool = False
    session_reset: bool = False


@dataclass
class TrialRecord:
    trial_dir: Path
    job: str
    task_id: str | None
    reward: float | None
    exception: Any
    wall_seconds: float | None
    turns: list[TurnRecord]
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def passed(self) -> bool:
        return self.reward is not None and self.reward >= 1.0


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _has_truthy_key(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        if obj.get(key):
            return True
        return any(_has_truthy_key(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_truthy_key(v, key) for v in obj)
    return False


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_turn(index: int, event: dict[str, Any]) -> TurnRecord:
    raw = event.get("raw_original") or {}
    original = event.get("original") or {}
    reasons: list[str] = []
    if raw.get("invalid_tool_calls"):
        reasons.append("invalid_tool_calls")
    if event.get("candidate_issue"):
        reasons.append("candidate_issue")
    content = original.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content)
    if any(m in content for m in UNPARSED_MARKUP):
        reasons.append("markup_in_content")
    command = ((original.get("tool_call") or {}).get("command")) or None
    return TurnRecord(
        index=index,
        command=command,
        parse_failure=bool(reasons),
        parse_reasons=reasons,
        subagent_call=bool(command and SUBAGENT_RE.search(command)),
        session_reset=_has_truthy_key(event, "session_reset"),
    )


def load_trial(teacher_dir: Path, job: str) -> TrialRecord:
    events = [
        e for e in _read_jsonl(teacher_dir / "trial" / "trajectory.jsonl")
        if e.get("type", "reviewed_turn") == "reviewed_turn"
    ]
    turns = [parse_turn(i, e) for i, e in enumerate(events)]
    trial_dir = teacher_dir.parent.parent
    harbor = _read_json(trial_dir / "result.json")
    local = _read_json(teacher_dir / "trial" / "result.json")

    reward: float | None = None
    exception: Any = None
    task_id = events[0].get("task_id") if events else None
    wall: float | None = None
    if local is not None and "reward" in local:
        reward = float(local["reward"])
        task_id = task_id or local.get("task_id")
    if harbor is not None:
        exception = harbor.get("exception_info")
        with contextlib.suppress(KeyError, TypeError, ValueError):
            reward = float(harbor["verifier_result"]["rewards"]["reward"])
        task_id = task_id or harbor.get("task_name") or harbor.get("task_id")
        start, end = _parse_time(harbor.get("started_at")), _parse_time(harbor.get("finished_at"))
        if start and end:
            wall = (end - start).total_seconds()
    elif local is None:
        exception = "missing result.json"
    return TrialRecord(trial_dir, job, task_id, reward, exception, wall, turns, events)


def iter_trials(job_dir: Path, job: str | None = None) -> Iterator[TrialRecord]:
    """Yield every trial under a Harbor job dir, including ones that crashed before
    the TrialStore existed (Harbor `result.json` beside `config.json`, no agent logs)."""
    label = job or job_dir.name
    seen: set[Path] = set()
    for trial_subdir in sorted(job_dir.glob("**/teacher/trial")):
        record = load_trial(trial_subdir.parent, label)
        seen.add(record.trial_dir.resolve())
        yield record
    for result in sorted(job_dir.glob("**/result.json")):
        trial_dir = result.parent
        if trial_dir.resolve() in seen or not (trial_dir / "config.json").is_file():
            continue
        if "teacher" in trial_dir.parts:
            continue
        yield load_trial(trial_dir / "agent" / "teacher", label)
