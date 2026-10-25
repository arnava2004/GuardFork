"""TrialRecord (raw Harness-Zero events) -> `guardfork.types.Trajectory` for guards/oracle.

Event i holds the context *before* live turn i+1, so the observation of turn i+1 is the
trailing tool message in event i+2's context. The last turn has no recorded observation.
"""

from __future__ import annotations

import json
from typing import Any

from guardfork.telemetry.trials import MAX_TURNS_DEFAULT, TrialRecord
from guardfork.types import N_DEMO_ASSISTANT_TURNS, ToolCall, Trajectory, TrajectoryStatus, Turn


def message_text(message: dict[str, Any] | None) -> str:
    if not message:
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return "" if content is None else json.dumps(content)


def _is_tool_message(message: dict[str, Any]) -> bool:
    return message.get("type") == "tool" or message.get("role") == "tool"


def observation_after(events: list[dict[str, Any]], i: int) -> str | None:
    """Observation produced by the action in event i (0-based), if recorded."""
    if i + 1 >= len(events):
        return None
    context = events[i + 1].get("context") or []
    if context and _is_tool_message(context[-1]):
        return message_text(context[-1])
    return None


def _tool_calls(raw: dict[str, Any]) -> tuple[ToolCall, ...]:
    calls = []
    for tc in raw.get("tool_calls") or []:
        args = tc.get("args")
        if args is None and isinstance(tc.get("function"), dict):
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc["function"].get("arguments")}
        calls.append(ToolCall(name=tc.get("name") or (tc.get("function") or {}).get("name", ""),
                              args=args if isinstance(args, dict) else {}))
    return tuple(calls)


def status_of(record: TrialRecord, max_turns: int = MAX_TURNS_DEFAULT) -> TrajectoryStatus:
    if record.reward is None:
        return TrajectoryStatus.ERROR
    if record.passed:
        return TrajectoryStatus.PASSED
    if record.n_turns >= max_turns:
        return TrajectoryStatus.MAX_TURNS
    return TrajectoryStatus.FAILED


def to_trajectory(record: TrialRecord) -> Trajectory:
    turns = []
    for i, event in enumerate(record.events):
        raw = event.get("raw_original") or {}
        original = event.get("original") or {}
        turns.append(Turn(
            turn=i + 1,
            assistant_index=N_DEMO_ASSISTANT_TURNS + i,
            content=original.get("content") or "",
            reasoning=original.get("reasoning") or "",
            tool_calls=_tool_calls(raw),
            invalid_tool_calls=tuple(raw.get("invalid_tool_calls") or ()),
            observation=observation_after(record.events, i),
            session_reset=record.turns[i].session_reset if i < len(record.turns) else False,
        ))
    return Trajectory(
        task_id=record.task_id or "",
        turns=tuple(turns),
        reward=record.reward,
        status=status_of(record),
    )
