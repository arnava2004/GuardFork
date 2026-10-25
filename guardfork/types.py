"""Trajectory schema shared by guards, fork selection and the oracle."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

N_DEMO_ASSISTANT_TURNS = 10


class TrajectoryStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    MAX_TURNS = "max_turns"
    CONTEXT_OVERFLOW = "context_overflow"
    REPLAY_DIVERGED = "replay_diverged"
    ERROR = "error"


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]

    @property
    def command(self) -> str | None:
        cmd = self.args.get("command")
        return cmd if isinstance(cmd, str) else None


@dataclass(frozen=True)
class Turn:
    """One assistant turn. `turn` is 1-based over live turns (s_t is the state before it);
    `assistant_index` counts demo turns too, so live turns have assistant_index >= 10."""

    turn: int
    assistant_index: int
    content: str
    reasoning: str
    tool_calls: tuple[ToolCall, ...] = ()
    invalid_tool_calls: tuple[dict[str, Any], ...] = ()
    observation: str | None = None
    session_reset: bool = False

    @property
    def command(self) -> str | None:
        return self.tool_calls[0].command if len(self.tool_calls) == 1 else None


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    turns: tuple[Turn, ...]
    reward: float | None
    status: TrajectoryStatus
    allowed_apps: frozenset[str] = field(default_factory=frozenset)
    completion_status: bool | None = None

    @property
    def T(self) -> int:  # noqa: N802
        return len(self.turns)
