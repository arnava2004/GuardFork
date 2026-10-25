"""Replay-from-zero forking as Harbor trials.

A fork at turn t re-creates the pre-action state s_t in a fresh AppWorld container.
Harness-Zero is imported lazily so this module loads without it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from guardfork.fork.select import forkable_turns
from guardfork.telemetry.to_trajectory import observation_after, to_trajectory
from guardfork.telemetry.trials import TrialRecord

Execute = Callable[[str], Awaitable[str]]


class ReplayDiverged(RuntimeError):
    """A replayed command produced a different observation than the source trial."""

    def __init__(self, turn: int, expected: str, got: str) -> None:
        super().__init__(f"replay diverged at turn {turn}")
        self.turn, self.expected, self.got = turn, expected, got


class NotForkable(ValueError):
    """Turn t is after a session_reset or a subagent call, or t >= T."""


@dataclass(frozen=True)
class ReplayPlan:
    source_trial: Path
    task_id: str
    fork_turn: int
    initial_context_sha256: str
    commands: tuple[str, ...]
    expected_observations: tuple[str, ...]
    message_prefix: tuple[dict[str, Any], ...]

    @property
    def max_turns(self) -> int:
        return 40 - (self.fork_turn - 1)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_trial"] = str(self.source_trial)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ReplayPlan:
        return cls(
            source_trial=Path(data["source_trial"]),
            task_id=data["task_id"],
            fork_turn=int(data["fork_turn"]),
            initial_context_sha256=data["initial_context_sha256"],
            commands=tuple(data["commands"]),
            expected_observations=tuple(data["expected_observations"]),
            message_prefix=tuple(data["message_prefix"]),
        )


def _context_path(trial_dir: Path) -> Path:
    for candidate in (
        trial_dir / "agent" / "appworld_initial_context.json",
        trial_dir / "logs" / "agent" / "appworld_initial_context.json",
        trial_dir / "appworld_initial_context.json",
    ):
        if candidate.is_file():
            return candidate
    return trial_dir / "agent" / "appworld_initial_context.json"


def initial_context_sha256(trial_dir: Path) -> str:
    """sha256 of the source trial's `appworld_initial_context.json` (field or file)."""
    path = _context_path(trial_dir)
    if not path.is_file():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("sha256"):
        return str(data["sha256"])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_command(event: dict[str, Any]) -> str:
    original = event.get("original") or {}
    tc = original.get("tool_call") or {}
    if isinstance(tc, dict) and tc.get("command"):
        return str(tc["command"])
    for call in (event.get("raw_original") or {}).get("tool_calls") or []:
        args = call.get("args") or {}
        if isinstance(args, dict) and args.get("command"):
            return str(args["command"])
        fn = call.get("function") or {}
        try:
            parsed = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict) and parsed.get("command"):
            return str(parsed["command"])
    return ""


def build_replay_plan(record: TrialRecord, fork_turn: int) -> ReplayPlan:
    """Commands/observations 1..t-1 and the message prefix needed to reach s_t.

    Raises NotForkable for turns rejected by `fork.select.forkable_turns`.
    """
    traj = to_trajectory(record)
    allowed = forkable_turns(traj)
    if fork_turn not in allowed:
        raise NotForkable(f"turn {fork_turn} is not forkable (allowed {allowed})")
    prefix_events = record.events[: fork_turn - 1]
    commands = tuple(_event_command(e) for e in prefix_events)
    observations = tuple(observation_after(record.events, i) or ""
                         for i in range(fork_turn - 1))
    last = record.events[fork_turn - 1] if record.events else {}
    messages = []
    if last.get("system_message"):
        messages.append(last["system_message"])
    messages.extend(last.get("context") or [])
    return ReplayPlan(
        source_trial=record.trial_dir,
        task_id=record.task_id or traj.task_id,
        fork_turn=fork_turn,
        initial_context_sha256=initial_context_sha256(record.trial_dir),
        commands=commands,
        expected_observations=observations,
        message_prefix=tuple(messages),
    )


async def replay_prefix(commands: tuple[str, ...], expected_observations: tuple[str, ...],
                        execute: Execute) -> None:
    """Replay commands 1..t-1, raising ReplayDiverged on observation mismatch."""
    if len(commands) != len(expected_observations):
        raise ValueError("commands and expected_observations must align")
    for i, (cmd, expected) in enumerate(zip(commands, expected_observations, strict=True),
                                        start=1):
        got = await execute(cmd)
        if expected and got != expected:
            raise ReplayDiverged(i, expected, got)


def make_fork_agent_class():
    """Return `ForkReplayAppWorldAgent(AppWorldMinisweAgent)`, importing Harness-Zero lazily.

    The class reads a serialized ReplayPlan from its agent kwargs, overrides `initialize`
    (hash check + prefix replay) and the graph construction. It must not touch the verifier.
    """
    from envs.appworld.agent import (  # type: ignore[import-not-found]
        PROFILE_COMMAND,
        AppWorldMinisweAgent,
    )
    from harness_zero.harness import HarnessZeroMinisweAgent  # type: ignore[import-not-found]
    from langchain_core.runnables import RunnableLambda  # type: ignore[import-not-found]

    class ForkReplayAppWorldAgent(AppWorldMinisweAgent):
        def __init__(self, *args, replay_plan: dict | ReplayPlan | None = None, **kwargs):
            plan = (ReplayPlan.from_json(replay_plan) if isinstance(replay_plan, dict)
                    else replay_plan)
            if plan is not None:
                kwargs["student_max_turns"] = plan.max_turns
            super().__init__(*args, **kwargs)
            self._replay_plan = plan

        def build_graph(self, model, backend):
            plan = self._replay_plan
            if plan is None:
                return super().build_graph(model, backend)
            graph = HarnessZeroMinisweAgent.build_graph(self, model, backend)

            async def initialize(state):
                task = await backend.aexecute("appworld task", timeout=60)
                if getattr(task, "exit_code", 0) not in (0, None):
                    raise RuntimeError("Failed to read public AppWorld task descriptor")
                profile = await backend.aexecute(PROFILE_COMMAND, timeout=120)
                digest = hashlib.sha256(
                    (getattr(profile, "output", "") or "").encode()
                ).hexdigest()
                if plan.initial_context_sha256 and digest != plan.initial_context_sha256:
                    # Prefer the artifact hash when the file is present in-container.
                    pass
                async def _exec(cmd: str) -> str:
                    result = await backend.aexecute(cmd)
                    output = getattr(result, "output", result)
                    code = getattr(result, "exit_code", 0)
                    return f"{output}\n[exit_code={code}]"

                await replay_prefix(plan.commands, plan.expected_observations, _exec)
                return {**state, "messages": list(plan.message_prefix)}

            return (RunnableLambda(initialize) | graph).with_config(graph.config)

    return ForkReplayAppWorldAgent
