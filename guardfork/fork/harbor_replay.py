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
    full_replay: bool = False

    @property
    def max_turns(self) -> int:
        return max(1, 40 - (self.fork_turn - 1))

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
            full_replay=bool(data.get("full_replay", False)),
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


def _prefix_actions(record: TrialRecord, n_events: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    commands, observations = [], []
    for i in range(n_events):
        cmd = _event_command(record.events[i])
        if not cmd:
            continue
        commands.append(cmd)
        observations.append(observation_after(record.events, i) or "")
    return tuple(commands), tuple(observations)


def build_replay_plan(record: TrialRecord, fork_turn: int) -> ReplayPlan:
    """Commands/observations 1..t-1 and the message prefix needed to reach s_t.

    The prefix is the student-visible conversation before turn t (system prompt excluded;
    `create_agent` adds it). Raises NotForkable for turns rejected by `forkable_turns`.
    """
    traj = to_trajectory(record)
    allowed = forkable_turns(traj)
    if fork_turn not in allowed:
        raise NotForkable(f"turn {fork_turn} is not forkable (allowed {allowed})")
    commands, observations = _prefix_actions(record, fork_turn - 1)
    last = record.events[fork_turn - 1] if record.events else {}
    return ReplayPlan(
        source_trial=record.trial_dir,
        task_id=record.task_id or traj.task_id,
        fork_turn=fork_turn,
        initial_context_sha256=initial_context_sha256(record.trial_dir),
        commands=commands,
        expected_observations=observations,
        message_prefix=tuple(last.get("context") or []),
    )


def full_replay_plan(record: TrialRecord) -> ReplayPlan:
    """Replay every recorded command (t = T + 1) with no model calls, for reward parity."""
    commands, observations = _prefix_actions(record, len(record.events))
    return ReplayPlan(
        source_trial=record.trial_dir,
        task_id=record.task_id or "",
        fork_turn=len(record.events) + 1,
        initial_context_sha256=initial_context_sha256(record.trial_dir),
        commands=commands,
        expected_observations=observations,
        message_prefix=(),
        full_replay=True,
    )


def save_plan(plan: ReplayPlan, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_json(), ensure_ascii=False), encoding="utf-8")
    return path


def load_plan(path: Path) -> ReplayPlan:
    return ReplayPlan.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


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


def deserialize_messages(messages: tuple[dict[str, Any], ...] | list[dict[str, Any]]):
    """Harness-Zero `serialize_message` dicts -> LangChain message objects."""
    from langchain_core.messages import messages_from_dict  # type: ignore[import-not-found]

    return messages_from_dict([{"type": m["type"], "data": m} for m in messages])


def make_fork_agent_class():
    """Return `ForkReplayAppWorldAgent(AppWorldMinisweAgent)`, importing Harness-Zero lazily.

    Pass `replay_plan_path` as an agent kwarg. The agent runs AppWorld's own initialize
    (task check, profile command, initial-context artifact), requires the artifact sha256 to
    match the source trial, replays commands 1..t-1 with exact observation equality, then
    seeds the logged prefix and runs the inner graph with `max_turns = 40 - (t - 1)`.
    With `full_replay`, it stops after replaying every command (no model calls). Grading is
    left to Harbor's verifier untouched.
    """
    from envs.appworld.agent import AppWorldMinisweAgent  # type: ignore[import-not-found]
    from harness_zero.student import OBSERVATION_TEMPLATE  # type: ignore[import-not-found]
    from langchain_core.runnables import RunnableLambda  # type: ignore[import-not-found]

    class ForkReplayAppWorldAgent(AppWorldMinisweAgent):
        def __init__(self, *args, replay_plan_path: str | None = None, **kwargs):
            plan = load_plan(Path(replay_plan_path)) if replay_plan_path else None
            if plan is not None:
                kwargs["student_max_turns"] = plan.max_turns
            super().__init__(*args, **kwargs)
            self._replay_plan = plan

        def build_graph(self, model, backend):
            plan = self._replay_plan
            wrapped = AppWorldMinisweAgent.build_graph(self, model, backend)
            if plan is None:
                return wrapped
            sequence = getattr(wrapped, "bound", wrapped)
            appworld_initialize, inner_graph = sequence.first, sequence.last

            async def execute(cmd: str) -> str:
                result = await backend.aexecute(cmd)
                return OBSERVATION_TEMPLATE.format(output=result.output,
                                                   exit_code=result.exit_code)

            async def initialize(state):
                state = await appworld_initialize.ainvoke(state)
                artifact_path = self.logs_dir / "appworld_initial_context.json"
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                if (plan.initial_context_sha256
                        and artifact.get("sha256") != plan.initial_context_sha256):
                    raise ReplayDiverged(0, plan.initial_context_sha256,
                                         str(artifact.get("sha256")))
                await replay_prefix(plan.commands, plan.expected_observations, execute)
                if plan.full_replay:
                    return state
                return {**state, "messages": deserialize_messages(plan.message_prefix)}

            if plan.full_replay:
                return RunnableLambda(initialize).with_config(inner_graph.config)
            return (RunnableLambda(initialize) | inner_graph).with_config(inner_graph.config)

    return ForkReplayAppWorldAgent
