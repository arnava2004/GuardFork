"""Replay-from-zero forking as Harbor trials (Weekends 1–2). Agent-written; stubs until W0 passes.

A fork at turn t re-creates the pre-action state s_t in a fresh AppWorld container:

1. `initialize` runs the normal `appworld task` + `PROFILE_COMMAND`, then checks the sha256 of
   the produced `appworld_initial_context.json` against the source trial.
2. Commands 1..t-1 are replayed via `backend.aexecute`, asserting each observation equals the
   recorded one (`ReplayDiverged` otherwise).
3. The logged message prefix up to turn t is seeded and the inner
   `create_supervised_miniswe` graph runs with `max_turns = 40 - (t - 1)`.
4. Harbor stops `main`, runs `snapshot.py` on `world`, and grades with the official verifier.

Each of the K continuations is its own Harbor trial (`docker compose -p fork_<uuid>`,
`PYTHONHASHSEED=0`, `down -v` in `finally`). Harness-Zero is imported lazily so this module
loads without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guardfork.telemetry.trials import TrialRecord


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


def initial_context_sha256(trial_dir: Path) -> str:
    """sha256 of the source trial's `appworld_initial_context.json`."""
    raise NotImplementedError("W1: harbor_replay.initial_context_sha256")


def build_replay_plan(record: TrialRecord, fork_turn: int) -> ReplayPlan:
    """Commands/observations 1..t-1 and the message prefix needed to reach s_t.

    Raises NotForkable for turns rejected by `fork.select.forkable_turns`.
    """
    raise NotImplementedError("W1: harbor_replay.build_replay_plan")


def make_fork_agent_class():
    """Return `ForkReplayAppWorldAgent(AppWorldMinisweAgent)`, importing Harness-Zero lazily.

    The class reads a serialized ReplayPlan from its agent kwargs, overrides `initialize`
    (steps 1–2 above) and the graph construction (step 3). It must not touch the verifier.
    """
    raise NotImplementedError("W1: harbor_replay.make_fork_agent_class")
