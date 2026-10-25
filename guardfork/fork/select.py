"""Fork-point selection. HAND-WRITTEN BY ARNAV (AGENTS.md rule 1)."""

from __future__ import annotations

import random

from guardfork.guards.rules import Guard, GuardFiring
from guardfork.types import Trajectory


def forkable_turns(trajectory: Trajectory) -> list[int]:
    """Turns t in 1..T-1 whose pre-action state s_t can be replayed: none after a
    `session_reset=True` turn or an `agent "<task>"` subagent call. s_t itself (the state
    before such a turn executes) stays forkable."""
    raise NotImplementedError("Arnav: fork/select.py")


def guard_fork_turn(trajectory: Trajectory, firings: list[GuardFiring],
                    exclude: frozenset[Guard] = frozenset()) -> int | None:
    """GF-DPO: pre-action turn g of the first non-terminal firing, if forkable."""
    raise NotImplementedError("Arnav: fork/select.py")


def random_fork_turn(trajectory: Trajectory, rng: random.Random) -> int | None:
    """R-FORK-DPO: uniform over forkable turns t < T."""
    raise NotImplementedError("Arnav: fork/select.py")
