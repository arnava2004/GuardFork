"""Fork-point selection: forkable turns, first-guard fork, random-turn fork."""

from __future__ import annotations

import random
import re

from guardfork.guards.rules import Guard, GuardFiring, first_nonterminal
from guardfork.types import Trajectory

SUBAGENT_RE = re.compile(r"(?:^|[\s;&|(])agent\s+[\"']")


def _blocks_later(turn) -> bool:
    if turn.session_reset:
        return True
    cmd = turn.command or ""
    return bool(SUBAGENT_RE.search(cmd))


def forkable_turns(trajectory: Trajectory) -> list[int]:
    """Turns t in 1..T-1 whose pre-action state s_t can be replayed: none after a
    `session_reset=True` turn or an `agent "<task>"` subagent call. s_t itself (the state
    before such a turn executes) stays forkable."""
    out: list[int] = []
    blocked_after = None
    for turn in trajectory.turns:
        if blocked_after is not None and turn.turn > blocked_after:
            break
        if turn.turn < trajectory.T:
            out.append(turn.turn)
        if _blocks_later(turn):
            blocked_after = turn.turn
    return out


def guard_fork_turn(trajectory: Trajectory, firings: list[GuardFiring],
                    exclude: frozenset[Guard] = frozenset()) -> int | None:
    """GF-DPO: pre-action turn g of the first non-terminal firing, if forkable."""
    first = first_nonterminal(firings, trajectory.T, exclude)
    if first is None:
        return None
    return first.turn if first.turn in forkable_turns(trajectory) else None


def random_fork_turn(trajectory: Trajectory, rng: random.Random) -> int | None:
    """R-FORK-DPO: uniform over forkable turns t < T."""
    turns = forkable_turns(trajectory)
    if not turns:
        return None
    return rng.choice(turns)
