"""Offline harness guards G1–G5. HAND-WRITTEN BY ARNAV (AGENTS.md rule 1).

Guards see only the trajectory prefix up to and including turn t (no lookahead) and
never fire on demo turns (assistant_index < 10). MAX_TURNS / CONTEXT_OVERFLOW /
REPLAY_DIVERGED are TrajectoryStatus values, not guards.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from guardfork.types import Trajectory, Turn


class Guard(StrEnum):
    G1_INVALID_ACTION = "G1"
    G2_LOOP = "G2"
    G3_FAILED_PRECONDITION = "G3"
    G4_CONSTRAINT_VIOLATION = "G4"
    G5_UNVERIFIED_COMPLETION = "G5"


@dataclass(frozen=True)
class GuardFiring:
    guard: Guard
    turn: int
    evidence: str


def check_turn(prefix: tuple[Turn, ...], trajectory: Trajectory,
               *, scan_reasoning_for_code: bool = False) -> list[GuardFiring]:
    """Guards firing on the last turn of `prefix`, using only `prefix`.

    G1 is evaluated first; G5 only if G1 did not fire. G1 scans visible `content` for
    code fences / `appworld exec`, and `reasoning` only for structural tool markup
    (`<tool_call>`, `<function=`) unless `scan_reasoning_for_code` (the sensitivity split).
    """
    raise NotImplementedError("Arnav: guards/rules.py")


def evaluate(trajectory: Trajectory, *, scan_reasoning_for_code: bool = False
             ) -> list[GuardFiring]:
    """All firings over the trajectory, ordered by turn."""
    raise NotImplementedError("Arnav: guards/rules.py")


def first_nonterminal(firings: list[GuardFiring], T: int,  # noqa: N803
                      exclude: frozenset[Guard] = frozenset()) -> GuardFiring | None:
    """First firing with turn g < T whose guard is not in `exclude` (e.g. held-out G3)."""
    raise NotImplementedError("Arnav: guards/rules.py")
