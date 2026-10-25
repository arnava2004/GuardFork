"""Weekend 1 pilot gates W1-a..e (thresholds in config/kill.toml [weekend_1_pilot])."""

from __future__ import annotations

from dataclasses import dataclass

from guardfork.config import KillConfig


@dataclass(frozen=True)
class PilotRun:
    trial_id: str
    scenario: str
    T: int  # noqa: N815
    passed: bool
    first_nonterminal_guard: int | None
    values: dict[int, tuple[int, int]]  # probed turn -> (passes, graded) at K continuations
    t_star: int | None
    t_star_retest: int | None = None


@dataclass(frozen=True)
class GateResult:
    name: str
    value: float | None
    threshold: float
    passed: bool | None
    note: str = ""


def w1a_replayable_fraction(determinism: float, kill: KillConfig) -> GateResult:
    raise NotImplementedError("W1: pilot.gates.w1a_replayable_fraction")


def w1b_nonterminal_guard_coverage(runs: list[PilotRun], kill: KillConfig) -> GateResult:
    """Failed pass-band runs with a non-terminal (g < T) G1–G5 firing."""
    raise NotImplementedError("W1: pilot.gates.w1b_nonterminal_guard_coverage")


def w1c_value_drop_vs_random(runs: list[PilotRun], kill: KillConfig) -> GateResult:
    """Mean (V(s_{g-1}) - V(s_{g+1})) minus matched-random drop; 80% CI lower bound from
    `guardfork.stats.bootstrap` (Arnav's), not computed here."""
    raise NotImplementedError("W1: pilot.gates.w1c_value_drop_vs_random")


def w1d_recoverable_guard_states(runs: list[PilotRun], kill: KillConfig) -> GateResult:
    """Fraction of guard states with V_hat(s_g) > 0."""
    raise NotImplementedError("W1: pilot.gates.w1d_recoverable_guard_states")


def w1e_oracle_test_retest(runs: list[PilotRun], kill: KillConfig) -> GateResult:
    """Agreement of t* and t*_retest within ±1 turn, over runs with both defined."""
    raise NotImplementedError("W1: pilot.gates.w1e_oracle_test_retest")


def guard_false_positive_rates(passed_runs: list[PilotRun]) -> dict[str, float]:
    """Per-guard firing rate on PASSED runs; guards above 0.30 are excluded from defining
    process failures."""
    raise NotImplementedError("W1: pilot.gates.guard_false_positive_rates")
