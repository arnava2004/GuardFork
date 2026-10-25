"""Weekend 1 pilot gates W1-a..e (thresholds in config/kill.toml [weekend_1_pilot])."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from guardfork.config import KillConfig
from guardfork.stats.bootstrap import clustered_bootstrap_ci


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
    fired_guards: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateResult:
    name: str
    value: float | None
    threshold: float
    passed: bool | None
    note: str = ""


def _hat(pair: tuple[int, int] | None) -> float | None:
    if pair is None or pair[1] <= 0:
        return None
    return pair[0] / pair[1]


def w1a_replayable_fraction(determinism: float, kill: KillConfig) -> GateResult:
    thr = kill.weekend_1_pilot["W1a_min_replayable_fraction"]
    return GateResult("W1a replayable fraction", determinism, thr, determinism >= thr)


def w1b_nonterminal_guard_coverage(runs: list[PilotRun], kill: KillConfig) -> GateResult:
    """Failed pass-band runs with a non-terminal (g < T) G1–G5 firing."""
    failed = [r for r in runs if not r.passed]
    thr = kill.weekend_1_pilot["W1b_min_nonterminal_guard_coverage"]
    if not failed:
        return GateResult("W1b nonterminal guard coverage", None, thr, None, "no failed runs")
    frac = sum(r.first_nonterminal_guard is not None for r in failed) / len(failed)
    return GateResult("W1b nonterminal guard coverage", frac, thr, frac >= thr,
                      f"{len(failed)} failed runs")


def w1c_value_drop_vs_random(runs: list[PilotRun], kill: KillConfig) -> GateResult:
    """Mean (V(s_{g-1}) - V(s_{g+1})) minus matched-random drop; 80% CI lower bound."""
    thr = kill.weekend_1_pilot["W1c_min_value_drop_ci_lower_bound"]
    deltas: list[float] = []
    clusters: list[str] = []
    for run in runs:
        g = run.first_nonterminal_guard
        if g is None:
            continue
        v_before = _hat(run.values.get(g - 1))
        v_after = _hat(run.values.get(g + 1))
        if v_before is None or v_after is None:
            continue
        random_drops = []
        for t, pair in run.values.items():
            other = run.values.get(t + 2)
            if other is None or t in {g - 1, g, g + 1}:
                continue
            a, b = _hat(pair), _hat(other)
            if a is not None and b is not None:
                random_drops.append(a - b)
        random_drop = (sum(random_drops) / len(random_drops)) if random_drops else 0.0
        deltas.append((v_before - v_after) - random_drop)
        clusters.append(run.scenario)
    if not deltas:
        return GateResult("W1c value drop vs random", None, thr, None, "no comparable windows")
    _mean, lo, _hi = clustered_bootstrap_ci(deltas, clusters, n_boot=2000, alpha=0.20, seed=0)
    return GateResult("W1c value drop vs random", lo, thr, lo > thr, "80% CI lower bound")


def w1d_recoverable_guard_states(runs: list[PilotRun], kill: KillConfig) -> GateResult:
    """Fraction of guard states with V_hat(s_g) > 0."""
    thr = kill.weekend_1_pilot["W1d_min_recoverable_guard_states"]
    usable = []
    for run in runs:
        g = run.first_nonterminal_guard
        if g is None:
            continue
        hat = _hat(run.values.get(g))
        if hat is None:
            continue
        usable.append(hat > 0)
    if not usable:
        return GateResult("W1d recoverable guard states", None, thr, None)
    frac = sum(usable) / len(usable)
    return GateResult("W1d recoverable guard states", frac, thr, frac >= thr)


def w1e_oracle_test_retest(runs: list[PilotRun], kill: KillConfig) -> GateResult:
    """Agreement of t* and t*_retest within ±1 turn, over runs with both defined."""
    thr = kill.weekend_1_pilot["W1e_min_oracle_test_retest_pm1"]
    both = [r for r in runs if r.t_star is not None and r.t_star_retest is not None]
    if not both:
        return GateResult("W1e oracle test-retest ±1", None, thr, None, "no paired t*")
    frac = sum(abs(r.t_star - r.t_star_retest) <= 1 for r in both) / len(both)
    return GateResult("W1e oracle test-retest ±1", frac, thr, frac >= thr, f"n={len(both)}")


def guard_false_positive_rates(passed_runs: list[PilotRun]) -> dict[str, float]:
    """Per-guard firing rate on PASSED runs; guards above 0.30 are excluded from defining
    process failures."""
    if not passed_runs:
        return {}
    counts: Counter[str] = Counter()
    for run in passed_runs:
        counts.update(set(run.fired_guards))
    n = len(passed_runs)
    return {g: counts[g] / n for g in sorted(counts)}


def evaluate_pilot(runs: list[PilotRun], determinism: float, kill: KillConfig
                   ) -> list[GateResult]:
    return [
        w1a_replayable_fraction(determinism, kill),
        w1b_nonterminal_guard_coverage(runs, kill),
        w1c_value_drop_vs_random(runs, kill),
        w1d_recoverable_guard_states(runs, kill),
        w1e_oracle_test_retest(runs, kill),
    ]
