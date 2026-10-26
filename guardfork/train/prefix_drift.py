"""Prefix drift: primary metric for the Section 5 credit pilot (zero container cost).

For a held-out oracle-labelled failed trajectory and two policies (base pi_before, fine-tuned
pi_after), the per-turn change in action log-probability is

    delta_t = log pi_after(a_t | s_t) - log pi_before(a_t | s_t).

Turns are grouped by the oracle's change-point interval [t*_lo, t*_hi]:
pre (t < t*_lo), at (t*_lo <= t <= t*_hi), post (t > t*_hi). A credit rule that stops
punishing good pre-mistake turns shows a positive prefix-protection gap
mean(delta_pre) - mean(delta_at). Inference is a paired, scenario-clustered bootstrap.

Log-probs are computed elsewhere (forward passes over saved token ids); this module only
scores them, so it is stdlib-only and testable offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean

from guardfork.stats.bootstrap import clustered_bootstrap_ci

Group = str  # "pre" | "at" | "post"


@dataclass(frozen=True)
class TurnLogProbs:
    turn: int
    logp_before: float
    logp_after: float
    n_tokens: int = 1

    @property
    def delta(self) -> float:
        return self.logp_after - self.logp_before

    @property
    def delta_per_token(self) -> float:
        return self.delta / max(self.n_tokens, 1)


@dataclass(frozen=True)
class DriftRun:
    trial_id: str
    scenario: str
    t_star_lo: int
    t_star_hi: int
    turns: Sequence[TurnLogProbs]


@dataclass
class DriftSummary:
    n_runs: int
    n_turns: dict[Group, int]
    mean_delta: dict[Group, float | None]
    gap_pre_minus_at: float | None
    gap_ci: tuple[float, float] | None
    gap_pre_minus_post: float | None
    per_run: list[dict] = field(default_factory=list)


def group_of(turn: int, lo: int, hi: int) -> Group:
    if turn < lo:
        return "pre"
    if turn > hi:
        return "post"
    return "at"


def run_group_means(run: DriftRun, *, per_token: bool = False) -> dict[Group, float | None]:
    buckets: dict[Group, list[float]] = {"pre": [], "at": [], "post": []}
    for t in run.turns:
        d = t.delta_per_token if per_token else t.delta
        buckets[group_of(t.turn, run.t_star_lo, run.t_star_hi)].append(d)
    return {g: (fmean(v) if v else None) for g, v in buckets.items()}


def summarize(runs: Sequence[DriftRun], *, per_token: bool = False, n_boot: int = 5000,
              alpha: float = 0.05, seed: int = 0) -> DriftSummary:
    """Paired gap per run (mean pre minus mean at), clustered by scenario."""
    n_turns = {"pre": 0, "at": 0, "post": 0}
    pooled: dict[Group, list[float]] = {"pre": [], "at": [], "post": []}
    gaps, gaps_post, clusters, per_run = [], [], [], []
    for run in runs:
        means = run_group_means(run, per_token=per_token)
        for t in run.turns:
            g = group_of(t.turn, run.t_star_lo, run.t_star_hi)
            n_turns[g] += 1
            pooled[g].append(t.delta_per_token if per_token else t.delta)
        row = {"trial_id": run.trial_id, "scenario": run.scenario, **means}
        if means["pre"] is not None and means["at"] is not None:
            row["gap_pre_minus_at"] = means["pre"] - means["at"]
            gaps.append(row["gap_pre_minus_at"])
            clusters.append(run.scenario)
        if means["pre"] is not None and means["post"] is not None:
            gaps_post.append(means["pre"] - means["post"])
        per_run.append(row)
    gap = gap_ci = None
    if gaps:
        mean, lo, hi = clustered_bootstrap_ci(gaps, clusters, n_boot=n_boot, alpha=alpha,
                                              seed=seed)
        gap, gap_ci = mean, (lo, hi)
    return DriftSummary(
        n_runs=len(runs),
        n_turns=n_turns,
        mean_delta={g: (fmean(v) if v else None) for g, v in pooled.items()},
        gap_pre_minus_at=gap,
        gap_ci=gap_ci,
        gap_pre_minus_post=fmean(gaps_post) if gaps_post else None,
        per_run=per_run,
    )


def compare_checkpoints(summaries: dict[str, DriftSummary]) -> str:
    """Markdown table: one row per fine-tuned checkpoint (uniform, hard-mask, soft-posterior,
    GF-DPO, ...), the prefix-protection gap with its CI, and the group means."""
    lines = [
        "| checkpoint | runs | mean d_pre | mean d_at | mean d_post | gap pre-at [95% CI] |",
        "|---|---|---|---|---|---|",
    ]

    def f(x: float | None) -> str:
        return "n/a" if x is None else f"{x:+.3f}"

    for name, s in summaries.items():
        ci = ("n/a" if s.gap_ci is None
              else f"{f(s.gap_pre_minus_at)} [{f(s.gap_ci[0])}, {f(s.gap_ci[1])}]")
        lines.append(f"| {name} | {s.n_runs} | {f(s.mean_delta['pre'])} | {f(s.mean_delta['at'])} "
                     f"| {f(s.mean_delta['post'])} | {ci} |")
    return "\n".join(lines) + "\n"
