"""Q1 study assembly: eligibility, per-run rows, and the report table."""

from __future__ import annotations

from dataclasses import dataclass

from guardfork.guards.rules import evaluate, first_nonterminal
from guardfork.splits import scenario_of
from guardfork.stats.bootstrap import analytic_random_hit
from guardfork.telemetry.to_trajectory import to_trajectory
from guardfork.telemetry.trials import TrialRecord


@dataclass(frozen=True)
class Q1Row:
    trial_id: str
    scenario: str
    T: int  # noqa: N815
    g: int | None
    g_terminal: bool
    t_star: int | None
    t_star_drop: int | None


def eligible_failed_runs(records: list[TrialRecord], task_pass: dict[str, tuple[int, int]],
                         band: tuple[float, float] = (0.3, 0.8), min_samples: int = 8
                         ) -> list[TrialRecord]:
    """Failed runs on tasks with p_hat in `band` over >= `min_samples` base rollouts."""
    lo, hi = band
    eligible_tasks = {
        tid for tid, (p, n) in task_pass.items()
        if n >= min_samples and lo <= p / n <= hi
    }
    return [r for r in records if r.task_id in eligible_tasks and r.reward == 0.0]


def build_rows(records: list[TrialRecord], oracle_results: dict[str, dict]) -> list[Q1Row]:
    rows = []
    for rec in records:
        traj = to_trajectory(rec)
        trial_id = f"{rec.job}/{rec.trial_dir.name}"
        firings = evaluate(traj)
        first = first_nonterminal(firings, traj.T)
        terminal_only = first is None and any(f.turn == traj.T for f in firings)
        oracle = oracle_results.get(trial_id) or oracle_results.get(rec.trial_dir.name) or {}
        rows.append(Q1Row(
            trial_id=trial_id,
            scenario=scenario_of(rec.task_id) if rec.task_id else "",
            T=traj.T,
            g=first.turn if first else None,
            g_terminal=terminal_only,
            t_star=oracle.get("t_star"),
            t_star_drop=oracle.get("t_star_drop"),
        ))
    return rows


def report_table(rows: list[Q1Row], windows: tuple[int, ...] = (0, 1, 2, 3)) -> str:
    """Markdown: non-terminal headline, terminal column, undefined-t* rate, Hit(w) per w."""
    n = len(rows)
    undefined = sum(r.t_star is None for r in rows)
    headline = [r for r in rows if r.g is not None and r.t_star is not None]
    terminal = sum(r.g_terminal for r in rows)
    lines = [
        f"runs: {n}; comparable (non-terminal g and defined t*): {len(headline)}; "
        f"t* undefined: {undefined}/{n}; terminal-only firings: {terminal}",
        "",
        "| window | hits | Hit(w) | mean random |",
        "|---|---|---|---|",
    ]
    for w in windows:
        hits = sum(abs(r.g - r.t_star) <= w for r in headline)
        rate = hits / len(headline) if headline else 0.0
        rand = (
            sum(analytic_random_hit(r.T, r.t_star, w) for r in headline) / len(headline)
            if headline else 0.0
        )
        lines.append(f"| w={w} | {hits}/{len(headline)} | {rate:.3f} | {rand:.3f} |")
    return "\n".join(lines) + "\n"
