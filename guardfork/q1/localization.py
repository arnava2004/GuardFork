"""Q1 study assembly: eligibility, per-run rows, and the report table.

Rows carry several guard localizers so the report can show results per guard type,
with G3 excluded, and with a stricter "unrecovered G3" (an API-error firing counts only if
the next RECOVERY_WINDOW turns do not read API docs or log in), plus the zero-rollout
LLM-judge baseline (`judge`, any turn 1..T, from judge.json).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from guardfork.freeze import DevSet
from guardfork.guards.rules import GUARDS_VERSION, Guard, GuardFiring, evaluate
from guardfork.splits import scenario_of
from guardfork.stats.bootstrap import analytic_random_hit
from guardfork.telemetry.to_trajectory import to_trajectory
from guardfork.telemetry.trials import TrialRecord
from guardfork.types import Trajectory

RECOVERY_WINDOW = 2
RECOVERY_RE = re.compile(r"apis\.api_docs\.|appworld\s+docs|apis\.\w+\.login\s*\(")
LOCALIZERS = ("all", "no_g3", "g3_unrecovered", *(g.value for g in Guard), "judge")


@dataclass(frozen=True)
class Q1Row:
    trial_id: str
    scenario: str
    T: int  # noqa: N815
    g: int | None
    g_terminal: bool
    t_star: int | None
    t_star_drop: int | None
    localizers: dict[str, int | None] = field(default_factory=dict)
    guards_version: str = GUARDS_VERSION


def eligible_failed_runs(records: list[TrialRecord], task_pass: dict[str, tuple[int, int]],
                         band: tuple[float, float] = (0.3, 0.8), min_samples: int = 8,
                         dev_set: DevSet | None = None) -> list[TrialRecord]:
    """Failed runs on tasks with p_hat in `band` over >= `min_samples` base rollouts,
    excluding anything in the dev set the guards were tuned on."""
    lo, hi = band
    eligible_tasks = {
        tid for tid, (p, n) in task_pass.items()
        if n >= min_samples and lo <= p / n <= hi
    }
    return [
        r for r in records
        if r.task_id in eligible_tasks and r.reward == 0.0
        and not (dev_set and dev_set.excludes(r.job, r.task_id))
    ]


def g3_recovered(traj: Trajectory, turn: int, window: int = RECOVERY_WINDOW) -> bool:
    later = [t for t in traj.turns if turn < t.turn <= turn + window]
    return any(RECOVERY_RE.search(t.command or "") for t in later)


def _first(firings: list[GuardFiring], T: int, keep) -> int | None:  # noqa: N803
    turns = [f.turn for f in firings if f.turn < T and keep(f)]
    return min(turns) if turns else None


def localizer_turns(traj: Trajectory, firings: list[GuardFiring]) -> dict[str, int | None]:
    T = traj.T  # noqa: N806
    out: dict[str, int | None] = {
        "all": _first(firings, T, lambda f: True),
        "no_g3": _first(firings, T, lambda f: f.guard is not Guard.G3_FAILED_PRECONDITION),
        "g3_unrecovered": _first(
            firings, T,
            lambda f: f.guard is not Guard.G3_FAILED_PRECONDITION
            or not g3_recovered(traj, f.turn),
        ),
    }
    for g in Guard:
        out[g.value] = _first(firings, T, lambda f, g=g: f.guard is g)
    return out


def _judge_turn(rec: TrialRecord, trial_id: str, judge_results: dict[str, dict] | None,
                judge_root: Path | None) -> int | None:
    entry = None
    if judge_results is not None:
        entry = judge_results.get(trial_id) or judge_results.get(rec.trial_dir.name)
    if entry is None and judge_root is not None:
        path = judge_root / rec.job / rec.trial_dir.name / "judge.json"
        if path.is_file():
            entry = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(entry, dict):
        turn = entry.get("turn")
        return turn if isinstance(turn, int) and not isinstance(turn, bool) else None
    return entry if isinstance(entry, int) and not isinstance(entry, bool) else None


def build_rows(records: list[TrialRecord], oracle_results: dict[str, dict],
               judge_results: dict[str, dict] | None = None,
               judge_root: Path | None = None) -> list[Q1Row]:
    """`judge_results` maps trial_id (or trial dir name) to judge.json contents (or a turn);
    otherwise `judge_root/<job>/<trial>/judge.json` is read when given."""
    rows = []
    for rec in records:
        traj = to_trajectory(rec)
        trial_id = f"{rec.job}/{rec.trial_dir.name}"
        firings = evaluate(traj)
        loc = localizer_turns(traj, firings)
        loc["judge"] = _judge_turn(rec, trial_id, judge_results, judge_root)
        terminal_only = loc["all"] is None and any(f.turn == traj.T for f in firings)
        oracle = oracle_results.get(trial_id) or oracle_results.get(rec.trial_dir.name) or {}
        rows.append(Q1Row(
            trial_id=trial_id,
            scenario=scenario_of(rec.task_id) if rec.task_id else "",
            T=traj.T,
            g=loc["all"],
            g_terminal=terminal_only,
            t_star=oracle.get("t_star"),
            t_star_drop=oracle.get("t_star_drop"),
            localizers=loc,
        ))
    return rows


def _hit_rows(rows: list[Q1Row], localizer: str, windows: tuple[int, ...]) -> list[str]:
    comparable = [(r.localizers.get(localizer) if localizer != "all" else r.g, r)
                  for r in rows if r.t_star is not None]
    comparable = [(g, r) for g, r in comparable if g is not None]
    n = len(comparable)
    cells = []
    for w in windows:
        hits = sum(abs(g - r.t_star) <= w for g, r in comparable)
        rand = sum(analytic_random_hit(r.T, r.t_star, w) for _, r in comparable) / n if n else 0
        cells.append(f"{hits}/{n} ({hits / n:.2f} vs {rand:.2f})" if n else "–")
    return [f"| {localizer} | {n} | " + " | ".join(cells) + " |"]


def report_table(rows: list[Q1Row], windows: tuple[int, ...] = (0, 1, 2, 3)) -> str:
    """Markdown: non-terminal headline, terminal column, undefined-t* rate, Hit(w) per w,
    then the same per localizer (all, no_g3, g3_unrecovered, G1..G5, judge)."""
    n = len(rows)
    undefined = sum(r.t_star is None for r in rows)
    headline = [r for r in rows if r.g is not None and r.t_star is not None]
    terminal = sum(r.g_terminal for r in rows)
    versions = sorted({r.guards_version for r in rows}) or [GUARDS_VERSION]
    lines = [
        f"guards: {', '.join(versions)}; runs: {n}; comparable (non-terminal g and defined t*): "
        f"{len(headline)}; t* undefined: {undefined}/{n}; terminal-only firings: {terminal}",
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
    lines += [
        "",
        "Per localizer: hits/n (Hit vs mean analytic random)",
        "",
        "| localizer | n | " + " | ".join(f"w={w}" for w in windows) + " |",
        "|---|---|" + "---|" * len(windows),
    ]
    for loc in LOCALIZERS:
        lines += _hit_rows(rows, loc, windows)
    return "\n".join(lines) + "\n"
