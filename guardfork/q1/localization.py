"""Q1 study assembly: eligibility, per-run rows, and the report table.

Rows carry several guard localizers so the report can show results per guard type,
with G3 excluded, and with a stricter "unrecovered G3" (an API-error firing counts only if
the next RECOVERY_WINDOW turns do not read API docs or log in), plus the zero-rollout
LLM-judge rows (`judge_open`, `judge_local`, `judge_claude`; any turn 1..T, argmax of the
distribution in judge_<id>.json; `judge_claude` is eval-only). The legacy single slot
`judge` (judge.json) is still accepted and reported when present.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

from guardfork.freeze import DevSet
from guardfork.guards.rules import GUARDS_VERSION, Guard, GuardFiring, evaluate
from guardfork.judge import JUDGE_SLOTS, LEGACY_JUDGE_ID, judge_filename
from guardfork.splits import scenario_of
from guardfork.stats.bootstrap import analytic_random_hit
from guardfork.telemetry.to_trajectory import to_trajectory
from guardfork.telemetry.trials import TrialRecord
from guardfork.types import Trajectory

RECOVERY_WINDOW = 2
RECOVERY_RE = re.compile(r"apis\.api_docs\.|appworld\s+docs|apis\.\w+\.login\s*\(")
LOCALIZERS = ("all", "no_g3", "g3_unrecovered", *(g.value for g in Guard), *JUDGE_SLOTS)
CALIBRATION_BINS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0))
AGREEMENT_WINDOW = 1


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
    judge_dists: dict[str, list[float]] = field(default_factory=dict)


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


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _judge_entry(rec: TrialRecord, trial_id: str, results: dict[str, Any] | None,
                 judge_root: Path | None, slot: str) -> Any:
    entry = None
    if results is not None:
        entry = results.get(trial_id)
        if entry is None:
            entry = results.get(rec.trial_dir.name)
    if entry is None and judge_root is not None:
        path = judge_root / rec.job / rec.trial_dir.name / judge_filename(slot)
        if path.is_file():
            entry = json.loads(path.read_text(encoding="utf-8"))
    return entry


def _judge_turn(entry: Any) -> int | None:
    if isinstance(entry, dict):
        turn = entry.get("turn")
        return turn if _is_int(turn) else None
    return entry if _is_int(entry) else None


def _judge_dist(entry: Any, T: int) -> list[float] | None:  # noqa: N803
    if not isinstance(entry, dict):
        return None
    dist = entry.get("distribution")
    if (isinstance(dist, list) and len(dist) == T
            and all(isinstance(p, int | float) and not isinstance(p, bool) for p in dist)):
        return [float(p) for p in dist]
    return None


def build_rows(records: list[TrialRecord], oracle_results: dict[str, dict],
               judge_results: dict[str, Any] | None = None,
               judge_root: Path | None = None,
               judge_results_by_slot: dict[str, dict[str, Any]] | None = None,
               ) -> list[Q1Row]:
    """Judge inputs, per slot in `JUDGE_SLOTS` (+ the legacy `judge` slot):
    `judge_results_by_slot[slot]` maps trial_id (or trial dir name) to the judge record (or
    a bare turn); `judge_results` is the legacy single-slot form (slot `judge`); otherwise
    `judge_root/<job>/<trial>/judge_<slot>.json` (legacy `judge.json`) is read when present.
    The legacy `judge` localizer is only set when such data exists."""
    by_slot: dict[str, dict[str, Any] | None] = {s: None for s in JUDGE_SLOTS}
    by_slot[LEGACY_JUDGE_ID] = judge_results
    for slot, results in (judge_results_by_slot or {}).items():
        by_slot[slot] = results
    rows = []
    for rec in records:
        traj = to_trajectory(rec)
        trial_id = f"{rec.job}/{rec.trial_dir.name}"
        firings = evaluate(traj)
        loc = localizer_turns(traj, firings)
        dists: dict[str, list[float]] = {}
        for slot, results in by_slot.items():
            entry = _judge_entry(rec, trial_id, results, judge_root, slot)
            if slot == LEGACY_JUDGE_ID and entry is None:
                continue
            loc[slot] = _judge_turn(entry)
            dist = _judge_dist(entry, traj.T)
            if dist is not None:
                dists[slot] = dist
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
            judge_dists=dists,
        ))
    return rows


def judge_slots_present(rows: list[Q1Row]) -> list[str]:
    """`JUDGE_SLOTS` plus the legacy `judge` slot when any row carries it."""
    slots = list(JUDGE_SLOTS)
    if any(LEGACY_JUDGE_ID in r.localizers for r in rows):
        slots.insert(0, LEGACY_JUDGE_ID)
    return slots


def judge_agreement(rows: list[Q1Row], window: int = AGREEMENT_WINDOW,
                    ) -> dict[tuple[str, str], tuple[int, int]]:
    """`(slot_a, slot_b) -> (agree, n)` over rows where both slots picked a turn;
    agree = `|a - b| <= window`."""
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for a, b in combinations(judge_slots_present(rows), 2):
        pairs = [(r.localizers.get(a), r.localizers.get(b)) for r in rows]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        out[(a, b)] = (sum(abs(x - y) <= window for x, y in pairs), len(pairs))
    return out


def pick_position(pick: int, T: int) -> str:  # noqa: N803
    """`first` / `middle` / `last` third of 1..T (`3 * pick <= T`, `<= 2T`, else last)."""
    if 3 * pick <= T:
        return "first"
    if 3 * pick <= 2 * T:
        return "middle"
    return "last"


def pick_histogram(rows: list[Q1Row]) -> dict[str, dict[str, int]]:
    """Per judge slot: counts of picks in the first/middle/last third and exactly at T
    (`exact_last`, also counted in `last`), plus `n`."""
    out: dict[str, dict[str, int]] = {}
    for slot in judge_slots_present(rows):
        hist = {"n": 0, "first": 0, "middle": 0, "last": 0, "exact_last": 0}
        for r in rows:
            pick = r.localizers.get(slot)
            if pick is None:
                continue
            hist["n"] += 1
            hist[pick_position(pick, r.T)] += 1
            hist["exact_last"] += int(pick == r.T)
        out[slot] = hist
    return out


def calibration_bin(p: float) -> int | None:
    for i, (lo, hi) in enumerate(CALIBRATION_BINS):
        last = i == len(CALIBRATION_BINS) - 1
        if lo <= p < hi or (last and p == hi):
            return i
    return None


def judge_calibration(rows: list[Q1Row], window: int = 1,
                      ) -> dict[str, list[tuple[tuple[float, float], int, int]]]:
    """Per judge slot, per `CALIBRATION_BINS` bin of the top probability: `(bin, hits, n)`
    with hit = `|argmax - t*| <= window` over rows with a distribution and a defined t*."""
    out: dict[str, list[tuple[tuple[float, float], int, int]]] = {}
    for slot in judge_slots_present(rows):
        hits = [0] * len(CALIBRATION_BINS)
        ns = [0] * len(CALIBRATION_BINS)
        for r in rows:
            dist = r.judge_dists.get(slot)
            if not dist or r.t_star is None:
                continue
            top = min(range(len(dist)), key=lambda i, d=dist: (-d[i], i))
            b = calibration_bin(dist[top])
            if b is None:
                continue
            ns[b] += 1
            hits[b] += int(abs((top + 1) - r.t_star) <= window)
        out[slot] = [(CALIBRATION_BINS[i], hits[i], ns[i]) for i in range(len(CALIBRATION_BINS))]
    return out


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


def _ratio(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den:.2f})" if den else "–"


def judge_sections(rows: list[Q1Row], windows: tuple[int, ...] = (0, 1, 2, 3)) -> list[str]:
    """Markdown blocks: pairwise judge agreement (±AGREEMENT_WINDOW), pick-position
    histogram per judge, and calibration of the top probability per judge."""
    lines = [
        "",
        f"Judge agreement (|pick_a - pick_b| <= {AGREEMENT_WINDOW}, runs where both picked)",
        "",
        "| judges | n | agree |",
        "|---|---|---|",
    ]
    for (a, b), (agree, n) in judge_agreement(rows).items():
        lines.append(f"| {a} vs {b} | {n} | {_ratio(agree, n)} |")
    lines += [
        "",
        "Judge pick position (thirds of 1..T; `=T` is the exact last turn, also in `last`)",
        "",
        "| slot | n | first | middle | last | =T |",
        "|---|---|---|---|---|---|",
    ]
    for slot, h in pick_histogram(rows).items():
        lines.append(f"| {slot} | {h['n']} | {h['first']} | {h['middle']} | {h['last']} | "
                     f"{h['exact_last']} |")
    lines += [
        "",
        "Judge calibration: top stated p vs observed Hit(w=1) (rows with a distribution and "
        "defined t*)",
        "",
        "| slot | " + " | ".join(f"p in [{lo:g},{hi:g}{']' if hi == 1.0 else ')'}"
                                  for lo, hi in CALIBRATION_BINS) + " |",
        "|---|" + "---|" * len(CALIBRATION_BINS),
    ]
    for slot, bins in judge_calibration(rows).items():
        lines.append(f"| {slot} | " + " | ".join(_ratio(h, n) for _, h, n in bins) + " |")
    return lines


def report_table(rows: list[Q1Row], windows: tuple[int, ...] = (0, 1, 2, 3)) -> str:
    """Markdown: non-terminal headline, terminal column, undefined-t* rate, Hit(w) per w,
    then the same per localizer (all, no_g3, g3_unrecovered, G1..G5, judge slots), then the
    judge agreement / pick-position / calibration blocks."""
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
    guard_locs = [loc for loc in LOCALIZERS if loc not in JUDGE_SLOTS]
    for loc in (*guard_locs, *judge_slots_present(rows)):
        lines += _hit_rows(rows, loc, windows)
    lines += judge_sections(rows, windows)
    return "\n".join(lines) + "\n"
