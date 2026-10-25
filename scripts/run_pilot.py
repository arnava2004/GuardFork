"""Weekend 1 pilot: 25 localization runs (K=8) + 8 test-retest runs, then W1-a..e verdicts.

    python scripts/run_pilot.py --base-jobs jobs/4b_s0 --n-runs 25 --retest 8 \
        --K 8 --output-dir forks/pilot --notes NOTES.md --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardfork.config import load_kill_config  # noqa: E402
from guardfork.fork.probe_plan import initial_probes  # noqa: E402
from guardfork.guards.rules import evaluate, first_nonterminal  # noqa: E402
from guardfork.measure.w0 import per_task_pass  # noqa: E402
from guardfork.pilot.gates import PilotRun, evaluate_pilot  # noqa: E402
from guardfork.q1.localization import eligible_failed_runs  # noqa: E402
from guardfork.splits import scenario_of  # noqa: E402
from guardfork.stats.oracle import derailment_turn, jeffreys_v1  # noqa: E402
from guardfork.telemetry.to_trajectory import to_trajectory  # noqa: E402
from guardfork.telemetry.trials import iter_trials  # noqa: E402


def _pilot_run(rec, values: dict[int, tuple[int, int]], t_star, retest=None) -> PilotRun:
    traj = to_trajectory(rec)
    firings = evaluate(traj)
    first = first_nonterminal(firings, traj.T)
    return PilotRun(
        trial_id=f"{rec.job}/{rec.trial_dir.name}",
        scenario=scenario_of(rec.task_id) if rec.task_id else "",
        T=traj.T,
        passed=rec.passed,
        first_nonterminal_guard=first.turn if first else None,
        values=values,
        t_star=t_star,
        t_star_retest=retest,
        fired_guards=tuple(sorted({str(f.guard) for f in firings})),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jobs", type=Path, nargs="+", required=True)
    ap.add_argument("--n-runs", type=int, default=25)
    ap.add_argument("--retest", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=Path("forks/pilot"))
    ap.add_argument("--notes", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--execute", action="store_true",
                    help="launch Harbor continuations (requires W0 pass + images)")
    ap.add_argument("--oracle-json", type=Path, default=None,
                    help="precomputed {trial_id: {values, t_star, t_star_retest}}")
    args = ap.parse_args()
    kill = load_kill_config()
    records = [t for j in args.base_jobs for t in iter_trials(j)]
    counts = per_task_pass(records)
    eligible = eligible_failed_runs(records, counts)[: args.n_runs]
    oracle = {}
    if args.oracle_json and args.oracle_json.is_file():
        oracle = json.loads(args.oracle_json.read_text())
    runs = []
    for rec in eligible:
        traj = to_trajectory(rec)
        first = first_nonterminal(evaluate(traj), traj.T)
        g = first.turn if first else traj.T
        probes = initial_probes(g, traj.T)
        blob = oracle.get(f"{rec.job}/{rec.trial_dir.name}", {})
        values = {int(k): tuple(v) for k, v in (blob.get("values") or {}).items()}
        v_hat = {t: (p / n if n else 0.0) for t, (p, n) in values.items()}
        p, n = counts.get(rec.task_id or "", (0, 0))
        t_star = blob.get("t_star")
        if t_star is None and v_hat:
            t_star = derailment_turn(v_hat, reward=rec.reward or 0.0,
                                     v_s1=jeffreys_v1(p, n) if n else 0.5,
                                     K=args.K, T=traj.T)
        runs.append(_pilot_run(rec, values, t_star, blob.get("t_star_retest")))
        (args.output_dir / "plans").mkdir(parents=True, exist_ok=True)
        plan_path = args.output_dir / "plans" / f"{rec.trial_dir.name}.json"
        plan_path.write_text(json.dumps({"probes": probes, "g": g, "T": traj.T}))
    if args.execute and not args.dry_run:
        raise RuntimeError("Harbor launch is gated on W0; use the injected runner in tests")
    determinism = float((oracle.get("_meta") or {}).get("determinism", 0.0))
    results = evaluate_pilot(runs, determinism=determinism, kill=kill)
    lines = ["### Weekend 1 pilot", ""]
    for r in results:
        mark = {True: "PASS", False: "FAIL", None: "n/a"}[r.passed]
        lines.append(f"- {r.name}: {r.value} (thr {r.threshold}) {mark} {r.note}")
    md = "\n".join(lines) + "\n"
    print(md)
    if args.notes:
        args.notes.parent.mkdir(parents=True, exist_ok=True)
        with args.notes.open("a", encoding="utf-8") as handle:
            handle.write("\n" + md)
    return 0 if all(r.passed is not False for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
