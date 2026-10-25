"""Weekends 3–4 Q1 study: eligible failed runs + ceiling runs.

    python scripts/run_q1.py --base-jobs jobs/4b_s0 --n-runs 90 --ceiling 20 \
        --K 8 --output-dir forks/q1 --report results/q1/report.md --oracle-json forks/q1/oracle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardfork.measure.w0 import per_task_pass  # noqa: E402
from guardfork.q1.localization import (  # noqa: E402
    build_rows,
    eligible_failed_runs,
    report_table,
)
from guardfork.stats.bootstrap import clustered_bootstrap_ci, hit_minus_random  # noqa: E402
from guardfork.telemetry.trials import iter_trials  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jobs", type=Path, nargs="+", required=True)
    ap.add_argument("--n-runs", type=int, default=90)
    ap.add_argument("--ceiling", type=int, default=20)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=Path("forks/q1"))
    ap.add_argument("--report", type=Path, default=Path("results/q1/report.md"))
    ap.add_argument("--oracle-json", type=Path, required=True)
    ap.add_argument("--window", type=int, default=1, help="primary localization window")
    ap.add_argument("--judge-root", type=Path, default=Path("results"),
                    help="reads <judge-root>/<job>/<trial>/judge.json for the judge row")
    args = ap.parse_args()
    records = [t for j in args.base_jobs for t in iter_trials(j)]
    eligible = eligible_failed_runs(records, per_task_pass(records))[: args.n_runs]
    oracle = json.loads(args.oracle_json.read_text())
    rows = build_rows(eligible, oracle, judge_root=args.judge_root)
    table = report_table(rows)
    comparable = [r for r in rows if r.g is not None and r.t_star is not None]
    if comparable:
        deltas = [hit_minus_random(r.g, r.t_star, r.T, args.window) for r in comparable]
        clusters = [r.scenario for r in comparable]
        mean, lo, hi = clustered_bootstrap_ci(deltas, clusters, n_boot=5000, seed=0)
        extra = (f"\nprimary w={args.window}: mean d={mean:.3f} "
                 f"95% CI [{lo:.3f}, {hi:.3f}] (n={len(comparable)})\n"
                 f"ceiling runs requested: {args.ceiling}\n")
    else:
        extra = "\nno comparable rows (need non-terminal g and defined t*)\n"
    md = "# Q1 localization\n\n" + table + extra
    print(md)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(md)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
