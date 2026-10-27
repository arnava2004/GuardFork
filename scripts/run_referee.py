"""Monte Carlo referee (Q1 oracle) on fresh failed runs, K continuations per probed state.

    uv run python scripts/run_referee.py --base-jobs jobs/w0-eval-9b-s0 jobs/w0-eval-9b-s1 \
        --n-runs 20 --K 8 --concurrency 4 \
        --student-model openai:mlx-community/Qwen3.5-9B-4bit

Dev-set jobs/tasks (config/dev_set.toml) are excluded unless --allow-dev (smoke tests only).
Results go to results/<job>/<trial>/{oracle.json,forks.json}; the viewer shows them.
Re-running skips trials that already have oracle.json and reuses finished probe jobs (their
env-seconds are not charged to the cost ledger again). Judge picks from
<judge-root>/<job>/<trial>/judge_<slot>.json (--judge-slots) add mandatory {j-1, j, j+1}
probe windows. A probe with no graded continuation stops that trial: no oracle.json is
written, the failed probe job is deleted so the next run relaunches it, and Harbor's output
is kept in <forks-dir>/<job_name>.harbor.log.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_appworld_rollout import build_harbor_command, harbor_env  # noqa: E402

from guardfork.cost import CostEntry, CostLedger  # noqa: E402
from guardfork.fork.referee import run_referee, write_results  # noqa: E402
from guardfork.freeze import load_dev_set  # noqa: E402
from guardfork.judge import judge_path  # noqa: E402
from guardfork.measure.w0 import per_task_pass  # noqa: E402
from guardfork.q1.localization import eligible_failed_runs  # noqa: E402
from guardfork.telemetry.trials import TrialRecord, iter_trials  # noqa: E402


def judge_picks(rec: TrialRecord, judge_root: Path, slots: list[str]) -> dict[str, list[int]]:
    """slot -> [argmax turn] for every slot with a judge record and a pick."""
    picks: dict[str, list[int]] = {}
    for slot in slots:
        path = judge_path(judge_root, rec.job, rec.trial_dir.name, slot)
        if path.is_file():
            turn = json.loads(path.read_text(encoding="utf-8")).get("turn")
            if isinstance(turn, int) and not isinstance(turn, bool):
                picks[slot] = [turn]
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jobs", type=Path, nargs="+", required=True)
    ap.add_argument("--trial", action="append", default=[],
                    help="restrict to these trial dir names")
    ap.add_argument("--n-runs", type=int, default=20)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-probes", type=int, default=10)
    ap.add_argument("--band", type=float, nargs=2, default=(0.3, 0.8))
    ap.add_argument("--min-samples", type=int, default=8)
    ap.add_argument("--allow-dev", action="store_true", help="smoke tests only")
    ap.add_argument("--student-model", default="openai:mlx-community/Qwen3.5-9B-4bit")
    ap.add_argument("--student-base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--hz-root", type=Path, default=ROOT.parent / "harness-zero")
    ap.add_argument("--forks-dir", type=Path, default=ROOT / "forks" / "referee")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--judge-root", type=Path, default=ROOT / "results",
                    help="where scripts/llm_judge.py wrote judge_<slot>.json")
    ap.add_argument("--judge-slots", nargs="*", default=["judge_open", "judge_claude"],
                    help="judge picks that add mandatory probe windows (none: no judge windows)")
    args = ap.parse_args()

    hz_root = args.hz_root.resolve()
    records = [r for j in args.base_jobs for r in iter_trials(j)]
    counts = per_task_pass(records)
    dev = None if args.allow_dev else load_dev_set()
    runs = eligible_failed_runs(records, counts, tuple(args.band), args.min_samples, dev)
    if args.trial:
        runs = [r for r in runs if r.trial_dir.name in set(args.trial)]
    runs = runs[: args.n_runs]
    print(f"{len(runs)} eligible failed runs (band {args.band}, >= {args.min_samples} samples)",
          flush=True)
    ledger = CostLedger(ROOT / "runs" / "cost" / "referee.jsonl")
    forks_dir = args.forks_dir.resolve()
    reused_seconds = [0.0]

    def launch(plan_path: Path, K: int, job_name: str) -> Path:  # noqa: N803
        job_dir = forks_dir / job_name
        if (job_dir / "result.json").is_file() and json.loads(
                (job_dir / "result.json").read_text()).get("finished_at"):
            reused_seconds[0] += sum(r.wall_seconds or 0.0
                                     for r in iter_trials(job_dir, job_dir.name))
            return job_dir
        if job_dir.exists():
            # An interrupted probe job: Harbor will not reuse its dir, so a rerun would
            # silently produce 0/0. Start the probe over.
            print(f"rerunning interrupted probe job {job_name}", flush=True)
            shutil.rmtree(job_dir)
        plan = json.loads(plan_path.read_text())
        cmd = build_harbor_command(
            hz_root=hz_root, tasks=[plan["task_id"]], student_model=args.student_model,
            student_base_url=args.student_base_url, job_name=job_name, output_dir=forks_dir,
            attempts=K, concurrency=args.concurrency, replay_plan_path=plan_path,
        )
        forks_dir.mkdir(parents=True, exist_ok=True)
        with (forks_dir / f"{job_name}.harbor.log").open("w") as log:
            proc = subprocess.run(cmd, cwd=hz_root, env=harbor_env(hz_root), check=False,
                                  stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            print(f"harbor exited {proc.returncode} for {job_name}; see {log.name}", flush=True)
        return job_dir

    for rec in runs:
        out = args.results_dir / rec.job / rec.trial_dir.name / "oracle.json"
        if out.is_file():
            print(f"skip {rec.trial_dir.name}: oracle.json exists", flush=True)
            continue
        reused_seconds[0] = 0.0
        res = run_referee(rec, base_pass=counts.get(rec.task_id or "", (0, 0)), K=args.K,
                          launch=launch, plans_dir=forks_dir / "plans",
                          max_probes=args.max_probes,
                          extra_turns=judge_picks(rec, args.judge_root, args.judge_slots))
        ledger.charge(CostEntry("referee", res.trial_id, res.env_seconds - reused_seconds[0],
                                0.0, 0, 0))
        if res.failed_probes:
            for t in res.failed_probes:
                shutil.rmtree(forks_dir / f"ref-{rec.trial_dir.name}-t{t}", ignore_errors=True)
            print(f"{res.trial_id}: INCOMPLETE, no graded continuations at t={res.failed_probes}; "
                  "no oracle.json written, rerun to retry", flush=True)
            continue
        write_results(res, args.results_dir)
        vals = {t: f"{p}/{n}" for t, (p, n) in sorted(res.values.items())}
        print(f"{res.trial_id}: T={res.T} g={res.g} t*={res.t_star} drop={res.t_star_drop} "
              f"V={vals} {res.skipped or ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
