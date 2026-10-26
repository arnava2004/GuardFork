"""W1 reward parity: replay N complete trajectories (0 model calls) through Harbor and
require the official verifier to reproduce each original reward.

    uv run python scripts/replay_parity.py --jobs jobs/w0-dev-9b-t06 --n 20
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_appworld_rollout import build_harbor_command, harbor_env  # noqa: E402

from guardfork.fork.harbor_replay import full_replay_plan, save_plan  # noqa: E402
from guardfork.telemetry.trials import iter_trials  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--hz-root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "harness-zero")
    ap.add_argument("--student-model", default="openai:mlx-community/Qwen3.5-4B-4bit")
    ap.add_argument("--student-base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--output-dir", type=Path, default=Path("forks/replay_parity"))
    args = ap.parse_args()

    hz_root = args.hz_root.resolve()
    out = args.output_dir.resolve()
    records = [r for r in iter_trials(args.jobs) if r.reward is not None and r.events]
    passed = [r for r in records if r.passed]
    failed = [r for r in records if not r.passed]
    picked = (passed[: args.n // 2] + failed)[: args.n]
    if len(picked) < args.n:
        picked = records[: args.n]

    reproduced, rows = 0, []
    for i, rec in enumerate(picked):
        plan = full_replay_plan(rec)
        plan_path = save_plan(plan, out / "plans" / f"{rec.trial_dir.name}.json")
        job_name = f"parity-{i:02d}-{rec.trial_dir.name}"
        cmd = build_harbor_command(
            hz_root=hz_root, tasks=[plan.task_id], student_model=args.student_model,
            student_base_url=args.student_base_url, job_name=job_name, output_dir=out,
            replay_plan_path=plan_path,
        )
        subprocess.run(cmd, cwd=hz_root, env=harbor_env(hz_root), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        got = next((r for r in iter_trials(out / job_name)), None)
        new_reward = got.reward if got else None
        ok = new_reward is not None and new_reward == rec.reward
        reproduced += ok
        error = None
        if got and got.exception:
            error = str(got.exception)[:120]
        rows.append((rec.trial_dir.name, rec.reward, new_reward, ok, error))
        print(f"{'OK ' if ok else 'BAD'} {rec.trial_dir.name}: original={rec.reward} "
              f"replayed={new_reward} commands={len(plan.commands)}"
              + (f" error={error}" if error else ""), flush=True)
    print(f"\nreward parity: {reproduced}/{len(picked)}")
    return 0 if reproduced == len(picked) else 1


if __name__ == "__main__":
    raise SystemExit(main())
