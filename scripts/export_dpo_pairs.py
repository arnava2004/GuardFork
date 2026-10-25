"""Export same-state DPO pairs from fork continuations (neutral + mlx-lm-lora formats).

    # one ReplayPlan + its fork job (e.g. the Mac fork smoke test)
    uv run python scripts/export_dpo_pairs.py --plan forks/guard_fork/302c169_2__UZLvP3g_t3.json \
        --fork-job forks/guard_fork/guard-fork-302c169_2-t3 --out runs/dpo/smoke
    # referee output: results/<job>/<trial>/forks.json + the base jobs holding the sources
    uv run python scripts/export_dpo_pairs.py --forks-json-root results \
        --base-jobs jobs/w0-eval-9b-s0 --out runs/dpo/referee

Writes <out>/neutral.jsonl, <out>/mlx/{train,valid}.jsonl (mlx-lm-lora `--data <out>/mlx`),
and <out>/stats.json. Output stays under runs/ (git-blocked: it contains task text).
--smoke (--allow-passing-source / --allow-collision / --allow-identical) exists only to get a
pair out of the greedy-era Mac fork smoke jobs for a plumbing dry run; never train on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guardfork.telemetry.trials import iter_trials  # noqa: E402
from guardfork.train.export import (  # noqa: E402
    ExportStats,
    pairs_from_forks_json,
    pairs_from_plan,
    to_mlx_row,
    to_neutral_row,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", type=Path, action="append", default=[],
                    help="ReplayPlan JSON (pair with --fork-job, same order)")
    ap.add_argument("--fork-job", type=Path, action="append", default=[])
    ap.add_argument("--forks-json-root", type=Path, default=None,
                    help="results dir with <job>/<trial>/forks.json")
    ap.add_argument("--base-jobs", type=Path, nargs="*", default=[])
    ap.add_argument("--max-per-fork", type=int, default=1)
    ap.add_argument("--allow-prefix-mismatch", action="store_true")
    ap.add_argument("--allow-passing-source", action="store_true",
                    help="plumbing smoke tests only: use a passing source's action as rejected")
    ap.add_argument("--allow-collision", action="store_true",
                    help="plumbing smoke tests only: keep same-API-signature pairs (flagged)")
    ap.add_argument("--allow-identical", action="store_true",
                    help="plumbing smoke tests only: keep byte-identical chosen/rejected")
    ap.add_argument("--smoke", action="store_true",
                    help="= --allow-passing-source --allow-collision --allow-identical")
    ap.add_argument("--drop-think-only", action="store_true")
    ap.add_argument("--valid-copies", type=int, default=1,
                    help="mlx-lm-lora needs a non-empty valid.jsonl; copy the first N rows")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if len(args.plan) != len(args.fork_job):
        ap.error("--plan and --fork-job must be given the same number of times")

    if args.smoke:
        args.allow_passing_source = args.allow_collision = args.allow_identical = True
    kwargs = {"max_per_fork": args.max_per_fork,
              "require_prefix_match": not args.allow_prefix_mismatch,
              "allow_collision": args.allow_collision,
              "allow_identical": args.allow_identical}
    pairs, stats = [], ExportStats()
    for plan, fork_job in zip(args.plan, args.fork_job, strict=True):
        got, st = pairs_from_plan(plan, fork_job, root=ROOT, **kwargs)
        if not got and args.allow_passing_source and st.skipped_source:
            got, st = _passing_source(plan, fork_job, kwargs)
        pairs += got
        stats.add(st)
    if args.forks_json_root is not None:
        sources = {f"{r.job}/{r.trial_dir.name}": r for j in args.base_jobs for r in iter_trials(j)}
        for forks_json in sorted(args.forks_json_root.glob("*/*/forks.json")):
            got, st = pairs_from_forks_json(forks_json, sources, root=ROOT, **kwargs)
            pairs += got
            stats.add(st)
    if args.drop_think_only:
        pairs = [p for p in pairs if not p.think_only_diff]

    write_jsonl([to_neutral_row(p) for p in pairs], args.out / "neutral.jsonl")
    mlx_rows = [to_mlx_row(p) for p in pairs]
    write_jsonl(mlx_rows, args.out / "mlx" / "train.jsonl")
    write_jsonl(mlx_rows[: args.valid_copies], args.out / "mlx" / "valid.jsonl")
    summary = {**asdict(stats), "exported": len(pairs)}
    (args.out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if pairs else 1


def _passing_source(plan: Path, fork_job: Path, kwargs: dict):
    from guardfork.fork.harbor_replay import load_plan
    from guardfork.telemetry.trials import load_trial
    from guardfork.train.export import build_fork_pairs

    p = load_plan(plan)
    src_dir = p.source_trial if p.source_trial.is_absolute() else ROOT / p.source_trial
    source = load_trial(src_dir / "agent" / "teacher", src_dir.parent.name)
    source.reward = 0.0
    conts = list(iter_trials(fork_job, fork_job.name))
    return build_fork_pairs(source, p.fork_turn, conts, message_prefix=p.message_prefix,
                            arm="SMOKE", **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
