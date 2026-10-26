"""Zero-rollout LLM-judge rows: for each failed trial, ask a judge slot from
config/judges.toml which live turn first went irrecoverably wrong (distribution over turns).
Writes results/<job>/<trial>/judge_<judge_id>.json.

    # 1) debug the template on design-split failures (prompt may still be a draft)
    uv run python scripts/llm_judge.py --jobs jobs/w0-mini-9b-mlx --dev-only --limit 5 \
        --all-judges --max-usd 2
    # 2) freeze config/judge_freeze.toml, then run on fresh non-dev failures
    uv run python scripts/llm_judge.py --jobs jobs/w0-screen-9b-t06-s12 --judge-id judge_open

Non-dev runs require the current template to match a frozen entry in
config/judge_freeze.toml. Existing judge_<id>.json files are skipped unless --overwrite.
API keys come only from the environment variable named per slot (`api_key_env`, override
with --api-key-env). judge_claude is eval-only (AGENTS.md rule 3): its records are tagged
eval_only and never feed training. Serve judge_local with e.g.
    uvx --from mlx-lm mlx_lm.server --model mlx-community/Qwen3-30B-A3B-4bit --port 8200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guardfork.freeze import load_dev_set  # noqa: E402
from guardfork.judge import (  # noqa: E402
    JudgeSpec,
    estimate_usd,
    frozen_version_for,
    judge_path,
    judge_trial,
    load_judges,
    load_template,
    make_call,
    prepare_prompt,
    prompt_sha256,
    top_k,
    write_judge,
)
from guardfork.telemetry.trials import iter_trials  # noqa: E402


def _summary(result: dict, est_usd: float) -> str:
    dist = result.get("distribution")
    top3 = [(t, round(p, 3)) for t, p in top_k(dist, 3)]
    trunc = result.get("truncation") or {}
    return json.dumps({
        "trial_id": result["trial_id"], "judge_id": result["judge_id"], "T": result["T"],
        "turn": result["turn"], "top3": top3, "elided": len(trunc.get("elided_turns") or []),
        "prompt_chars": trunc.get("prompt_chars"), "est_usd": round(est_usd, 5),
        "error": result["error"],
    }, ensure_ascii=False)


def main() -> int:
    judges = load_judges()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=Path, nargs="+", required=True)
    ap.add_argument("--judge-id", choices=sorted(judges), default=None,
                    help="judge slot from config/judges.toml")
    ap.add_argument("--all-judges", action="store_true", help="run every slot in judges.toml")
    ap.add_argument("--api-key-env", default=None,
                    help="override the env var name holding the API key (never the key)")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="stop when the cumulative cost estimate exceeds this")
    ap.add_argument("--trial", action="append", default=[], help="restrict to trial dir names")
    ap.add_argument("--dev-only", action="store_true",
                    help="judge only dev-set failures (prompt debugging)")
    ap.add_argument("--include-dev", action="store_true",
                    help="do not exclude dev-set jobs/tasks (default: excluded)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=None, help="override the slot timeout")
    ap.add_argument("--obs-chars", type=int, default=None, help="override [common] obs_chars")
    ap.add_argument("--max-prompt-chars", type=int, default=None,
                    help="override [common] max_prompt_chars")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--print-prompt", action="store_true",
                    help="print the first prompt (+ truncation log) and exit (no model call)")
    args = ap.parse_args()

    if args.all_judges:
        slots: list[JudgeSpec] = [judges[k] for k in sorted(judges)]
    elif args.judge_id:
        slots = [judges[args.judge_id]]
    else:
        ap.error("give --judge-id or --all-judges")

    template = load_template()
    sha = prompt_sha256()
    dev = load_dev_set()
    records = [r for j in args.jobs for r in iter_trials(j) if r.reward == 0.0 and r.events]
    if args.trial:
        records = [r for r in records if r.trial_dir.name in set(args.trial)]
    if args.dev_only:
        records = [r for r in records if dev.excludes(r.job, r.task_id)]
    elif not args.include_dev:
        records = [r for r in records if not dev.excludes(r.job, r.task_id)]
    if args.limit is not None:
        records = records[: args.limit]

    def trunc_args(spec: JudgeSpec) -> dict:
        return {"obs_chars": args.obs_chars if args.obs_chars is not None else spec.obs_chars,
                "max_prompt_chars": (args.max_prompt_chars if args.max_prompt_chars is not None
                                     else spec.max_prompt_chars)}

    if args.print_prompt:
        if not records:
            print("no trials")
            return 0
        prompt, trunc = prepare_prompt(records[0], template, **trunc_args(slots[0]))
        print(prompt)
        print(json.dumps({"truncation": trunc}), file=sys.stderr)
        return 0
    version = frozen_version_for(sha)
    if not args.dev_only and version is None:
        print(f"judge prompt sha256 {sha[:12]} is not frozen in config/judge_freeze.toml; "
              "debug with --dev-only, then freeze it", file=sys.stderr)
        return 2

    total_usd = 0.0
    for spec in slots:
        call = make_call(spec, api_key_env=args.api_key_env, timeout=args.timeout)
        print(f"[{spec.judge_id}] {len(records)} failed trials; prompt {sha[:12]} "
              f"({version or 'draft'}); {spec.provider} {spec.model} @ "
              f"{spec.base_url or 'default'}{' (eval_only)' if spec.eval_only else ''}",
              flush=True)
        for rec in records:
            if args.max_usd is not None and total_usd > args.max_usd:
                print(f"stop: cumulative estimate ${total_usd:.4f} exceeds --max-usd "
                      f"{args.max_usd}", file=sys.stderr)
                return 3
            out = judge_path(args.results_dir, rec.job, rec.trial_dir.name, spec.judge_id)
            if out.is_file() and not args.overwrite:
                print(f"skip {rec.job}/{rec.trial_dir.name}: {out.name} exists", flush=True)
                continue
            result = judge_trial(
                rec, call, judge_id=spec.judge_id, provider=spec.provider, model=spec.model,
                template=template, template_sha256=sha, sampling=spec.sampling,
                eval_only=spec.eval_only, **trunc_args(spec))
            result["dev"] = dev.excludes(rec.job, rec.task_id)
            est = estimate_usd(result.get("usage"), spec.usd_per_m_input, spec.usd_per_m_output)
            result["est_usd"] = est
            total_usd += est
            write_judge(result, args.results_dir)
            print(_summary(result, est), flush=True)
    print(f"total estimated cost: ${total_usd:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
