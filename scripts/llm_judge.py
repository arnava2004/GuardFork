"""Zero-rollout LLM-judge baseline: for each failed trial, ask a local open-weight model
which live turn first went irrecoverably wrong. Writes results/<job>/<trial>/judge.json.

    # 1) debug the template on dev failures (prompt may still be a draft)
    uv run python scripts/llm_judge.py --jobs jobs/w0-mini-9b-mlx --dev-only --limit 5
    # 2) freeze config/judge_freeze.toml, then run on fresh non-dev failures
    uv run python scripts/llm_judge.py --jobs jobs/w0-screen-9b-t06-s12

Non-dev runs require the current template to match a frozen entry in
config/judge_freeze.toml. Existing judge.json files are skipped unless --overwrite.
Serve the judge with e.g.
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
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OBS_CHARS,
    build_prompt,
    frozen_version_for,
    judge_path,
    judge_trial,
    load_template,
    make_openai_call,
    prompt_sha256,
    write_judge,
)
from guardfork.telemetry.trials import iter_trials  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=Path, nargs="+", required=True)
    ap.add_argument("--trial", action="append", default=[], help="restrict to trial dir names")
    ap.add_argument("--dev-only", action="store_true",
                    help="judge only dev-set failures (prompt debugging)")
    ap.add_argument("--include-dev", action="store_true",
                    help="do not exclude dev-set jobs/tasks (default: excluded)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--obs-chars", type=int, default=OBS_CHARS)
    ap.add_argument("--no-think", action="store_true",
                    help="send chat_template_kwargs.enable_thinking=false")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--print-prompt", action="store_true",
                    help="print the first prompt and exit (no model call)")
    args = ap.parse_args()

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

    if args.print_prompt:
        print(build_prompt(records[0], template, args.obs_chars) if records else "no trials")
        return 0
    version = frozen_version_for(sha)
    if not args.dev_only and version is None:
        print(f"judge prompt sha256 {sha[:12]} is not frozen in config/judge_freeze.toml; "
              "debug with --dev-only, then freeze it", file=sys.stderr)
        return 2

    extra = {"chat_template_kwargs": {"enable_thinking": False}} if args.no_think else None
    call = make_openai_call(args.base_url, args.model, max_tokens=args.max_tokens,
                            timeout=args.timeout, extra_body=extra)
    print(f"{len(records)} failed trials; prompt {sha[:12]} ({version or 'draft'}); "
          f"model {args.model} @ {args.base_url}", flush=True)
    for rec in records:
        out = judge_path(args.results_dir, rec.job, rec.trial_dir.name)
        if out.is_file() and not args.overwrite:
            print(f"skip {rec.job}/{rec.trial_dir.name}: judge.json exists", flush=True)
            continue
        result = judge_trial(rec, call, model=args.model, template=template,
                             template_sha256=sha, obs_chars=args.obs_chars)
        result["dev"] = dev.excludes(rec.job, rec.task_id)
        write_judge(result, args.results_dir)
        print(json.dumps({k: result[k] for k in ("trial_id", "T", "turn", "reason", "error")},
                         ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
