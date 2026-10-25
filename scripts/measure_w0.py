"""Weekend 0 measurements and K0–K4 verdicts.

    python scripts/measure_w0.py --title "K0 9B B0 test_normal" --k0 \
        --jobs jobs/k0_seed0 --proxy-log runs/proxy/k0.jsonl --notes NOTES.md

    python scripts/measure_w0.py --title "4B x4 train147" \
        --jobs jobs/4b_s0 jobs/4b_s1 jobs/4b_s2 jobs/4b_s3 --proxy-log runs/proxy/4b.jsonl \
        --second-stage-out runs/w0/4b_second_stage.txt --notes NOTES.md

Each --jobs dir is one Harbor job (one seed pass); SGC is computed per job, then averaged.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardfork.config import DEFAULT_KILL_PATH, load_kill_config  # noqa: E402
from guardfork.measure import w0  # noqa: E402
from guardfork.telemetry.proxy_log import iter_proxy_log  # noqa: E402
from guardfork.telemetry.trials import iter_trials  # noqa: E402


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted"


def _k4_estimate(proxy) -> tuple[float | None, str]:
    """Provisional K4: median prompt tokens over live turns that ended in a tool call
    (non-terminal by construction). Swap for the first-non-terminal-guard turn once
    guards/rules.py exists."""
    vals = [p.prompt_tokens for p in proxy
            if p.ok and p.prompt_tokens is not None and p.live_turn_index >= 0
            and p.ended_with_tool_call]
    note = "provisional: median over non-terminal live turns, guards not implemented"
    return (statistics.median(vals) if vals else None), note


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--jobs", type=Path, nargs="+", required=True)
    ap.add_argument("--proxy-log", type=Path, default=None)
    ap.add_argument("--kill", type=Path, default=DEFAULT_KILL_PATH)
    ap.add_argument("--k0", action="store_true", help="these jobs are the 9B test_normal K0 run")
    ap.add_argument("--model", default="")
    ap.add_argument("--notes", type=Path, default=None, help="append the markdown block here")
    ap.add_argument("--second-stage-out", type=Path, default=None,
                    help="write task ids that passed 1-3 of 4 (for the +4-seed stage)")
    args = ap.parse_args()

    kill = load_kill_config(args.kill)
    trials = [t for j in args.jobs for t in iter_trials(j)]
    proxy = list(iter_proxy_log(args.proxy_log)) if args.proxy_log else []
    summary = w0.summarize(trials, proxy, kill)
    k4, k4_note = _k4_estimate(proxy)
    results = w0.evaluate_kill(summary, kill, is_k0_run=args.k0, k4_prompt_tokens=k4,
                               k4_note=k4_note)
    meta = {"commit": _commit(), "kill.toml sha256": kill.sha256[:12], "model": args.model,
            "jobs": ", ".join(str(j) for j in args.jobs)}
    md = w0.render_markdown(args.title, summary, results, meta)
    print(md)
    if args.notes:
        with args.notes.open("a", encoding="utf-8") as f:
            f.write("\n" + md)
    if args.second_stage_out:
        args.second_stage_out.parent.mkdir(parents=True, exist_ok=True)
        args.second_stage_out.write_text("\n".join(summary.second_stage) + "\n")
        print(f"wrote {len(summary.second_stage)} second-stage task ids to "
              f"{args.second_stage_out}")
    return 1 if any(r.passed is False for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
