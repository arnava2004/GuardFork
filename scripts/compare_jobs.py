"""Side-by-side comparison of two students on the same task list.

Each side is a label, a proxy log, and one or more job dirs (e.g. seeds 1-2 and 3-4):

    uv run python scripts/compare_jobs.py \
        --a 4B runs/proxy/<4b_log>.jsonl jobs/<4b_job> \
        --b 9B runs/proxy/<9b_log>.jsonl jobs/<9b_seed12_job> jobs/<9b_seed34_job>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from guardfork.telemetry.proxy_log import iter_proxy_log  # noqa: E402
from guardfork.telemetry.trials import iter_trials  # noqa: E402

TIMEOUT_EXCEPTIONS = {"OpenAITimeoutError", "APITimeoutError"}


def _window(job_dir: Path) -> tuple[float, float]:
    d = json.loads((job_dir / "result.json").read_text())
    start = datetime.fromisoformat(d["started_at"]).timestamp()
    end = datetime.fromisoformat(d.get("finished_at") or d["updated_at"]).timestamp()
    return start, end


def _exc_type(trial) -> str | None:
    e = trial.exception
    if isinstance(e, dict):
        return e.get("exception_type")
    return None if e is None else str(e)


def _stats(proxy_path: Path, job_dirs: list[Path], band: tuple[float, float],
           drop_timeouts: bool = False) -> dict:
    all_trials = [t for j in job_dirs for t in iter_trials(j)]
    dropped = [t for t in all_trials if drop_timeouts and _exc_type(t) in TIMEOUT_EXCEPTIONS]
    dropped_dirs = {t.trial_dir for t in dropped}
    trials = [t for t in all_trials if t.trial_dir not in dropped_dirs]
    windows = [_window(j) for j in job_dirs]
    proxy = [p for p in iter_proxy_log(proxy_path)
             if p.ok and any(s <= float(p.raw.get("ts", 0)) <= e for s, e in windows)]
    by_task: dict[str, list[float]] = defaultdict(list)
    for t in trials:
        by_task[t.task_id or "?"].append(t.reward or 0.0)
    turns = [turn for t in trials for turn in t.turns]
    walls = [t.wall_seconds for t in trials if t.wall_seconds is not None]
    prompt = [p.prompt_tokens for p in proxy if p.prompt_tokens is not None]
    completion = [p.completion_tokens for p in proxy if p.completion_tokens is not None]
    passes = {k: sum(1 for r in v if r >= 1.0) for k, v in by_task.items()}
    seeds = {k: len(v) for k, v in by_task.items()}
    n_seeds = max(seeds.values(), default=0)
    hist = {k: sum(1 for t, v in passes.items() if v == k and seeds[t] == n_seeds)
            for k in range(n_seeds + 1)}
    in_band = [t for t in passes if band[0] <= passes[t] / seeds[t] <= band[1]]
    return {
        "n": len(trials),
        "n_seeds": n_seeds,
        "short_tasks": sorted(t for t in seeds if seeds[t] < n_seeds),
        "tgc": 100 * statistics.mean([t.reward or 0.0 for t in trials]) if trials else 0.0,
        "passes": passes,
        "seeds": seeds,
        "hist": hist,
        "k1": len(in_band) / max(1, len(passes)),
        "k1_n": len(in_band),
        "n_tasks": len(passes),
        "parse": 100 * sum(t.parse_failure for t in turns) / max(1, len(turns)),
        "n_turns": len(turns),
        "max_turns": sum(1 for t in trials if t.hit_turn_limit),
        "timeouts": sum(1 for t in all_trials if _exc_type(t) in TIMEOUT_EXCEPTIONS),
        "dropped": len(dropped),
        "errors": sum(1 for t in trials
                      if t.reward is None and not t.hit_turn_limit
                      and _exc_type(t) not in TIMEOUT_EXCEPTIONS),
        "wall_p50": statistics.median(walls) if walls else float("nan"),
        "job_minutes": sum(e - s for s, e in windows) / 60,
        "prompt_p50": statistics.median(prompt) if prompt else float("nan"),
        "completion_per_ep": sum(completion) / max(1, len(trials)),
        "requests": len(proxy),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", metavar="LABEL PROXY JOB", required=True)
    ap.add_argument("--b", nargs="+", metavar="LABEL PROXY JOB", required=True)
    ap.add_argument("--band", nargs=2, type=float, default=(0.25, 0.75))
    ap.add_argument("--drop-timeouts", action="store_true",
                    help="exclude client-timeout trials (infra failures re-run in another job)")
    args = ap.parse_args()
    la, lb = args.a[0], args.b[0]
    band = tuple(args.band)
    a = _stats(Path(args.a[1]), [Path(p) for p in args.a[2:]], band, args.drop_timeouts)
    b = _stats(Path(args.b[1]), [Path(p) for p in args.b[2:]], band, args.drop_timeouts)

    def row(name, fa, fb):
        print(f"| {name} | {fa} | {fb} |")

    def hist(s):
        return " / ".join(f"{k}/{s['n_seeds']}: {v}" for k, v in s["hist"].items())

    print(f"| metric | {la} | {lb} |\n|---|---|---|")
    row("trials (seeds per task)", f"{a['n']} ({a['n_seeds']})", f"{b['n']} ({b['n_seeds']})")
    row("per-task passes", hist(a), hist(b))
    row(f"K1 share, p_hat in [{band[0]}, {band[1]}]",
        f"{a['k1_n']}/{a['n_tasks']} = {a['k1']:.2f}",
        f"{b['k1_n']}/{b['n_tasks']} = {b['k1']:.2f}")
    row("TGC (sanity only)", f"{a['tgc']:.1f}", f"{b['tgc']:.1f}")
    row("parse failures, % of turns", f"{a['parse']:.1f} of {a['n_turns']}",
        f"{b['parse']:.1f} of {b['n_turns']}")
    row("40-turn exhaustions", a["max_turns"], b["max_turns"])
    row("client timeouts (dropped) / other errors",
        f"{a['timeouts']} ({a['dropped']}) / {a['errors']}",
        f"{b['timeouts']} ({b['dropped']}) / {b['errors']}")
    row("median wall per episode (s)", f"{a['wall_p50']:.0f}", f"{b['wall_p50']:.0f}")
    row("total job wall (min)", f"{a['job_minutes']:.0f}", f"{b['job_minutes']:.0f}")
    row("median prompt tokens / request", f"{a['prompt_p50']:.0f}", f"{b['prompt_p50']:.0f}")
    row("completion tokens / episode", f"{a['completion_per_ep']:.0f}",
        f"{b['completion_per_ep']:.0f}")
    for s, label in ((a, la), (b, lb)):
        if s["short_tasks"]:
            print(f"\n{label}: fewer than {s['n_seeds']} trials for {', '.join(s['short_tasks'])}")

    tasks = sorted(set(a["passes"]) | set(b["passes"]))
    print(f"\n| task | {la} | {lb} |\n|---|---|---|")
    for t in tasks:
        fa = f"{a['passes'][t]}/{a['seeds'][t]}" if t in a["passes"] else "-"
        fb = f"{b['passes'][t]}/{b['seeds'][t]}" if t in b["passes"] else "-"
        print(f"| {t} | {fa} | {fb} |")
    both = sum(1 for t in tasks if a["passes"].get(t, 0) and b["passes"].get(t, 0))
    only_a = sum(1 for t in tasks if a["passes"].get(t, 0) and not b["passes"].get(t, 0))
    only_b = sum(1 for t in tasks if b["passes"].get(t, 0) and not a["passes"].get(t, 0))
    print(f"\nsolved at least once: both {both}, only {la} {only_a}, only {lb} {only_b}, "
          f"neither {len(tasks) - both - only_a - only_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
