"""Weekend 1 replay validations (0 LLM calls): request reconstruction, 20/20 reward parity,
and the determinism check that feeds W1-a.

    python scripts/replay_checks.py reconstruct --trials jobs/4b_s0
    python scripts/replay_checks.py reward-parity --trials jobs/4b_s0 --n 20
    python scripts/replay_checks.py determinism --trials jobs/4b_s0 --n 20 --turns 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardfork.fork.replay_checks import (  # noqa: E402
    determinism_check,
    offline_request_reconstruction,
)
from guardfork.telemetry.trials import iter_trials  # noqa: E402


def _trial_dirs(root: Path, n: int | None) -> list[Path]:
    dirs, seen = [], set()
    for rec in iter_trials(root):
        path = rec.trial_dir
        if path in seen:
            continue
        seen.add(path)
        dirs.append(path)
        if n is not None and len(dirs) >= n:
            break
    return dirs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["reconstruct", "reward-parity", "determinism"])
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--turns", type=int, default=3)
    args = ap.parse_args()
    if args.check == "reconstruct":
        ok = True
        for path in _trial_dirs(args.trials, args.n):
            report = offline_request_reconstruction(path)
            print(f"{path}: {report.canonical_equal}/{report.turns_checked} canonical")
            ok = ok and report.canonical_equal == report.turns_checked
        return 0 if ok else 1
    dirs = _trial_dirs(args.trials, args.n)
    if args.check == "reward-parity":
        raise SystemExit("reward-parity needs a Harbor replay callback; run via tests or --execute")
    frac = determinism_check(dirs, turns_per_trial=args.turns)
    print(f"determinism: {frac:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
