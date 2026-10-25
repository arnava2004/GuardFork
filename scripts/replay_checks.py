"""Weekend 1 replay validations (0 LLM calls): request reconstruction, 20/20 reward parity,
and the determinism check that feeds W1-a.

    python scripts/replay_checks.py reconstruct --trials jobs/4b_s0
    python scripts/replay_checks.py reward-parity --trials jobs/4b_s0 --n 20
    python scripts/replay_checks.py determinism --trials jobs/4b_s0 --n 20 --turns 3
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=["reconstruct", "reward-parity", "determinism"])
    ap.add_argument("--trials", type=Path, required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--turns", type=int, default=3)
    ap.parse_args()
    raise NotImplementedError("W1: see guardfork/fork/replay_checks.py")


if __name__ == "__main__":
    raise SystemExit(main())
