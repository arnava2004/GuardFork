"""Weekends 3–4 Q1 study: 80–100 eligible failed runs + 15–25 full-sweep ceiling runs.

    python scripts/run_q1.py --base-jobs jobs/4b_s{0..7} --n-runs 90 --ceiling 20 \
        --K 8 --output-dir forks/q1 --report results/q1/report.md

Stub until the W1 pilot gates pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jobs", type=Path, nargs="+", required=True)
    ap.add_argument("--n-runs", type=int, default=90)
    ap.add_argument("--ceiling", type=int, default=20)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=Path("forks/q1"))
    ap.add_argument("--report", type=Path, default=Path("results/q1/report.md"))
    ap.parse_args()
    raise NotImplementedError("W3: eligibility -> probes -> oracle -> rows -> stats -> report")


if __name__ == "__main__":
    raise SystemExit(main())
