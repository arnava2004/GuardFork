"""Weekend 1 pilot: 25 localization runs (K=8) + 8 test-retest runs, then W1-a..e verdicts.

    python scripts/run_pilot.py --base-jobs jobs/4b_s{0..7} --n-runs 25 --retest 8 \
        --K 8 --output-dir forks/pilot --notes NOTES.md

Stub until Weekend 0 passes config/kill.toml (AGENTS.md rule 7).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-jobs", type=Path, nargs="+", required=True)
    ap.add_argument("--n-runs", type=int, default=25)
    ap.add_argument("--retest", type=int, default=8)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--output-dir", type=Path, default=Path("forks/pilot"))
    ap.add_argument("--notes", type=Path, default=None)
    ap.parse_args()
    raise NotImplementedError("W1: select eligible runs -> probe plan -> launch -> oracle -> gates")


if __name__ == "__main__":
    raise SystemExit(main())
