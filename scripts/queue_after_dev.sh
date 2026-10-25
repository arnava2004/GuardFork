#!/usr/bin/env bash
# After the sampled dev re-check: referee smoke on 2 fresh failures at K=4 (prices the GPU
# run), then +2 seeds on all non-dev train147 tasks toward K1's 8 samples.
#   scripts/queue_after_dev.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="openai:mlx-community/Qwen3.5-9B-4bit"

while pgrep -f "job-name w0-dev-9b-t06" >/dev/null; do sleep 60; done
echo "$(date) dev re-check finished"

uv run python scripts/run_referee.py --base-jobs jobs/w0-screen-9b-t06-s12 \
  --trial c901732_3__8KZ792a --trial 60d0b5b_1__pHRgiha \
  --band 0.5 0.5 --min-samples 2 --K 4 --concurrency 3 --n-runs 2 \
  --student-model "$MODEL" \
  --forks-dir forks/referee_smoke --results-dir results/referee_smoke
echo "$(date) referee smoke finished"

uv run python scripts/run_appworld_rollout.py --tasks-file runs/w0/screen_nondev_tasks.txt \
  --attempts 2 --concurrency 3 --student-max-turns 40 --student-timeout 600 \
  --student-model "$MODEL" --job-name w0-screen-9b-t06-s34
echo "$(date) +2 seed screen finished"
