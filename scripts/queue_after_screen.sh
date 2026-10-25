#!/usr/bin/env bash
# Wait for a running screen job, restart the proxy from config/sampling.toml, then run the
# sampled dev-set re-check (21 dev tasks x 4 seeds) used to re-validate guards-v1.
#   scripts/queue_after_screen.sh w0-screen-9b-t06-s12
set -euo pipefail
cd "$(dirname "$0")/.."
SCREEN_JOB="$1"
MODEL="openai:mlx-community/Qwen3.5-9B-4bit"

while pgrep -f "job-name ${SCREEN_JOB}" >/dev/null; do sleep 60; done
echo "$(date) screen ${SCREEN_JOB} finished"

pkill -f "guardfork.serve.proxy" || true
sleep 2
nohup uv run python -m guardfork.serve.proxy --upstream http://127.0.0.1:8000 --port 8100 \
  --host 0.0.0.0 --backend mlx --sampling-config config/sampling.toml \
  --log runs/proxy/mlx_9b_t06.jsonl --tag mlx_9b_t06 > runs/proxy/proxy_t06.out 2>&1 &
sleep 5
curl -sf -o /dev/null http://127.0.0.1:8100/v1/models
echo "$(date) proxy restarted with config/sampling.toml"

grep -o '"[0-9a-f]\{7\}_[0-9]"' config/dev_set.toml | tr -d '"' | grep -v '^07b42fd_1$' \
  > runs/w0/dev_tasks_21.txt
uv run python scripts/run_appworld_rollout.py --tasks-file runs/w0/dev_tasks_21.txt \
  --attempts 4 --concurrency 3 --student-max-turns 40 --student-timeout 600 \
  --student-model "$MODEL" --job-name w0-dev-9b-t06
echo "$(date) dev re-check finished"
