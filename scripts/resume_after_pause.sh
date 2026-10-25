#!/usr/bin/env bash
# Resume after a pause: top up the dev re-check to 4 graded attempts per task (new jobs
# w0-dev-9b-t06-r1, -r2, ...; interrupted trial dirs in w0-dev-9b-t06 stay ungraded and are
# ignored), then hand over to queue_after_dev.sh (referee smoke, then +2 seeds).
# Needs mlx_lm.server on :8000 and the proxy on :8100 (config/sampling.toml).
#   scripts/resume_after_pause.sh
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="openai:mlx-community/Qwen3.5-9B-4bit"

curl -sf -o /dev/null http://127.0.0.1:8100/v1/models || { echo "proxy not up on :8100"; exit 1; }

round=1
while true; do
  uv run python - <<'PY' > runs/w0/dev_missing.txt
from collections import Counter
from pathlib import Path
from guardfork.telemetry.trials import iter_trials
c = Counter()
for job in sorted(Path("jobs").glob("w0-dev-9b-t06*")):
    c.update(r.task_id for r in iter_trials(job) if r.reward is not None)
for t in Path("runs/w0/dev_tasks_21.txt").read_text().split():
    if c.get(t, 0) < 4:
        print(t)
PY
  [ -s runs/w0/dev_missing.txt ] || break
  echo "$(date) round ${round}: $(wc -l < runs/w0/dev_missing.txt) tasks still short of 4 attempts"
  uv run python scripts/run_appworld_rollout.py --tasks-file runs/w0/dev_missing.txt \
    --attempts 1 --concurrency 3 --student-max-turns 40 --student-timeout 600 \
    --student-model "$MODEL" --job-name "w0-dev-9b-t06-r${round}"
  round=$((round + 1))
  [ "$round" -le 4 ] || { echo "giving up after 4 rounds"; break; }
done
echo "$(date) dev re-check complete"
exec scripts/queue_after_dev.sh
