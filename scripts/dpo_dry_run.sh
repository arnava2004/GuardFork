#!/usr/bin/env bash
# DPO plumbing dry run on the Mac (not a result): export pairs from the fork smoke job ->
# a few mlx-lm-lora DPO steps on Qwen3.5-9B-4bit -> mlx_lm.server with the adapter on a
# spare port -> one AppWorld episode through a second proxy.
#
# Run only after the AppWorld screen has finished (needs Docker + the Apple GPU):
#   scripts/dpo_dry_run.sh            # smoke pair from the greedy-era fork (zero DPO signal)
#   REFORK=1 scripts/dpo_dry_run.sh   # first re-fork t6 with sampling (K=4) for real pairs;
#                                     # needs the student server + proxy on :8000/:8100
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="mlx-community/Qwen3.5-9B-4bit"
OUT="runs/dpo/smoke"
PLAN="forks/fork_smoke/plan_t6.json"
FORK_JOB="forks/fork_smoke/fork-smoke-t6"
ADAPTER_PORT=8301
PROXY_PORT=8401
TASK=$(python3 -c "import json;print(json.load(open('$PLAN'))['task_id'])")
mkdir -p "$OUT" runs/proxy

if pgrep -f "harbor run" >/dev/null; then
  echo "a Harbor job is still running; wait for it to finish" >&2
  exit 1
fi

# 1 (optional). Sampled re-fork at t6 so continuations can differ from the source action.
EXTRA=()
if [ "${REFORK:-0}" = "1" ]; then
  uv run python scripts/run_appworld_rollout.py --replay-plan "$PLAN" --attempts 4 \
    --concurrency 2 --student-model "openai:$MODEL" --student-base-url http://127.0.0.1:8100/v1 \
    --job-name fork-smoke-t6-sampled --output-dir forks/fork_smoke
  EXTRA=(--plan "$PLAN" --fork-job forks/fork_smoke/fork-smoke-t6-sampled)
fi

# 2. Export (neutral + mlx-lm-lora rows). --smoke keeps the passing-source / identical pair.
uv run python scripts/export_dpo_pairs.py --plan "$PLAN" --fork-job "$FORK_JOB" \
  ${EXTRA[@]+"${EXTRA[@]}"} --max-per-fork 4 --smoke --out "$OUT"

# 3. Token parity: mlx-lm-lora's DPODataset ids == real chat-template ids.
HF_HUB_OFFLINE=1 uv run --no-project --with transformers --with jinja2 \
  python scripts/check_dpo_render.py --data "$OUT" --model "$MODEL"

# 4. A few LoRA DPO steps (sequences are ~8.5k tokens; stop other mlx servers if memory is
#    tight). VERIFY: flags from mlx_lm_lora/train.py on main (2026-10-25).
uvx --from mlx-lm-lora mlx_lm_lora.train --model "$MODEL" --train --train-mode dpo \
  --train-type lora --data "$OUT/mlx" --iters 3 --batch-size 1 --num-layers 8 \
  --learning-rate 1e-6 --beta 0.1 --max-seq-length 10240 --grad-checkpoint \
  --steps-per-report 1 --steps-per-eval 1000 --val-batches 1 --save-every 3 \
  --adapter-path "$OUT/adapters"

# 5. Serve the adapter on a spare port, behind a second proxy (same sampling config).
#    VERIFY: mlx-lm-lora adapters load in mlx_lm.server --adapter-path.
nohup uvx --from mlx-lm==0.32.2 mlx_lm.server --model "$MODEL" --adapter-path "$OUT/adapters" \
  --port "$ADAPTER_PORT" > "$OUT/server.out" 2>&1 &
SERVER_PID=$!
nohup uv run python -m guardfork.serve.proxy --upstream "http://127.0.0.1:$ADAPTER_PORT" \
  --port "$PROXY_PORT" --host 0.0.0.0 --backend mlx --sampling-config config/sampling.toml \
  --log runs/proxy/dpo_smoke.jsonl --tag dpo_smoke > "$OUT/proxy.out" 2>&1 &
PROXY_PID=$!
trap 'kill $PROXY_PID $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf -o /dev/null "http://127.0.0.1:$PROXY_PORT/v1/models" && break
  sleep 5
done
curl -sf "http://127.0.0.1:$PROXY_PORT/v1/models"

# 6. One episode with the adapter (dev task; plumbing only).
uv run python scripts/run_appworld_rollout.py --task "$TASK" --attempts 1 --concurrency 1 \
  --student-model "openai:$MODEL" --student-base-url "http://127.0.0.1:$PROXY_PORT/v1" \
  --student-max-turns 40 --student-timeout 600 --job-name dpo-smoke-episode
echo "$(date) dry run finished: $OUT, jobs/dpo-smoke-episode, runs/proxy/dpo_smoke.jsonl"
