#!/usr/bin/env bash
# Start vLLM for MODEL on PORT (default 8000) and the GuardFork logging proxy on 8100 in front
# of it, wait for both health checks, send one logged smoke request, then leave both running.
#
#   TAG=k0_9b_seed0 USD_PER_HOUR=2.49 scripts/vm_serve.sh Qwen/Qwen3.5-9B [PORT]
#   REUSE_VLLM=1 TAG=<new_tag> scripts/vm_serve.sh Qwen/Qwen3.5-9B   # new proxy log only
#
# Sampling comes only from config/sampling.toml (injected by the proxy); vLLM is told not to
# load the model's generation_config.json defaults.
# Stop: kill "$(cat runs/serve/<tag>.proxy.pid)" "$(cat runs/serve/<tag>.vllm.pid)"
set -euo pipefail

MODEL="${1:?usage: scripts/vm_serve.sh MODEL [PORT]}"
PORT="${2:-8000}"
PROXY_PORT=8100
GF_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="${VLLM_VENV:-$HOME/vllm-env}"
TAG="${TAG:-$(basename "$MODEL" | tr -c 'A-Za-z0-9._\n-' '_')_$(date -u +%Y%m%dT%H%M%SZ)}"
REUSE_VLLM="${REUSE_VLLM:-0}"
# Bind to loopback: Harbor's agent runs on the host, and Lambda VMs have public IPs.
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
# VERIFY: fits the KV cache on the chosen GPU (vLLM fails fast at startup if it does not).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TP="${TP:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"
# Qwen3.5 defaults (https://recipes.vllm.ai/Qwen/Qwen3.5-9B).
# VERIFY: parsers for non-Qwen models (mistral / gpt-oss) before the cross-family K2 smoke.
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-2400}"
SMOKE="${SMOKE:-1}"

step() { printf '\n==> [%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
port_in_use() { ss -Hltn "sport = :$1" | grep -q .; }

cd "$GF_DIR"
mkdir -p runs/serve runs/proxy runs/cost
VLLM_LOG="runs/serve/$TAG.vllm.log"
PROXY_OUT="runs/serve/$TAG.proxy.out"
PROXY_LOG="runs/proxy/$TAG.jsonl"

started=()
cleanup_on_fail() {
  echo "startup failed; stopping what this script started: ${started[*]:-none}" >&2
  for pid in "${started[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup_on_fail ERR

wait_http() {  # wait_http URL PID LABEL
  local url="$1" pid="$2" label="$3" waited=0
  until curl -sf -o /dev/null "$url"; do
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "$label exited before becoming healthy" >&2; return 1
    fi
    if ((waited >= HEALTH_TIMEOUT_S)); then
      echo "$label not healthy after ${HEALTH_TIMEOUT_S}s" >&2; return 1
    fi
    sleep 5; waited=$((waited + 5))
    ((waited % 60 == 0)) && echo "  waiting for $label (${waited}s)"
  done
  echo "$label healthy: $url"
}

step "vLLM $MODEL on $VLLM_HOST:$PORT (tag $TAG)"
if [[ "$REUSE_VLLM" == "1" ]]; then
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/health" \
    || { echo "REUSE_VLLM=1 but nothing healthy on :$PORT" >&2; exit 1; }
  echo "reusing running vLLM on :$PORT"
else
  if port_in_use "$PORT"; then echo "port $PORT already in use" >&2; exit 1; fi
  vllm_args=(
    serve "$MODEL"
    --host "$VLLM_HOST" --port "$PORT"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --enable-prefix-caching
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_PARSER"
    --reasoning-parser "$REASONING_PARSER"
    # No generation_config.json defaults: https://docs.vllm.ai/en/stable/cli/serve/
    --generation-config vllm
  )
  # Qwen3.5 checkpoints are multimodal; text-only skips the vision encoder's memory.
  [[ "$LANGUAGE_MODEL_ONLY" == "1" ]] && vllm_args+=(--language-model-only)
  [[ -n "$MAX_NUM_SEQS" ]] && vllm_args+=(--max-num-seqs "$MAX_NUM_SEQS")
  printf '+ %s/bin/vllm %s\n' "$VLLM_VENV" "${vllm_args[*]}"
  nohup "$VLLM_VENV/bin/vllm" "${vllm_args[@]}" >"$VLLM_LOG" 2>&1 &
  vllm_pid=$!
  started+=("$vllm_pid")
  echo "$vllm_pid" >"runs/serve/$TAG.vllm.pid"
  echo "vLLM pid $vllm_pid, log $VLLM_LOG"
  wait_http "http://127.0.0.1:$PORT/health" "$vllm_pid" vLLM
fi

step "Proxy $PROXY_HOST:$PROXY_PORT -> :$PORT, log $PROXY_LOG"
if port_in_use "$PROXY_PORT"; then echo "port $PROXY_PORT already in use" >&2; false; fi
proxy_args=(
  run python -m guardfork.serve.proxy
  --upstream "http://127.0.0.1:$PORT" --port "$PROXY_PORT" --host "$PROXY_HOST"
  --backend vllm --sampling-config config/sampling.toml
  --log "$PROXY_LOG" --tag "$TAG"
)
printf '+ uv %s\n' "${proxy_args[*]}"
nohup uv "${proxy_args[@]}" >"$PROXY_OUT" 2>&1 &
proxy_pid=$!
started+=("$proxy_pid")
echo "$proxy_pid" >"runs/serve/$TAG.proxy.pid"
wait_http "http://127.0.0.1:$PROXY_PORT/health" "$proxy_pid" proxy
curl -sf "http://127.0.0.1:$PROXY_PORT/v1/models" | grep -qF "\"$MODEL\"" \
  || { echo "proxy /v1/models does not list $MODEL" >&2; false; }

if [[ "$SMOKE" == "1" ]]; then
  step "Smoke request through the proxy (checks token ids + sampling land in the log)"
  curl -sf "http://127.0.0.1:$PROXY_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"$MODEL\", \"max_tokens\": 256,
         \"messages\": [{\"role\": \"user\", \"content\": \"Reply with the word OK.\"}]}" \
    -o /dev/null
  python3 - "$PROXY_LOG" <<'PY'
import json, sys
rec = json.loads(open(sys.argv[1]).read().splitlines()[-1])
req = rec.get("request") or {}
problems = [k for k in ("temperature", "top_p", "top_k", "seed") if k not in req]
if rec.get("status") != 200:
    problems.append(f"status={rec.get('status')}")
if not rec.get("prompt_token_ids"):
    problems.append("no prompt_token_ids")
if not any(rec.get("completion_token_ids") or []):
    problems.append("no completion_token_ids")
print(f"smoke record: status={rec.get('status')} sampling={rec.get('sampling')} "
      f"seed={req.get('seed')} finish={rec.get('finish_reasons')}")
if problems:
    sys.exit(f"smoke check failed: {problems}")
PY
fi

trap - ERR
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" serve_start "$(hostname)" \
  "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" "$MODEL" "$TAG" \
  "${USD_PER_HOUR:-?}" >>runs/cost/vm_sessions.tsv

step "Ready"
echo "student: --student-model openai:$MODEL --student-base-url http://127.0.0.1:$PROXY_PORT/v1"
echo "proxy log: $PROXY_LOG"
echo "prefix cache: curl -s http://127.0.0.1:$PORT/metrics | grep -E 'prefix_cache_(hits|queries)'"
echo "stop: kill \$(cat runs/serve/$TAG.proxy.pid) \$(cat runs/serve/$TAG.vllm.pid 2>/dev/null)"
