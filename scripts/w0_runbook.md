# Weekend 0 runbook (Local Apple Silicon smoke test + GCP full W0)

Goal: measure, then decide per `config/kill.toml`. Everything raw goes under `runs/` / `jobs/` (gitignored, hook-blocked).

Confirmed against Harness-Zero `a4a6223`:
- passthrough (bare h): `--teacher-provider passthrough --teacher-model none --teacher-reasoning-effort none`
- test split: `data/appworld-harbor/split_test168.txt`
- student base URL: `--student-base-url http://<host>:8100/v1`
- AppWorld agent/env: `--agent envs.appworld.agent:AppWorldMinisweAgent --env envs.appworld.harbor_environment:AppWorldDockerEnvironment`
- h*: `--student-harness-dir harness_bank/appworld_student`

## 0A. Local Mac Smoke Test (Apple Silicon GPU via `mlx-lm` + Docker Desktop, 1 task)

Run 1 AppWorld task end-to-end on the local Mac (`M4 Pro 48GB`) to verify Docker images, proxy logging, Harbor verifier grading, guards, and the trajectory viewer before renting a GPU.

```bash
# 1. Start mlx_lm.server on port 8000 (4-bit Qwen3.5-4B on Apple GPU)
uvx --from mlx-lm mlx_lm.server --model mlx-community/Qwen3.5-4B-4bit --port 8000

# 2. Start GuardFork logging proxy on port 8100 (--backend mlx)
uv run python -m guardfork.serve.proxy --upstream http://127.0.0.1:8000 --port 8100 \
  --host 0.0.0.0 --backend mlx --log runs/proxy/local_smoke.jsonl --tag local_smoke

# 3. Run 1 train task through Harness-Zero + AppWorldDockerEnvironment
head -n 1 ../harness-zero/data/appworld-harbor/split_train147.txt > /tmp/smoke_task.txt
cd ../harness-zero && uv run harness-zero run-rollout \
  --dataset data/appworld-harbor/tasks \
  --components harness_bank/appworld \
  --tasks-file /tmp/smoke_task.txt --attempts 1 --concurrency 1 \
  --student-model openai:mlx-community/Qwen3.5-4B-4bit \
  --student-base-url http://127.0.0.1:8100/v1 \
  --student-reasoning-effort high \
  --teacher-provider passthrough --teacher-model none --teacher-reasoning-effort none \
  --student-max-turns 15 --job-name local_smoke --output-dir ../guardfork/jobs
```

## 0. VM setup (L4 spot, e.g. g2-standard-8)

```bash
git clone https://github.com/metaevo-ai/harness-zero.git && cd harness-zero
git rev-parse HEAD                      # record in NOTES.md
uv sync
python envs/appworld/build.py           # needs official AppWorld data
python envs/appworld/validate.py --image-tag v2
docker network inspect <compose-net>    # confirm internal: true + inhibit_ipv4
```

Clone guardfork next to it; install hooks: `git config core.hooksPath hooks`.

## 1. Serve + proxy

```bash
vllm serve Qwen/Qwen3.5-9B --port 8000 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
uv run python -m guardfork.serve.proxy --upstream http://127.0.0.1:8000 --port 8100 \
  --host 0.0.0.0 --backend vllm --log runs/proxy/k0.jsonl --tag k0
```

Point the student at the proxy (`--student-model openai:Qwen/Qwen3.5-9B` with
`OPENAI_BASE_URL=http://<host-ip>:8100/v1` in Harbor's `.env`). Containers must be able to
reach the proxy; if the compose network is `internal: true`, run the proxy on the host
bridge IP and confirm one request lands in the log before launching the full job.
Record vLLM prefix-cache hit rate from `/metrics` at the end of each job.

## 2. K0 first: 9B B0 on test_normal (168 episodes)

```bash
harness-zero run-rollout --dataset data/appworld-harbor/tasks \
  --agent envs.appworld.agent:AppWorldMinisweAgent \
  --env envs.appworld.harbor_environment:AppWorldDockerEnvironment \
  --tasks-file data/appworld-harbor/split_test168.txt --attempts 1 --concurrency 8 \
  --student-model openai:Qwen/Qwen3.5-9B --student-base-url http://HOST:8100/v1 \
  --student-reasoning-effort high --teacher-provider passthrough \
  --teacher-model none --teacher-reasoning-effort none \
  --job-name k0_9b_seed0 --output-dir jobs
uv run python scripts/measure_w0.py --title "K0 9B B0 test_normal" --k0 \
  --model Qwen/Qwen3.5-9B --jobs jobs/k0_9b_seed0 --proxy-log runs/proxy/k0.jsonl \
  --notes NOTES.md
```

SGC outside [18, 36] (paper: 26.8): stop, fix setup. Do not continue.

## 3. 4B × 4 on train147, then +4 on 1–3/4 tasks

Restart vLLM with `Qwen/Qwen3.5-4B`, new proxy log `runs/proxy/4b.jsonl`. Four jobs
`4b_s0..4b_s3` over `split_train147.txt`, then:

```bash
uv run python scripts/measure_w0.py --title "4B x4 train147" --model Qwen/Qwen3.5-4B \
  --jobs jobs/4b_s{0,1,2,3} --proxy-log runs/proxy/4b.jsonl \
  --second-stage-out runs/w0/4b_second_stage.txt --notes NOTES.md
```

Run `4b_s4..4b_s7` with `--tasks-file runs/w0/4b_second_stage.txt`, then re-run
`measure_w0.py` over all eight jobs for the K1 verdict.

## 4. 9B × 4 on train147 — only if 4B fails K1

## 5. 4B under h* × 2 seeds

Same as step 3 with `--student-harness-dir harness_bank/appworld_student`, jobs
`4b_hstar_s0..1`. Compare TGC to `4b_s0..1`; Gap Closure (P4) needs h* − h ≥ 10 TGC.

## 6. Template parity (20 requests)

```bash
uv run --with tinker-cookbook python scripts/check_template_parity.py \
  --log runs/proxy/4b.jsonl --model Qwen/Qwen3.5-4B --n 20
```

Any DIFF: fix vLLM `--chat-template` to match `Qwen3_5Renderer(strip=False)` and re-run.

## 7. Decide

Fill the Weekend 0 section of `NOTES.md` with the verdict table and a one-line keep/kill.
