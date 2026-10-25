# Weekend 0 runbook (Local Apple Silicon smoke test + Lambda GPU VM full W0; GCP fallback)

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

## Lambda GPU VM (K0, referee, training, final evals)

x86_64 Ubuntu only (not GH200, which is arm64). The commands below assume `~/guardfork` and
`~/harness-zero` sit side by side. All raw output stays under `runs/`, `jobs/`, `forks/`,
`results/` on the VM (gitignored; hook-blocked).

**L0. Setup (once per VM, idempotent; re-run after any failure).** From the Mac (the repo is
never pushed publicly):

```bash
rsync -az --exclude .venv --exclude runs --exclude jobs --exclude forks --exclude results \
  --exclude teacher --exclude __pycache__ \
  /Users/arnavarora/Desktop/Developer/guardfork/ ubuntu@<VM_IP>:~/guardfork/
ssh ubuntu@<VM_IP>
mkdir -p ~/guardfork/runs/cost && printf '%s\tvm_start\t%s\t%s\n' "$(date -u +%FT%TZ)" \
  "<instance type>" "<USD/h>" >> ~/guardfork/runs/cost/vm_sessions.tsv
bash ~/guardfork/scripts/vm_setup.sh
```

`vm_setup.sh` checks or installs Docker, compose v2, the NVIDIA container toolkit, and uv. It then
clones Harness-Zero at `a4a6223` and runs `uv sync`, builds `ahd-appworld-deps:v2` natively,
downloads the official AppWorld data, and builds `ahd-appworld-{client,world,verifier}:v2` on top
of the deps image. Finally it runs the Harbor oracle smoke on `07b42fd_1` (must score 1.0), installs
vLLM in `~/vllm-env`, and writes versions plus image IDs to `runs/vm/env_<utc>.txt`. Copy that file's
key lines into `NOTES.md`. Pin vLLM with `VLLM_VERSION=...` if the latest release breaks parity.

**L1. Serve + proxy (smoke tag).**

```bash
cd ~/guardfork
TAG=vm_smoke_9b USD_PER_HOUR=<rate> scripts/vm_serve.sh Qwen/Qwen3.5-9B
```

The script starts vLLM on `127.0.0.1:8000` with prefix caching, `qwen3_coder` tool parsing, and
`qwen3` reasoning parsing. It passes `--generation-config vllm` so the model's
`generation_config.json` cannot add sampling defaults. It then starts the proxy on `127.0.0.1:8100`
with `config/sampling.toml`, waits for both `/health` endpoints, and sends one smoke request. The
smoke request must log prompt and completion token ids, the sampling fields, and a seed. Knobs:
`MAX_MODEL_LEN` (default 131072), `GPU_MEM_UTIL`, `MAX_NUM_SEQS`, `TP`. Parity note: the proxy
caps completions at 4096 tokens only for `--backend mlx`, so vLLM turns may run longer than Mac
turns did.

**L2. One-task smoke (dev task, pipeline check only).**

```bash
uv run python scripts/run_appworld_rollout.py --task 07b42fd_1 --attempts 1 --concurrency 1 \
  --student-max-turns 40 --student-timeout 600 \
  --student-model openai:Qwen/Qwen3.5-9B --job-name vm-9b-smoke-07b42fd_1
python3 -c 'import glob,json; [print(f, json.load(open(f))["verifier_result"]) for f in glob.glob("jobs/vm-9b-smoke-07b42fd_1/*/result.json")]'
wc -l runs/proxy/vm_smoke_9b.jsonl
```

Pass = the trial graded (reward 0 or 1, no exception) and every turn appears in the proxy log.

**L3. Template parity** (on the smoke log; use `--n` up to the number of logged requests):

```bash
uv run --with tinker-cookbook python scripts/check_template_parity.py \
  --log runs/proxy/vm_smoke_9b.jsonl --model Qwen/Qwen3.5-9B --n 20
```

Any DIFF: stop. Fix vLLM's `--chat-template` before K0.

**L4. K0: 9B B0 on test_normal (168 episodes).** test_normal is used only for K0 and final evals.
Restart only the proxy so K0 gets a clean log:

```bash
kill "$(cat runs/serve/vm_smoke_9b.proxy.pid)"
REUSE_VLLM=1 TAG=k0_9b_seed0 scripts/vm_serve.sh Qwen/Qwen3.5-9B
uv run python scripts/run_appworld_rollout.py \
  --tasks-file ../harness-zero/data/appworld-harbor/split_test168.txt --attempts 1 \
  --concurrency <C> --student-max-turns 40 --student-timeout 600 \
  --student-model openai:Qwen/Qwen3.5-9B --job-name k0_9b_seed0
uv run python scripts/measure_w0.py --title "K0 9B B0 test_normal (Lambda vLLM bf16)" --k0 \
  --model Qwen/Qwen3.5-9B --jobs jobs/k0_9b_seed0 --proxy-log runs/proxy/k0_9b_seed0.jsonl
```

SGC outside [18, 36]: stop and fix the setup. Then run the train147 seeds for K1 the same way
(steps 3 to 4 below), each under its own `REUSE_VLLM=1 TAG=...` proxy log.

Concurrency `<C>` (rough; the GPU is rarely the bottleneck for a 9B model):
- Each Harbor trial runs its own compose project (client + world, then verifier). Budget about
  1 vCPU and 3 GB RAM per concurrent trial. Start with `C = min(nproc - 4, RAM_GiB / 3, 32)`.
- 1x A100/H100 80 GB with 9B bf16: start at 16 and raise to 24 to 32 if the checks below stay clean.
  1x A10 24 GB: 9B bf16 leaves little KV cache, so use `MAX_MODEL_LEN=32768` and C ≤ 8, or prefer
  an 80 GB card.
- Check during the first 10 minutes. `curl -s :8000/metrics | grep -E 'num_requests_waiting|cache_usage_perc'`
  should show waiting ≈ 0 and KV usage < 0.9, and `grep -c preempt runs/serve/<tag>.vllm.log`
  should not grow. `docker stats --no-stream` should show no host swap. Otherwise lower C.
  Record C in `NOTES.md`.

**L5. Referee.** First run a pricing run, which is part of Weekend 0 (it feeds the K3 cost projection):

```bash
uv run python scripts/run_referee.py --base-jobs jobs/<9b train147 seed jobs> \
  --n-runs 2 --K 4 --concurrency <C/2> --student-model openai:Qwen/Qwen3.5-9B
```

The full Q1 referee (`--n-runs 20 --K 8`, then scale-up) is a Weekend 1+ job. Launch it only after
L6 shows every Weekend 0 check passing (AGENTS.md rule 7). Referee base jobs must be train147 runs,
never test_normal. Dev-set tasks are excluded automatically.

**L6. Measure + decide.** First set `[cost].gpu_usd_per_hour` in `config/kill.toml` to the Lambda
rate and `episodes_per_gpu_concurrently` to C, with a dated `NOTES.md` entry. Then:

```bash
uv run python scripts/measure_w0.py --title "W0 Lambda 9B" --model Qwen/Qwen3.5-9B \
  --jobs jobs/k0_9b_seed0 jobs/<train147 seed jobs> --proxy-log runs/proxy/<tag>.jsonl \
  --notes NOTES.md
```

**Cost/time logging (`runs/cost/`, gitignored).** `vm_serve.sh` appends a `serve_start` row to
`runs/cost/vm_sessions.tsv` (UTC time, host, GPU, model, tag, `USD_PER_HOUR`). Append a
`vm_start` row at boot (L0) and a `vm_stop` row before terminating the instance. Billing runs
until the instance is terminated, not when jobs finish. Record wall-clock start and end per Harbor job
next to it. `run_referee.py` writes env-seconds to `runs/cost/referee.jsonl`. Before terminating,
pull results to the Mac. rsync `jobs/ runs/ results/ forks/` from the VM into the Mac's gitignored
directories; never commit them.

```bash
printf '%s\tvm_stop\t%s\n' "$(date -u +%FT%TZ)" "<instance type>" >> runs/cost/vm_sessions.tsv
```

## 0. VM setup (GCP fallback: L4 spot, e.g. g2-standard-8)

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
