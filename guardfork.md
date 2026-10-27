# GuardFork: Plan

Owner: Arnav. Drafted 2026-10-24 (revised after red-team audits 1–4 and the final go/no-go audit).
The "Rules for agents" section is mirrored in `AGENTS.md`. Kill thresholds live in `config/kill.toml`.

## The Question

1. **Q1 — Localization (Weekends 1–4, standalone publishable):** Do an agent harness's own free, deterministic rule checks ("guards" G1–G5, computed offline over trajectory prefixes) localize the non-terminal turn where an agent rollout derails on AppWorld — measured against a test-retest-bounded Monte Carlo continuation oracle, analytic random-turn, and last-turn baselines — and what is their true rollout cost per usable preference pair?
2. **Q2 — Offline post-training (Weekends 5–7, conditional on W1/W2 gates):** Using a single shared offline rollout pool, does turn-level DPO at guard-forked turns (`GF-DPO`) outperform whole-trajectory SFT (`RFT`), whole-trajectory DPO (`T-DPO`), offline guard-penalized trajectory DPO (`G-REW`), and random-turn fork DPO (`R-FORK-DPO`) — reducing untargeted/held-out guard firings and narrowing the gap between bare mini-SWE-agent (h) and Harness-Zero's evolved student harness (h*)?

## Why It's Open (Oct 2026)

- **Prior step-level localization pays for extra rollouts, critics, or LLM judges:** GPT-4o judges (SDPO, 2501.01821), trained process reward models plus expert rewrites (CSO, 2602.03412; Salesforce, 2609.09134; Scale OEC, 2512.14895), or Monte Carlo rollout trees (ELPO, 2602.09598; VinePPO).
- **Prior deterministic signals are not evaluated as error-avoidance fork localizers:** Fission-GRPO (2601.15625) branches *after* an execution error with injected error feedback to train *recovery*; DepGPO (2610.03634) traces command dependencies from the verifier's read set; 2610.11655 uses first-firing signals (F1–F4) to label whole trajectories for SFT and explicitly declines RL ("the route we do not take"); HarnessRL (2610.26426) applies trajectory/turn penalties without same-state counterfactual forking.
- **Defensible claim:** To our knowledge, no prior work evaluates verifier-free deterministic harness guards as fork-point localizers for same-state error-avoidance preference pairs against a reliability-bounded Monte Carlo value oracle, or isolates guard-as-localizer (`GF-DPO`) against guard-as-reward (`G-REW`) and random-turn forking (`R-FORK-DPO`) under a matched offline rollout budget.

## Predictions

Q1 and Section 5 predictions (P1–P8) live in `PREREGISTRATION.md`, which is binding. The Q2 training predictions below (Q2-a to Q2-d) are filled before Weekend 5 training starts.

2. **Q2-a — Pair yield & cost per usable pair:** Guard forking yields __ usable AST-distinct preference pairs per failed run at 1×K continuations per failure, at __× lower rollout-token and env-second cost per usable pair than MC probe localization (~5×K per failure) and __× higher usable-pair yield than random-turn forking (1×K per failure).
3. **Q2-b — Offline post-training on test_normal (5 eval seeds):** Across 168 test_normal tasks (56 scenarios) × 5 eval seeds (primary: TGC, 3-contrast Holm-adjusted 80% MDE ≈ 9.5–10.0 pts; secondary: SGC, MDE ≈ 14 pts), `GF-DPO` beats `T-DPO` by __, `R-FORK-DPO` by __, and `G-REW` by __ TGC pts. Sub-MDE contrasts are reported as 95% CIs / equivalence bounds, never as wins.
4. **Q2-c — Harness internalization:** Under bare h, `GF-DPO` reduces non-terminal guard firings per episode by __% (including held-out G3, excluded from fork selection). If the base h* − h gap on train147 is ≥ 10 TGC pts (95% CI excluding 0), report bootstrapped Gap Closure; otherwise report raw ΔTGC under h and h*.
5. **Q2-d — Process vs. content failures:** `GF-DPO` vs `R-FORK-DPO` at matched budget: gains concentrate on trajectories that failed with a non-terminal guard rather than unguarded content failures.

## Setup, Splits & Guard Definitions

- **Benchmark:** AppWorld in Harness-Zero's container setup (`AppWorldMinisweAgent` + `AppWorldDockerEnvironment`; 23 initial messages = 10 demo assistant turns before `[APPWORLD_CURRENT_TASK]`; one `execute` bash tool).
- **Splits** (`scenario = task_id.rsplit("_", 1)[0]`, 3 tasks/scenario):
  - Q1: train147's 49 scenarios split into **17 design / 32 evaluation** at the scenario level (`config/dev_set.toml`, see v2 Protocol below; 15/34 before amendment 1).
  - Q2 training/checkpoint selection: `guardfork/splits.py` (train / dev_heldout by salted hash); reconcile with the Q1 split before Weekend 5 so no evaluation scenario is used for checkpoint selection.
  - `test_normal`: 56 scenarios (168 tasks), locked in the loader unless `allow_final=True` (`--final`); K0 and final Q2 evals only.
- **Models (open-weight only):** see Model Lineup below; `Qwen/Qwen3.5-9B` is the core model. Serving/training template parity: `Qwen3_5Renderer(strip_thinking_from_history=False)` verified against vLLM `prompt_token_ids` on the GPU VM before any training data is built.
- **Offline guards** (`guardfork/guards/rules.py`, evaluated on `assistant_index >= 10` only):
  - G1 `INVALID_ACTION` (evaluated first): `raw_original.invalid_tool_calls` non-empty, >1 tool call, tool name ≠ `execute`, empty command, empty message, visible `content` with code fences / `appworld exec`, or `reasoning_content` with structural tool markup (`<tool_call>`, `<function=`).
  - G2 `LOOP`: identical/near-identical AST API signature + outcome ≥ 3 times, or A/B alternation over 6 turns.
  - G3 `FAILED_PRECONDITION`: HTTP 401/403/404/422 API errors, or entity IDs never observed in prior tool outputs (held-out guard in the Q2 ablation).
  - G4 `CONSTRAINT_VIOLATION`: mutating an app outside the task's allowed apps, or unrequested destructive write/delete.
  - G5 `UNVERIFIED_COMPLETION` (only if not G1): no tool call while completion status is not True, or `complete_task` without a post-write verification read.
  - `MAX_TURNS` / `CONTEXT_OVERFLOW` / `REPLAY_DIVERGED` are `TrajectoryStatus` values, not localizers.

## v2 Protocol (binding; details in `PREREGISTRATION.md`)

- **Split:** scenario-level design (17 scenarios, incl. all 8 debugged ones and the 2 referee-smoke ones) vs evaluation (32 untouched scenarios). Headline Q1 uses evaluation only.
- **Single bf16 vLLM stack** for the Q1 failure pool and continuations, after a GPU concurrency pilot (C in {8, 16, 32}). Mac 4-bit runs are design/pilot only.
- **s1 screen** at K = 8; hopeless = 0/8 confirmed 0/16; everything ≥ 1/16 is eligible.
- **Candidate-augmented probe grid:** thinned grid plus mandatory windows around each guard's first firing, the open-weight judge pick and the original run's entropy peak; K = 16 refinement around the candidate change-point; deterministic snapping; 10–15 fully dense runs to bound snapping bias.
- **Primary oracle:** change-point first-irrecoverable turn at τ = 0.5·V̂(s1) with a 90% bootstrap interval; τ in {0.3, 0.7} and largest-drop as sensitivity. 8/8 split-half on K = 16 states, Beta bootstrap elsewhere.
- **Localizer ladder** from the LVRL-style error-marker floor to bisection and `guard_bayesect`; LLM judges (`judge_open` primary, `judge_claude` eval-only) return turn distributions used as search priors.
- **Section 5:** step credit scored against the learner's own Monte Carlo values, anchored on QVal (2606.32034) with two stated differences (on-policy values; harness-rule family); credit pilot judged primarily by **prefix drift** (`train/prefix_drift.py`), test_normal TGC secondary. Silent harm and in-flight abort are one paragraph each.

## Compute Plan

- **Mac (M4 Pro 48 GB, `mlx_lm.server`, 4-bit, arm64 AppWorld images):** pipeline checks, train147 pass-band screens, dev-set work, human failure labels, the LLM-judge baseline, the DPO plumbing dry run, and a 2-failure referee smoke test at K=4 to price the GPU run. Not used for K0 or any headline number.
- **Lambda GPU VMs (vLLM, full-precision weights, amd64 images rebuilt on the VM):** K0 anchor (9B B0 on test_normal), the Q1 referee at scale (~6000 episodes, ~130 h on the Mac), training arms, and 5-seed test_normal evals. GCP is the fallback. Same `config/sampling.toml` through the same proxy on both machines; one serving stack per machine.

## Model Lineup & Phased Benchmark Roadmap (Budget-Ordered)

### Phase 1: AppWorld Parameter-Size Ladder (Now $\to$ Lambda GPU Cluster)

We separate **Q1 (Localization & J-Lens)** from **Q2 (Offline Post-Training)** by what each question requires:

| Tier | Models | Scope | Where It Runs |
|---|---|---|---|
| **Q1 Scaling Ladder (No Training)** | `Qwen3.5-2B` $\to$ `4B` $\to$ `9B` $\to$ `27B / 35B-A3B` $\to$ `235B+` (plus `test_challenge` `split_challenge417.txt` if large models saturate `train147`) | Base rollouts + `G1–G5` guards + LLM-judge baseline + MC referee subset (`t*`) to plot **guard localization accuracy & failure-mode shifts across model scale** | Mac (`2B/4B/9B` 4-bit screens) $\to$ Lambda `vLLM` (`bf16` up to `35B-A3B`) $\to$ hosted open-weight API only for the `235B+` top-of-curve point |
| **Q2 Core Trained Models (All 5 Arms)** | 1. **`Qwen/Qwen3.5-9B` (`bf16`)** — primary headline (matches Harness-Zero's base model)<br>2. **Cross-family control** (`mistral-small-24b-instruct-2501` or `openai/gpt-oss-20b`, chosen by 10-task `K2` parse test)<br>3. **`Qwen/Qwen3.5-4B` (`bf16`)** (weak-model anchor) and/or **`Qwen3.5-35B-A3B`** (MoE, 3B active params) as budget allows | Full offline post-training comparison (`B0`, `RFT`, `T-DPO`, `G-REW`, `R-FORK-DPO`, `GF-DPO`) + 5-seed `test_normal` eval | Mac (`mlx-lm-lora` plumbing dry run on 4B/9B) $\to$ Lambda GPU (`bf16` LoRA; Tinker optional for 30B+ MoE) |
| **J-Lens Stretch (Offline Readout)** | Public-lens models (`qwen3.6-27b`, `gpt-oss-20b`, `mistral-small-24b`; `gemma-4` pending outside-work check) | 1 offline forward pass over saved token IDs on that model's own AppWorld rollouts + referee labels (`t*`) | Mac (10-task K2 smoke + 2-failure lens test) $\to$ Lambda GPU |

### Phase 2: Second-Benchmark Transfer Validation (After AppWorld Q1 + Q2 Lock)

Once the AppWorld parameter-size ladder (`Q1`) and post-training comparison (`Q2`) are locked, validate transfer on a **second benchmark** so the paper establishes both **scale generality** and **domain generality**:
- **Primary Benchmark #2: `SpreadsheetBench Verified`** (already shipped in Harbor format at `harness-zero/data/spreadsheetbench-verified/`, 300 train / 100 test, single-container bash + Python `.xlsx` manipulation).
- **Backup / Harder Agentic Benchmark #2:** `tau2-bench` or `Terminal-Bench` if a multi-turn tool-use environment is needed for 30B+ models.
- **Scope on Benchmark #2:** Run `Q1` guard localization + `Q2` winning arm (`GF-DPO` vs. `T-DPO` and `R-FORK-DPO`) on the primary model (`Qwen3.5-9B`) without re-running the full 5-arm ablation matrix.

## Known Deviations from Harness-Zero (methods section)

- **Sampling:** Harness-Zero's repo does not specify student sampling, and its client sends none. We fix sampling in `config/sampling.toml` (Qwen3.5 thinking-mode coding: temperature 0.6, top_p 0.95, top_k 20), injected by the proxy with a logged per-request seed. Our baseline therefore cannot be guaranteed to match their 26.8 SGC; we report the difference rather than tune toward it (or confirm their settings with the authors).
- **Turn cap:** 40, equal to Harness-Zero's default `student_max_turns`; the paper's AppWorld value is to be confirmed.
- **Client timeout:** 600 s per model request (their default is 120 s, which kills long 9B reasoning turns locally).
- **Local runs** use 4-bit MLX weights; all reported numbers come from full-precision vLLM runs.

## Harbor-Native Forking & Oracle Definition

- **Replay-from-zero forks in Harbor (`ForkReplayAppWorldAgent`):** each continuation is an isolated Harbor trial (`docker compose -p fork_<uuid>`, `PYTHONHASHSEED=0`, `down -v` in `finally`). `initialize` runs `appworld task` + `PROFILE_COMMAND`, checks sha256 against `appworld_initial_context.json`, replays commands 1..t−1 via `backend.aexecute` asserting exact observation equality, seeds the logged message prefix, and runs the inner graph with `max_turns = 40 − (t − 1)`. Graded only by Harbor's official verifier. Turns after `session_reset=True` or any `agent "<task>"` subagent call are non-forkable.
- **Monte Carlo oracle:** primary = change-point first-irrecoverable turn (`stats/bayes_prior.py`, `PREREGISTRATION.md`). The v1 adjacent-pair rule below (`guardfork/stats/oracle.py`) is kept as a sensitivity analysis only:
  - t* = min{t ∈ 1..T : V̂(s_{t+1}) ≤ 0.5·Ṽ(s₁) ∧ V̂(s_t) − V̂(s_{t+1}) ≥ 2/K}, with V̂(s_{T+1}) ≡ R(τ) = 0; undefined if no adjacent probed pair qualifies.
  - Probing follows the v2 candidate-augmented grid and s1 screen, not the v1 {0, g−3, g−1, g, g+1} schedule.
- **Guard freeze & design/evaluation separation:** guards are frozen as `guards-v1` (`config/guards_freeze.toml`, enforced by `tests/test_guards_freeze.py`). Everything they were tuned or debugged on is the design split (`config/dev_set.toml`) and is excluded from Q1. Later rule changes are new versions reported separately.
- **Guard-type robustness (pre-registered):** G3 (API auth/precondition errors) dominates first firings on dev runs. Report Hit(w) per guard type, with G3 excluded (`no_g3`), and with an "unrecovered G3" variant (a G3 firing counts only if the next 2 turns neither read API docs nor log in), so the result is not just one AppWorld quirk.
- **Inference:** d_i(w) = 𝟙(|g_i − t*_i| ≤ w) − |{t ∈ [1, T_i − 1] : |t − t*_i| ≤ w}| / (T_i − 1); test E[d(w)] > 0 with a scenario-clustered bootstrap (McNemar secondary). Non-terminal firings (g < T) are the headline; terminal firings get their own column. Any guard firing on > 30% of PASSED runs is excluded from defining process failures.

## Offline Training Arms (matched rollout pool, Weekends 5–7)

| Arm | Type | Description |
|---|---|---|
| `B0` | Baseline | Base model under bare h and under h* (`harness_bank/appworld_student`) |
| `RFT` | Core | SFT on successful bare-h trajectories (`masked_assistant_turns = range(10)`) |
| `T-DPO` | Core | Whole-trajectory DPO (successful vs failed rollout on the same task) |
| `G-REW` | Core | `T-DPO` + within-task failure-vs-failure pairs ranked by `outcome − 0.1·#nonterminal_guards`, pair count matched to `GF-DPO` |
| `R-FORK-DPO` | Core | Uniform forkable turn t_rand < T on the same failed trajectory set as `GF-DPO`, K = 8 |
| `GF-DPO` | Core | Fork at pre-action state s_g of first non-terminal guard g < T; single-turn DPO on AST-distinct (a_g⁺, a_g⁻) |
| `MC-FORK-DPO` | Bonus | Reuses Weekend 2–3 MC probe continuations at t* |

Plain DPO is primary (length-normalized is an ablation). 2 training seeds for `GF-DPO` and the best baseline, 1 otherwise. Pairs must differ in AST API-call signature; pairs differing only in `<think>` are flagged.

## Stretch: J-Lens Internal-State Localizer (After Q1 Numbers)

Once Q1 produces failed trajectories with Monte Carlo oracle labels $t^\star$ and logged per-turn `prompt_token_ids` / `completion_token_ids`, evaluate a Jacobian-lens (`anthropics/jacobian-lens`, J-space global workspace) readout as a **third localizer** alongside free rule guards (`G1–G5`) and the expensive MC continuation oracle:

| Localizer | Online / Offline Cost | Signal | Question |
|---|---|---|---|
| Deterministic guards (`G1–G5`) | 0 model calls | Action/observation symptoms (often late, e.g. loop at turn $g > t^\star$) | How close is $g$ to $t^\star$? |
| **J-lens internal readout** | 1 offline forward pass over saved token IDs | Pre-verbal concept activations ("uncertainty", "stuck", constraint drift) at turn $t$ | Does J-space spike at $t^\star$ *before* any guard fires at $g$? |
| Monte Carlo oracle | $\sim 5 \times K$ rollouts per failure | Continuation value collapse $\hat{V}(s_t) \to \hat{V}(s_{t+1})$ | Ground-truth derailment turn $t^\star$ |

- **No changes to W0/Q1 data collection:** `guardfork/serve/proxy.py` already logs token IDs and full requests, so J-lens probing runs strictly offline on saved trajectories without re-running AppWorld containers.
- **Model compatibility:** either fit a J-lens on `Qwen3.5-4B` or run a small matched rollout set on an open model with a pre-fitted public lens (`solarkyle/jspace-lenses`, e.g. `Gemma-4-E4B`). Do not start until Q1 has real numbers.

## Rules for Agents

See `AGENTS.md` (authoritative copy). Agents may implement guards, oracle/statistics, fork selection, and losses; keep the spec tests passing. Split lock, no-API-teacher, no-raw-data, verifier, and W0 job-launch gates are unchanged.
