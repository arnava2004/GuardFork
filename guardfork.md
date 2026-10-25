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

## Pre-Registered Predictions & MDEs (lock numbers before the first referee run on fresh runs)

Arnav fills the blanks below (and the per-guard / no-G3 expectations) before `scripts/run_referee.py` runs on non-dev tasks; commit them so the timestamp precedes the results.

1. **P1 — Non-terminal localization (w = 1 primary):** On n ∈ [80, 100] eligible failed trajectories (p_task ∈ [0.3, 0.8]) with a non-terminal guard firing (g < T) and defined t*, Hit(w=1) = Pr(|g − t*| ≤ 1) exceeds analytic random E[Hit_rand(w=1)] by __ pp (scenario-clustered bootstrap; 80%-power MDE ≈ 16–20 pp at n = 90). Report w ∈ {0, 1, 2, 3} alongside the 15–25-run oracle test-retest ceiling and its 95% CI.
2. **P2 — Pair yield & cost per usable pair:** Guard forking yields __ usable AST-distinct preference pairs per failed run at 1×K continuations per failure, at __× lower rollout-token and env-second cost per usable pair than MC probe localization (~5×K per failure) and __× higher usable-pair yield than random-turn forking (1×K per failure).
3. **P3 — Offline post-training on test_normal (5 eval seeds):** Across 168 test_normal tasks (56 scenarios) × 5 eval seeds (primary: TGC, 3-contrast Holm-adjusted 80% MDE ≈ 9.5–10.0 pts; secondary: SGC, MDE ≈ 14 pts), `GF-DPO` beats `T-DPO` by __, `R-FORK-DPO` by __, and `G-REW` by __ TGC pts. Sub-MDE contrasts are reported as 95% CIs / equivalence bounds, never as wins.
4. **P4 — Harness internalization:** Under bare h, `GF-DPO` reduces non-terminal guard firings per episode by __% (including held-out G3, excluded from fork selection). If the base h* − h gap on train147 is ≥ 10 TGC pts (95% CI excluding 0), report bootstrapped Gap Closure; otherwise report raw ΔTGC under h and h*.
5. **P5 — Process vs. content failures:** `GF-DPO` vs `R-FORK-DPO` at matched budget: gains concentrate on trajectories that failed with a non-terminal guard rather than unguarded content failures.

## Setup, Splits & Guard Definitions

- **Benchmark:** AppWorld in Harness-Zero's container setup (`AppWorldMinisweAgent` + `AppWorldDockerEnvironment`; 23 initial messages = 10 demo assistant turns before `[APPWORLD_CURRENT_TASK]`; one `execute` bash tool).
- **Scenario-grouped splits** (`scenario = task_id.rsplit("_", 1)[0]`, 3 tasks/scenario), implemented in `guardfork/splits.py`:
  - `train`: 40 scenarios (120 tasks) from train147.
  - `dev_heldout`: 9 scenarios (27 tasks) from train147, for hyperparameter/checkpoint selection.
  - `test_normal`: 56 scenarios (168 tasks), locked in the loader unless `allow_final=True` (`--final`).
- **Models (open-weight only, served via vLLM on GCP L4/G4 spot):** `Qwen/Qwen3.5-4B` primary; `Qwen/Qwen3.5-9B` if 4B fails K1. Serving/training template parity: `Qwen3_5Renderer(strip_thinking_from_history=False)` verified against vLLM `prompt_token_ids` in Weekend 0.
- **Offline guards** (`guardfork/guards/rules.py`, evaluated on `assistant_index >= 10` only):
  - G1 `INVALID_ACTION` (evaluated first): `raw_original.invalid_tool_calls` non-empty, >1 tool call, tool name ≠ `execute`, empty command, empty message, visible `content` with code fences / `appworld exec`, or `reasoning_content` with structural tool markup (`<tool_call>`, `<function=`).
  - G2 `LOOP`: identical/near-identical AST API signature + outcome ≥ 3 times, or A/B alternation over 6 turns.
  - G3 `FAILED_PRECONDITION`: HTTP 401/403/404/422 API errors, or entity IDs never observed in prior tool outputs (held-out guard in the Q2 ablation).
  - G4 `CONSTRAINT_VIOLATION`: mutating an app outside the task's allowed apps, or unrequested destructive write/delete.
  - G5 `UNVERIFIED_COMPLETION` (only if not G1): no tool call while completion status is not True, or `complete_task` without a post-write verification read.
  - `MAX_TURNS` / `CONTEXT_OVERFLOW` / `REPLAY_DIVERGED` are `TrajectoryStatus` values, not localizers.

## Harbor-Native Forking & Oracle Definition

- **Replay-from-zero forks in Harbor (`ForkReplayAppWorldAgent`):** each continuation is an isolated Harbor trial (`docker compose -p fork_<uuid>`, `PYTHONHASHSEED=0`, `down -v` in `finally`). `initialize` runs `appworld task` + `PROFILE_COMMAND`, checks sha256 against `appworld_initial_context.json`, replays commands 1..t−1 via `backend.aexecute` asserting exact observation equality, seeds the logged message prefix, and runs the inner graph with `max_turns = 40 − (t − 1)`. Graded only by Harbor's official verifier. Turns after `session_reset=True` or any `agent "<task>"` subagent call are non-forkable.
- **Monte Carlo oracle (`guardfork/stats/oracle.py`):**
  - Eligible failed runs: tasks with p̂_task ∈ [0.3, 0.8] (≥ 8 base rollouts; Jeffreys-smoothed Ṽ(s₁)).
  - Probes at {0, g−3, g−1, g, g+1} around the first non-terminal guard g < T, plus bisection to adjacency, K = 8 Harbor trials per probed state.
  - t* = min{t ∈ 1..T : V̂(s_{t+1}) ≤ 0.5·Ṽ(s₁) ∧ V̂(s_t) − V̂(s_{t+1}) ≥ 2/K}, with V̂(s_{T+1}) ≡ R(τ) = 0. If no adjacent probed pair qualifies, t* is undefined: excluded, rate reported, sensitivity = argmax drop over probed adjacent pairs.
- **Guard freeze & dev/eval separation:** guards are frozen as `guards-v1` before any Q1 number (`config/guards_freeze.toml`, enforced by `tests/test_guards_freeze.py`). Runs and tasks used to tune or debug them are the dev set (`config/dev_set.toml`) and are excluded from Q1. Q1 uses only fresh failed runs on non-dev tasks; later rule changes are new versions reported separately.
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
