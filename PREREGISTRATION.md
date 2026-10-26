# GuardFork Q1 pre-registration (v2)

Locked before any referee run on evaluation scenarios. Numeric blanks are Arnav's and must be
committed before `scripts/run_referee.py` touches an evaluation-scenario failure. Changes after
that date are amendments, listed at the bottom with dates.

## Split (scenario level, `config/dev_set.toml`)

- **Design: 15 scenarios / 45 tasks.** The 8 scenarios guards-v1 was debugged on (`07b42fd`,
  `0d8a4ee`, `22cc237`, `23cf851`, `27e1026`, `287e338`, `2a163ab`, `302c169`; 6 of these carry
  the official `train` tag, so the official dev/train tags are not a leak-free split) plus 7
  untouched official-dev scenarios chosen by salted hash. Used for guard tuning, judge-prompt
  tuning, lag-prior calibration and thresholds.
- **Evaluation: 34 scenarios / 102 tasks** (24 untouched official-train + 10 untouched
  official-dev). Headline Q1 curves use evaluation scenarios only. Scenario is the bootstrap
  cluster.
- `test_normal` (168 tasks) is used only for K0 and final Q2 evals (`allow_final=True`).

## Data generation

- **Step 0, concurrency pilot:** on the GPU VM (bf16 vLLM), benchmark C in {8, 16, 32} parallel
  AppWorld environments at realistic 8–20k-token contexts; record continuations/hour, per-turn
  latency, RAM per container, before sizing anything.
- **Single stack:** the Q1 failure pool and all continuations come from the same bf16 vLLM
  stack. Mac 4-bit MLX runs are design/pilot only and never enter Q1 tables.
- Sampling: `config/sampling.toml` on both machines, logged per request with a seed.

## Screening and probing

- **s1 screen:** K = 8 continuations from s1 for every candidate failed run. **Hopeless** = 0/8,
  confirmed 0/16 with 8 more. Everything with ≥ 1/16 enters the localization pool. 0/16 tasks are
  not labelled environment faults automatically: cross-check whether 9B solves the task and
  manually review 10–20 before assigning fault.
- **Probe set P per eligible failure** = union of: thinned grid (every 2nd turn on 4B, every 3rd on
  9B) at K = 8; mandatory {g−1, g, g+1} around the first firing g of each guard; {j−1, j, j+1}
  around the open-weight judge pick j; {e−1, e, e+1} around the original failed run's token-entropy
  peak e (from the base run's logged logprobs); plus +8 rollouts (K = 16) at the two turns
  bracketing the candidate change-point.
- **Snapping:** a search policy asking for an unmeasured turn t snaps to argmin over u in P of
  (|u − t|, u). **10–15 failures are probed fully dense (every turn)** to bound snapping bias.

## Primary oracle

- **t\* = change-point first-irrecoverable turn at τ = 0.5·V̂(s1)**: the first turn after which
  the fitted continuation value stays ≤ τ, from a change-point fit over all K·T outcomes, with a
  90% bootstrap credible interval [t\*_lo, t\*_hi]. Implemented in
  `guardfork/stats/bayes_prior.py::fit_changepoint_oracle` (Arnav).
- **Sensitivity only:** τ in {0.3, 0.7}·V̂(s1); largest-drop t\*_drop; the v1 adjacent-pair rule
  in `guardfork/stats/oracle.py`.
- **Simulator unbiasedness:** 8/8 split-half on the K = 16 states (8 rollouts define t\*, 8
  independent rollouts drive the search policy); Beta-posterior parametric bootstrap
  p_t ~ Beta(1 + s_t, 1 + K_t − s_t) elsewhere.

## Localizers compared (all scored against t\* with the same windows w in {0, 1, 2, 3})

`lvrl_style_error_marker` (floor: explicit strings `Traceback`, `Execution failed`,
`Response status code is 4`, `Error:`), `random`, `last_turn`, `entropy_peak`, `guards_raw`
(first G1–G5), `guards_recovery_aware` (`g3_unrecovered` + G2/G4/G5), per-guard, `no_g3`,
LLM judges, `bisection_fixed`, `bisection_early_stop`, `elpo_bel`, `guard_bayesect` (Laplace
lag prior + sequential Beta stopping, Arnav) with the open-judge distribution as likelihood,
and the same with the Claude distribution as a separate **eval-only** row, plus human labels
where available. Primary metric: Hit(w = 1) minus analytic random, scenario-clustered
bootstrap; cost in continuations per failure alongside.

**LLM judges (zero rollouts, one call per failed run, `config/judges.toml`):**
- `judge_open` (primary): hosted Qwen3-235B (OpenRouter, pinned id).
- `judge_local` (optional cheap row): Qwen3-30B-A3B-4bit on the Mac.
- `judge_claude` (eval-only frontier reference): one pinned Claude Sonnet snapshot, non-training
  API. Never a training signal; continuations probed only because of its pick are `eval_only`.
- All judges get the identical prompt (`config/judge_prompt.md`, frozen as `judge-v1` on the
  design split) with the identical, logged transcript truncation, and return a probability
  distribution over turns (top-3 with p); the argmax is the point pick for Hit(w), the
  distribution is the prior likelihood for `guard_bayesect`. No judge promotion rule: rows are
  reported side by side. Per judge we report the pick-position histogram (first / middle / last
  third, last turn) and calibration (stated p vs observed hit).

## Predictions (Arnav fills before the first evaluation-scenario referee run)

- P1 Hit(w=1) of `guards_recovery_aware` minus random: __ pp on 9B, __ pp on 4B.
- P2 `guard_bayesect` reaches the same Hit(w=1) as `bisection_fixed` with __% of its continuations.
- P3 share of failures hopeless from s1: __% (4B), __% (9B).
- P4 first-third agreement with t\* (LVRL's weak zone): guards __% vs error-marker floor __%.
- P5 oracle test-retest (split-half, ±1 turn): __%.
- P6 `judge_open` vs `judge_claude` agreement within ±1 turn: __%; Hit(w=1) gap: __ pp.
- P7 judge calibration: among picks with stated p ≥ 0.7, observed Hit(w=1) = __% (open), __% (Claude).

## Amendments

- (none)
