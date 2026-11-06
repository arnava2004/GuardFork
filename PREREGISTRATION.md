# GuardFork Q1 pre-registration (v2)

Locked before any referee run on evaluation scenarios. Numeric blanks are Arnav's and must be
committed before `scripts/run_referee.py` touches an evaluation-scenario failure. Changes after
that date are amendments, listed at the bottom with dates.

## Split (scenario level, `config/dev_set.toml`)

- **Design: 17 scenarios / 51 tasks** (15 / 45 before amendment 1). The 8 scenarios guards-v1 was debugged on (`07b42fd`,
  `0d8a4ee`, `22cc237`, `23cf851`, `27e1026`, `287e338`, `2a163ab`, `302c169`; 6 of these carry
  the official `train` tag, so the official dev/train tags are not a leak-free split) plus 7
  untouched official-dev scenarios chosen by salted hash, plus the 2 referee-smoke scenarios
  (`60d0b5b`, `c901732`, official train; amendment 1). Used for guard tuning, judge-prompt
  tuning, lag-prior calibration and thresholds.
- **Evaluation: 32 scenarios / 96 tasks** (22 official-train + 10 official-dev, never used for
  tuning; the Mac pilot screen ran base rollouts on them, see amendment 2). Headline Q1 curves use evaluation scenarios only. Scenario is the bootstrap
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

- **Candidate pool:** every graded failed run (reward 0, verifier ran) of the bf16 failure pool on
  evaluation scenarios, with no base pass-rate band; when cost forces a subset it is a seeded
  random sample (`run_referee.py --sample-seed`), never a band (amendment 2).
- **s1 screen:** K = 8 continuations from s1 for every candidate failed run. **Hopeless** = 0/8,
  confirmed 0/16 with 8 more. Everything with ≥ 1/16 enters the localization pool. 0/16 tasks are
  not labelled environment faults automatically: cross-check whether 9B solves the task and
  manually review 10–20 before assigning fault.
- **Probe set P per eligible failure** = union of: thinned grid (every 2nd turn on 4B, every 3rd on
  9B) at K = 8; mandatory {g−1, g, g+1} around the first firing g of each guard; {j−1, j, j+1}
  around the open-weight judge pick j; {e−1, e, e+1} around the original failed run's token-entropy
  peak e (from the base run's logged logprobs: proxy records joined to the run by completion id;
  mean sampled-token surprisal, since `top_logprobs` are not logged); plus +8 rollouts (K = 16) at the two turns
  bracketing the candidate change-point.
- **Snapping:** a search policy asking for an unmeasured turn t snaps to argmin over u in P of
  (|u − t|, u). **10–15 failures are probed fully dense (every turn)** to bound snapping bias.

## Primary oracle

- **t\* = change-point first-irrecoverable turn at τ = 0.5·V̂(s1)**: the first turn after which
  the fitted continuation value stays ≤ τ, from a change-point fit over all K·T outcomes, with a
  90% bootstrap credible interval [t\*_lo, t\*_hi]. Implemented in
  `guardfork/stats/bayes_prior.py::fit_changepoint_oracle`: maximum-likelihood single step over
  the probed turns (pooled binomial levels p_hi up to c, p_lo after), restricted to steps with
  p_hi > τ ≥ p_lo, ties to the earlier turn; t\* = c, undefined when no step qualifies. Interval:
  1000 Beta-posterior parametric bootstrap refits, nearest-rank 5th/95th percentiles over
  replicates with a defined t\*, widened to contain the point estimate.
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
lag prior + sequential Beta stopping) with the open-judge distribution as likelihood,
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

## Credit pilot (Section 5) primary metric

Prefix drift (`guardfork/train/prefix_drift.py`): per-turn Δ_t = log π_after − log π_before on
held-out oracle-labelled failures, grouped by [t\*_lo, t\*_hi]; primary statistic is the
prefix-protection gap mean(Δ_pre) − mean(Δ_at) with a paired scenario-clustered bootstrap CI,
compared across 1-seed checkpoints (uniform DPO, hard-mask DPO, soft-posterior PW-DPO, GF-DPO).
test_normal TGC is secondary (appendix, paired CIs). Credit rules are also scored against the
learner's own Δ_t^MC (`guardfork/train/credit.py`); QVal (2606.32034) is the anchor citation,
with our two differences (on-policy learner values; deterministic harness-rule family) stated.

## Predictions (Arnav fills before the first evaluation-scenario referee run)

- P1 Hit(w=1) of `guards_recovery_aware` minus random: __ pp on 9B, __ pp on 4B.
- P2 `guard_bayesect` reaches the same Hit(w=1) as `bisection_fixed` with __% of its continuations.
- P3 share of failures hopeless from s1: __% (4B), __% (9B).
- P4 first-third agreement with t\* (LVRL's weak zone): guards __% vs error-marker floor __%.
- P5 oracle test-retest (split-half, ±1 turn): __%.
- P6 `judge_open` vs `judge_claude` agreement within ±1 turn: __%; Hit(w=1) gap: __ pp.
- P7 judge calibration: among picks with stated p ≥ 0.7, observed Hit(w=1) = __% (open), __% (Claude).
- P8 prefix-protection gap (nats/turn): uniform DPO __, hard-mask __, soft-posterior __, GF-DPO __.

## Amendments

- **1 (2026-10-25), split.** The 2-failure Mac referee smoke (`--K 4`, 9B-4bit, pricing only)
  probed `60d0b5b_1__pHRgiha` and `c901732_3__8KZ792a`, both in evaluation scenarios, before
  P1–P8 were committed; their value curves were then read. Both scenarios move to design
  (design 17 / 51 tasks, evaluation 32 / 96). Nothing else on the evaluation side was probed,
  judged or inspected. The same day the referee was fixed so `v_s1` pools the s_1 probe
  with the base runs (it had used the base-run Jeffreys value only), mandatory windows can no
  longer be cut by `max_probes`, and a failed probe no longer yields an oracle label; the
  smoke's oracle.json files predate those fixes.
- **2 (2026-11-05), split wording, pool, entropy peak, G4, oracle details.** Made before any
  referee run on an evaluation scenario and before P1–P8 are filled.
  - *Correction to amendment 1.* "Nothing else on the evaluation side was probed, judged or
    inspected" was wrong. The Mac 9B-4bit pilot screen `w0-screen-9b-t06-s12` (2 sampled seeds on
    125 non-design train147 tasks, 2026-10-25) ran base rollouts on all 32 evaluation scenarios.
    Per-task pass counts and the aggregate W0 statistics in NOTES (TGC, SGC, guard false-alarm
    rate K2, prompt tokens at the first guard K4) were computed over those runs, and the two smoke
    failures of amendment 1 were picked from them. No other evaluation-side run was
    referee-probed or judged. Whether individual evaluation trajectories were opened in the viewer
    is not logged; none fed guard, prompt or threshold tuning (guards-v1 was frozen before the
    screen). The evaluation scenarios stay as they are: the screen's runs are Mac pilot data and
    never enter Q1 tables (single-stack rule), and the pool below does not select on its pass
    counts. "Untouched" is dropped from the split description here and in `config/dev_set.toml`.
  - *Candidate pool.* Every graded failed run, no pass-rate band (the referee script's v1
    default `--band 0.3 0.8` is now `0 1`); subsets are seeded random samples. A band would
    condition the pool on task difficulty and make P3 (hopeless share) uninterpretable.
  - *Entropy peak.* Proxy records carry the completion id (`response_id`), which Harness-Zero
    also stores per turn, so a run's records are joined exactly (250/250 Mac screen trials matched
    on every turn). The statistic is mean sampled-token surprisal, not full-vocabulary entropy.
  - *G4.* Analysed as implemented in guards-v1: it fires on destructive calls (`delete_`,
    `remove_`, `drop_`, `destroy_`, `unlink`). The "apps outside the task" branch never fires,
    because a trajectory carries no allowed-app list, and none is added: the only source would be
    AppWorld's grader-side task metadata, which a free harness guard must not read. No
    guards-v2 bump.
  - *Implementation.* `bayes_prior.py` (oracle, Laplace prior, sequential Beta stopping,
    next-probe rule) was implemented by an agent at Arnav's direction against the pre-written
    specs, instead of being hand-written; the oracle details are stated under "Primary oracle".
