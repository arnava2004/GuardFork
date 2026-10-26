# Frontier tangents and the posterior-credit follow-up

Short, so it stays honest. IDs marked **(verified)** were checked against arXiv/HF on 2026-10-25;
**(unverified)** means the ID could not be resolved from this machine and must be checked before
it appears in any draft.

## Nearest neighbours and what we take from each

- **QVal, 2606.32034 (verified: "QVal: Cheaply Evaluating Dense Supervision Signals for
  Long-Horizon LLM Agents").** The closest prior for Section 5: it ranks many step-credit methods
  against reference-policy Monte Carlo Q-values. Our two differences, stated in one sentence in the
  paper: we score credit against the **learner's own on-policy** V̂(s_t) (4B/9B continuations),
  not an off-policy reference/expert policy's Q; and we add the **deterministic harness-rule
  family** (G1–G5, `g3_unrecovered`), which is absent from QVal's method families. QVal turns
  from a scoop into the anchor citation once those two sentences are in. Re-read before quoting
  its exact family list or method count.
- **AgentProcessBench, 2603.14465 (verified)** and **HCAPO, 2603.08754 (verified: "Hindsight
  Credit Assignment for Long-Horizon LLM Agents"):** step-level process quality and hindsight
  credit; cite in related work after re-reading. **REALM 2026 localization–repair gap:** no ID
  yet; find the venue/ID before citing.
- **LVRL, 2610.15840 (verified: "LLM Value Functions for Credit Assignment in Agentic RL").**
  A frozen frontier critic scores step credit against Monte Carlo ground truth. Three facts we
  use, all to be re-read from the paper before quoting: critic agreement is weakest in the first
  third of a run (~44% vs ~88% in the last third); cost on the order of $0.06–0.14 per trajectory
  in API calls; their only rule baseline is an error-marker string check catching ~13% of MC
  pitfalls. We implement an **LVRL-style error-marker floor** with our own explicit strings and
  never claim their exact strings. Do **not** cite any "9.8% / 0.0%" Table-4 number.
- **PivoARL, 2607.03702 (verified).** Spares the prefix before a pivotal turn.
- **C³, 2610.19695 (verified).** Rule-detected invalid tool calls / loops flip advantages.
- **Critical-State RL, 2609.24985 (ID verified).** The claim that its hard turn-mask control did
  not reproduce the gain is **not confirmed from the abstract; re-check the full text before
  citing it**. If it holds, it is a warning for Path 1 below.
- **DepGPO, 2610.03634 (verified).** Dependency tracing from the verifier's read set.
- **TRACE, 2607.13988 (verified)** and **SAGE-OPD, 2606.19659 (verified):** post-hoc per-turn
  loss/reward weighting, not rollout interventions.
- **Doomed-from-the-Start, 2607.06503 (verified)** and **Selective Rollout, 2605.05802
  (verified):** own in-flight abort. We report one offline savings number and stop.
- **ECHO 2610.16027 (verified: "Of Errors and Echoes"), LEGO-RL 2608.17393 (verified);
  Traverse 2609.17930, Checking the Checkers 2610.17824, CERO 2606.05606 / 2610.09679
  (unverified):** cite only after re-reading.

## Follow-up plan: posterior step credit, scored against the oracle

Decision: training-headline credit reweighting (PW-GRPO / PW-DPO) is **not** the paper's core.
Crowded neighbourhood above, and the power wall: test_normal × 5 seeds gives an MDE of roughly
4–8 TGC points while realistic offline reweighting gains are 0–3.

What we do instead, at zero extra container cost once the oracle table exists:

1. **Oracle-scored credit alignment (`guardfork/train/credit.py`).** For each failed run with
   t\* and per-turn V̂, compute Δ_t^MC = V̂(s_t) − V̂(s_{t+1}) and score each zero-rollout credit
   rule on: pitfall recall (turns with the largest drops), pre-derailment prefix preservation
   (credit mass kept on turns < t\*), and rank correlation with Δ_t^MC, overall and by trajectory
   third. Rules: `uniform_grpo`, `hard_mask_prefix` (Critical-State / PivoARL style),
   `c3_rule_flip`, `open_judge` (ECHO-style), `soft_posterior_pw` (GuardFork: Laplace guard
   posterior from `bayes_prior.py`). Prediction to lock: soft posterior beats hard masking
   whenever |g − t\*| ≥ 1, and both beat the error-marker floor in the first third.
2. **Fault triage (`guardfork/q1/triage.py`).** The s1 screen already splits failures into
   hopeless-from-s1 (0/8 → 0/16) vs mid-trajectory derailment. Add a 30-run manual audit of
   passed runs with G4/G5 firings for silent harm. Note: AppWorld's own evaluators assert
   `models.changed_model_names()` equals the expected table set and that no records were removed
   (checked in all 732 task evaluators), so collateral DB mutations already fail the verifier;
   the expected finding is near-zero silent harm on reward-1 runs, and G5 firings on passed runs
   are mostly skipped read-backs.
3. **Credit pilot, primary metric = prefix drift (Proposal 0, `guardfork/train/prefix_drift.py`).**
   test_normal TGC has a 4–8 point detection floor against a realistic 0–3 point effect, so it is
   secondary. Primary: on ~40 held-out oracle-labelled failures (~600 turns), compute per turn
   Δ_t = log π_after(a_t|s_t) − log π_before(a_t|s_t) for each 1-seed checkpoint (uniform DPO,
   hard-mask DPO, soft-posterior PW-DPO, GF-DPO), group turns by the oracle interval
   (pre: t < t\*_lo; at: t\*_lo ≤ t ≤ t\*_hi; post: t > t\*_hi), and report the prefix-protection
   gap mean(Δ_pre) − mean(Δ_at) with a paired scenario-clustered bootstrap CI. Zero container
   cost: forward passes over saved token ids on the Mac. Prediction to lock: soft posterior and
   GF-DPO show a larger gap than uniform and hard-mask.
4. **Appendix only:** the 1-seed pilot's test_normal TGC with paired CIs, and a one-paragraph
   offline abort estimate ("aborting when P(t\* ≤ t) > 0.9 saves X% of continuation tokens at 95%
   recall"; cite Selective Rollout and Doomed-from-the-Start). Silent harm stays one paragraph
   plus the 30-run audit table.

## Proposal A (separate workspace, after Q1 numbers exist)

"Does on-policy credit alignment predict RL gains?" QVal's stated open question. Test 5–6 credit
rules plus noisy variants in a fast text environment (ALFWorld or similar), correlating each
rule's oracle-alignment score with its actual RL gain. Kept out of `guardfork` so it does not
touch the AppWorld container budget; starts only once Q1 has numbers.

Implementation notes (agent, commit `c6f241a`): `elpo_bel`, `c3_rule_flip` and `open_judge`
in `guardfork/q1/simulate_search.py` / `guardfork/train/credit.py` are simplified stand-ins and
must be checked against ELPO 2602.09598, C³ 2610.19695 and ECHO 2610.16027 before any table
calls them by those names. `entropy_peak` uses sampled-token surprisal (the proxy logs no
top-k logprobs), not full-distribution entropy. `bisection_early_stop` stops early only on the
collapsed side; stopping on unanimous-alive half-samples inflated V̂ in simulation.

Ownership: Arnav hand-writes `guardfork/stats/bayes_prior.py` (change-point oracle, Laplace
guard prior, sequential Beta stopping, next-probe choice). Agents write stubs, the simulator,
credit/triage scaffolding and tests.
