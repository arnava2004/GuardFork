# Frontier tangents and the posterior-credit follow-up

Short, so it stays honest. IDs marked **(verified)** were checked against arXiv/HF on 2026-10-25;
**(unverified)** means the ID could not be resolved from this machine and must be checked before
it appears in any draft.

## Nearest neighbours and what we take from each

- **LVRL, 2610.15840 (verified: "LLM Value Functions for Credit Assignment in Agentic RL").**
  A frozen frontier critic scores step credit against Monte Carlo ground truth. Three facts we
  use, all to be re-read from the paper before quoting: critic agreement is weakest in the first
  third of a run (~44% vs ~88% in the last third); cost on the order of $0.06–0.14 per trajectory
  in API calls; their only rule baseline is an error-marker string check catching ~13% of MC
  pitfalls. We implement an **LVRL-style error-marker floor** with our own explicit strings and
  never claim their exact strings. Do **not** cite any "9.8% / 0.0%" Table-4 number.
- **PivoARL, 2607.03702 (verified).** Spares the prefix before a pivotal turn.
- **C³, 2610.19695 (verified).** Rule-detected invalid tool calls / loops flip advantages.
- **Critical-State RL, 2609.24985 (verified).** Ran a hard turn-mask control that did not
  reproduce their gain; a warning for Path 1 below.
- **DepGPO, 2610.03634 (verified).** Dependency tracing from the verifier's read set.
- **TRACE, 2607.13988 (verified)** and **SAGE-OPD, 2606.19659 (unverified):** post-hoc per-turn
  loss/reward weighting, not rollout interventions.
- **Doomed-from-the-Start, 2607.06503 (verified)** and **Selective Rollout, 2605.05802
  (unverified):** own in-flight abort. We report one offline savings number and stop.
- **ECHO 2610.16027, Traverse 2609.17930, Checking the Checkers 2610.17824, CERO 2606.05606 /
  2610.09679, LEGO-RL 2608.17393 (verified):** cite only after re-reading.

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
3. **Appendix only:** a 1-seed, equal-episode PW-DPO vs Fork-DPO pilot with paired CIs, and a
   one-paragraph offline abort estimate ("aborting when P(t\* ≤ t) > 0.9 saves X% of continuation
   tokens at 95% recall").

Implementation notes (agent, commit `c6f241a`): `elpo_bel`, `c3_rule_flip` and `open_judge`
in `guardfork/q1/simulate_search.py` / `guardfork/train/credit.py` are simplified stand-ins and
must be checked against ELPO 2602.09598, C³ 2610.19695 and ECHO 2610.16027 before any table
calls them by those names. `entropy_peak` uses sampled-token surprisal (the proxy logs no
top-k logprobs), not full-distribution entropy. `bisection_early_stop` stops early only on the
collapsed side; stopping on unanimous-alive half-samples inflated V̂ in simulation.

Ownership: Arnav hand-writes `guardfork/stats/bayes_prior.py` (change-point oracle, Laplace
guard prior, sequential Beta stopping, next-probe choice). Agents write stubs, the simulator,
credit/triage scaffolding and tests.
