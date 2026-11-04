# GuardFork

**Can an agent harness's own free rule checks tell you *where* a rollout went wrong?**

> 🚧 **Work in progress.** The pipeline is built and tested; the GPU experiments have not run
> yet. There are **no results in this repo yet**, only the design, the code, and a
> pre-registration. Numbers will land here when they exist.

Long-horizon LLM agents usually get one reward at the end of a run. When a 40-turn AppWorld
episode fails, which turn broke it? Existing step-level methods answer with extra rollouts,
trained critics or LLM judges, and every one of those costs compute.

Agent harnesses already compute cheap, deterministic checks on every turn: invalid tool
calls, loops, failed API preconditions, constraint violations, unverified completions.
GuardFork asks whether those **guards** can localize the derailment turn for free, and whether
forking at that turn gives better post-training data than forking anywhere else.

## The questions

**Q1: Localization.** Do offline guards G1–G5 land near the first irrecoverable turn `t*` of a
failed AppWorld run? We measure this against a test-retest-bounded Monte Carlo continuation
oracle, and against analytic random-turn, last-turn, error-marker and LLM-judge baselines. We
also report what each localizer costs in continuations per failure.

**Q2: Post-training (conditional on Q1).** From one shared offline rollout pool, does
turn-level DPO at guard-forked states (`GF-DPO`) beat whole-trajectory SFT, whole-trajectory
DPO, guard-as-reward DPO and random-turn fork DPO?

To our knowledge, no prior work evaluates verifier-free harness guards as fork-point
localizers against a reliability-bounded Monte Carlo oracle.

## How it works

```
failed rollout ──► guards G1–G5 (0 model calls) ──► first non-terminal firing g
       │
       └──► replay-from-zero forks at s_t ──► K continuations per probed turn
                                               │
                                               ▼
                         V̂(s_1), V̂(s_t), ...  ──► change-point oracle t*
                                               │
                         Hit(w) = |g − t*| ≤ w, minus analytic random baseline
```

| Guard | Fires on |
|---|---|
| G1 `INVALID_ACTION` | malformed or multiple tool calls, empty commands, leaked tool markup |
| G2 `LOOP` | the same API call and outcome repeated, or A/B alternation |
| G3 `FAILED_PRECONDITION` | HTTP 401/403/404/422, or entity IDs never seen in prior outputs |
| G4 `CONSTRAINT_VIOLATION` | apps outside the task's scope, or destructive writes |
| G5 `UNVERIFIED_COMPLETION` | completing without a verification read |

Guards are frozen (`guards-v1`, sha-pinned in `config/guards_freeze.toml`). A test fails on any
edit that isn't a version bump.

## Built for honesty

- **Pre-registered.** `PREREGISTRATION.md` fixes the oracle, localizers and metrics before
  any evaluation run. Changes are dated amendments.
- **Scenario-level split.** 17 design scenarios are used for tuning and 32 for evaluation
  (`config/dev_set.toml`). `test_normal` only loads with `--final`.
- **Kill criteria first.** `config/kill.toml` defines go/no-go gates checked before any
  GPU-scale job.
- **Exact replay.** Each fork re-executes commands 1..t−1 and asserts exact observation
  equality. Replay parity on the pilot was 20/20.
- **Open-weight only for training.** No API teachers, and no API-generated training data.
  Judges are evaluation-only rows.
- **No raw data in git.** A pre-commit hook blocks rollouts, forks, task text and API keys.

## Status

| | |
|---|---|
| ✅ | Guards, trajectory conversion, replay forking, MC referee, LLM-judge rows, Q1 table, search-policy simulator, DPO pair export, VM scripts, trajectory viewer. 201 offline tests. |
| ✅ | Mac pilot (Qwen3.5-4B/9B, 4-bit MLX) to validate the pipeline. These are not results. |
| 🔨 | Change-point oracle and Bayesian guard prior (`guardfork/stats/bayes_prior.py`), hand-written. |
| ⏳ | Pre-registered predictions, then bf16 vLLM runs on GPU (K0 anchor, Q1 referee at K=8/16). |
| ⏳ | Q2 training arms, a second benchmark, a paper. |

Running log of measurements and decisions: [`NOTES.md`](NOTES.md). Full plan:
[`guardfork.md`](guardfork.md).

## Layout

```
guardfork/
  guards/      G1–G5 rules + AST call signatures
  fork/        replay plans, fork agent, MC referee, probe planning
  stats/       oracle, bootstrap, Bayesian prior (in progress)
  q1/          localization table, search-policy simulator, triage
  train/       preference pairs, DPO export, losses, credit, prefix drift
  serve/       logging proxy (sampling injection, token ids), template parity
  telemetry/   Harbor trial -> trajectory
  judge.py     zero-rollout LLM-judge localizers
scripts/       rollouts, referee, judge, Q1 report, VM setup
config/        kill criteria, splits, sampling, judges, freezes
tools/viewer/  local trajectory viewer
```

## Quickstart

```bash
uv sync
uv run pytest -q
git config core.hooksPath hooks
```

Running rollouts needs AppWorld in [Harness-Zero](https://github.com/metaevo-ai/harness-zero)'s Harbor container
setup and an OpenAI-compatible model server. See `scripts/w0_runbook.md`.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
