# NOTES

Dated log of measurements and go/no-go decisions. `scripts/measure_w0.py --notes NOTES.md` appends a block here.

## Pre-W0 checklist

- [ ] Outside-work approval filed (date: ____). No public push/post until cleared.
- [ ] GCP credits confirmed personal; L4/G4 quota: ____; credit expiry: ____.
- [x] Harness-Zero commit pinned: `a4a6223241479c750fdab827c3a449bb1d41b99a`
- [~] AppWorld images built locally (arm64, Docker Desktop; deps layer needed `gcc libc6-dev` for psutil). Harbor oracle on `07b42fd_1` = 1.0. Full `validate.py` not run (uses Linux `ip -j`).
- [ ] `internal: true` + `inhibit_ipv4` network checked on the VM.

## Weekend 0

### 2026-10-24 Local Mac setup (M4 Pro 48GB, Docker Desktop, mlx-lm 0.32.2)

- Student: `mlx-community/Qwen3.5-4B-4bit` via `mlx_lm.server` -> `guardfork.serve.proxy --backend mlx`.
- Launcher: `scripts/run_appworld_rollout.py` (dedicated AppWorld agent/env; passthrough reviewer).
- Caveat: 4-bit MLX weights, not bf16 vLLM. Local numbers are a pipeline/sanity check, not the GCP K0 anchor.

### 2026-10-24 Local mini-screen: 4B-4bit, 21 train tasks (7 scenarios) x 2 seeds

- 42 trials: 10 passed, 26 failed, 6 hit the 40-turn limit (counted as reward 0).
- TGC 23.8, SGC 0.0. Wall p50/p95 125 / 296 s. Parse failures 0.7% of 766 turns.
- Per-task passes (of 2): 0/2 = 15 tasks, 1/2 = 2 tasks, 2/2 = 4 tasks.
  **K1 risk:** only 2/21 tasks look mixed; 4B may fail K1 at 8 seeds -> plan says switch to 9B.
- K2 PASS (0.007), K3 PASS ($54 projected), K4 PASS (7462 median prompt tokens at first non-terminal guard).
- Guards: G3 (real 401/422 API errors) fires on most failures, often at turn 3 (not logged in);
  fixed two false positives found on real runs (G5 login-then-read, G3 incidental numbers).

### 2026-10-24 Replay validation (W1 replay gate, 0 model calls)

- Full replay of 20 mini-screen trials (10 pass, 10 fail incl. 3 at 40 turns): **20/20 rewards reproduced**,
  exact observation equality on every replayed command, initial-context sha256 matched every time.
- Fork smoke (passed run, fork at t=6, K=1): seeded 33-message prefix, continued 9 turns, reward 1.0.
- Guard fork (failed `302c169_2` seed, 40-turn failure, first non-terminal guard G3 at t=3, K=2):
  both continuations passed (2/2). **Pipeline check only, not localization evidence:** this task is
  mixed from s_1 (1/2 base passes), so 2/2 from s_3 is roughly what we'd expect even if nothing broke
  at t=3. Localization needs V(s_1) vs V(s_{g+1}) at K=8 (the referee), on fresh runs.

### 2026-10-25 Guard freeze and dev/eval split

- Guards frozen as `guards-v1` (git tag; `config/guards_freeze.toml` pins the source sha256;
  `tests/test_guards_freeze.py` fails on any edit without a version bump).
- Dev set = everything v1 was tuned/debugged on (`config/dev_set.toml`: the 21 mini-screen tasks +
  `07b42fd_1`, and the mini-screen/smoke jobs). Q1 eligibility excludes these tasks and jobs.
- Q1 is evaluated only on fresh failed runs of non-dev tasks. Any later rule change is v2, reported
  separately, never re-scored on v1's eval runs as if it were v1.
- G3 dominates first firings on the dev runs (mostly API auth errors at turns 3-5). Report per guard,
  with G3 excluded, and with "unrecovered G3" (no api_docs read or login within 2 turns).
- The guard-vs-pass/fail firing-rate gap on dev runs (failed runs fire more often) is not the Q1
  claim and is based on only 10 passed runs; Q1 is whether g lands near t*.

### 2026-10-25 Local mini-screen: 9B-4bit vs 4B-4bit, same 21 tasks (7 scenarios)

Caveat: 4-bit MLX, pipeline sanity check, not official W0. No SGC/TGC here is a K0 result; the K0
anchor stays the full-precision GPU run. These 21 tasks are in the dev set (not Q1-eligible).

- Student `mlx-community/Qwen3.5-9B-4bit` (`mlx_lm.server`, max-tokens 4096), proxy log
  `runs/proxy/mlx_9b_mini.jsonl`. Jobs: `w0-mini-9b-mlx` (seeds 1-2, conc 2, 94 min),
  `w0-mini-9b-mlx-s34` (seeds 3-4, conc 3, 56 min), `-retry-a`/`-retry-b` (re-ran 5 slots).
- Infra: Harness-Zero's student client timeout is 120 s (x3 retries). Long 9B reasoning turns at
  conc 2 exceeded it and killed 5 trials of seeds 1-2 (`OpenAITimeoutError`, no reward). Those 5 are
  dropped and re-run with `--student-timeout 600` (new launcher flag); 0 timeouts at 600 s in 47 trials.
  Use `--student-timeout 600` for every 9B job.
- `scripts/compare_jobs.py --drop-timeouts --a 4B runs/proxy/mlx_4b_smoke.jsonl jobs/w0-mini-4b-mlx
  --b 9B runs/proxy/mlx_9b_mini.jsonl jobs/w0-mini-9b-mlx jobs/w0-mini-9b-mlx-s34 jobs/w0-mini-9b-mlx-retry-{a,b}`:

| metric | 4B (2 seeds) | 9B (4 seeds) |
|---|---|---|
| per-task passes | 0/2: 15, 1/2: 2, 2/2: 4 | 0/4: 3, 1/4: 0, 2/4: 3, 3/4: 0, 4/4: 15 |
| K1 share, p_hat in [0.25, 0.75] | 2/21 = 0.10 | 3/21 = 0.14 |
| TGC (sanity only) | 23.8 | 78.6 |
| parse failures, % of turns | 0.7 of 766 | 0.1 of 1716 |
| 40-turn exhaustions | 6 / 42 | 13 / 84 |
| median wall per episode (s) | 125 | 201 (170 at conc 2, 226 at conc 3) |
| throughput (min per episode, wall) | 1.2 (conc 2) | 1.3 (conc 3; conc 2 was 2.2 incl. timeouts) |
| median prompt tokens / request | 9134 | 9420 |
| completion tokens / episode | 4162 | 3509 |

- Per-task agreement: 9B solves every task 4B solves (6), plus 12 more; 3 tasks (all of scenario
  22cc237) are 0 for both. 9B's mixed tasks: 0d8a4ee_1, 2a163ab_2, 2a163ab_3 (2/4 each).
- Outcomes cluster by scenario (22cc237 0/12; most others 4/4 on every variant), so 21 tasks is
  effectively 7 draws; the K1 share has a wide interval.

**Decision (2026-10-25):** both models are below K1's 0.20 on this mini set, failing in opposite
directions (4B: mostly 0/2; 9B: mostly 4/4). Use 9B for the overnight train147 screen: it is the
model whose pass band can be widened by harder scenarios, it parses cleanly, and its failures are
genuine (turn limits, wrong answers) rather than inability to use the API. Screen train147 x 2 seeds
at conc 3 with `--student-timeout 600` (~294 episodes x 1.3 min ~ 6.5 h), then +2 seeds on tasks
with 1/2, and +6 on mixed tasks to reach K1's 8 samples. If the train147 K1 share is still < 0.20,
the plan's rule applies (switch to SpreadsheetBench Verified). The Monte Carlo referee
(~6000 episodes, ~1.3 min each on this Mac = ~130 h) belongs on GPU, not the Mac.

### 2026-10-25 Greedy-decoding bug (invalidates the per-seed numbers above)

- Harness-Zero's student client sends no sampling fields, and `mlx_lm.server` defaults to
  temperature 0.0. Every local run before this entry was greedy: in 12/21 dev tasks all four 9B
  seeds issued identical first three commands. Seeds were not independent samples.
- Invalid until rerun: the 4B/9B per-task pass counts and K1 shares, the 78.6% 9B TGC (a greedy
  pass is the same run counted 4x), and the 2/2 `302c169_2` fork (two greedy continuations from the
  same state are the same run). Still valid: 20/20 replay parity and the fork pipeline itself
  (replay makes no model calls), K2 parse rates, per-episode timings.
- Fix: `guardfork.serve.proxy --sampling qwen-thinking-code` (default) injects temperature 0.6,
  top_p 0.95, top_k 20, min_p 0.0 when the client omits them (Qwen3.5 model card, thinking mode,
  coding) and logs them per request. Check: `0d8a4ee_2` (greedy 4/4) x3 sampled seeds -> rewards
  0/1/1, 3/3 distinct trajectories.
- Turn cap: our 40 equals Harness-Zero's default (`student_max_turns=40`); the paper's exact
  AppWorld cap is still to be confirmed against the paper text.
- Open: Harness-Zero's own sampling settings for its Qwen students are not in the repo; if the
  train147 baseline stays far above their 26.8 SGC (test_normal), diff configs before the referee.
