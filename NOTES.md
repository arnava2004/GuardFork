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

### 2026-10-25 03:30 Sampling made explicit; overnight plan

- `config/sampling.toml` (qwen-thinking-code-v1; v2 adds the 4096 cap, see below) is now the single source of student sampling; the
  proxy loads it (`--sampling-config`), adds a logged per-request seed, and stamps each proxy record
  with the config name + sha. Launches write `guardfork_run.json` (config, commits, PYTHONHASHSEED).
- Running: `w0-screen-9b-t06-s12` (125 non-dev train147 tasks x 2 seeds, sampled, concurrency 3),
  started on the pre-config proxy with identical values (temp 0.6 / top_p 0.95 / top_k 20) but no
  per-request seed. Early: 16 graded, 5 passed, 0 turn-limit. Swap 7.5 GB used but 49% memory free
  (stale swap, not active pressure).
- Queued (`scripts/queue_after_screen.sh`): restart proxy from config, then `w0-dev-9b-t06`
  (21 dev tasks x 4 sampled seeds) to re-check guards-v1 false alarms; v1's freeze decisions were
  measured on greedy, partly duplicated runs. Rules unchanged -> freeze stands; a flip is still a
  dev-set decision (no leakage).
- Referee concurrency: if tonight's screen shows timeouts or slow (crawling, not looping) 40-turn
  runs, run the referee at concurrency 2.

### 2026-10-25 9B-4bit sampled screen: 125 non-dev train147 tasks x 2 seeds (`w0-screen-9b-t06-s12`)

- 250 trials: 105 passed, 110 failed, 35 hit the 40-turn limit. TGC 42.0 (seed 1 41.6, seed 2 42.4).
- **SGC per seed: 22.0 and 14.6 over 41 scenarios (mean 18.3).** Train147, 4-bit MLX, so not a K0
  number; but it rules out a 3x-inflated baseline (the earlier 78.6% was the greedy artifact).
  Harness-Zero reports 26.8 bare-h SGC on test_normal.
- Per-task passes of 2: 0/2 = 51, 1/2 = 43, 2/2 = 31. 43/125 (34%) mixed at 2 seeds; K1 (>= 20% of
  tasks in [0.2, 0.8] at >= 8 samples) not measurable yet but promising.
- K2 PASS (0.5% of 5864 turns, all candidate_issue). K3 PASS ($122 projected). **K4 FAIL**: median
  prompt tokens at first non-terminal guard 10,382 > 8,192 -> train on Lambda, not the Mac (planned).
- Wall p50/p95 264 / 704 s. Turn-limit runs: wall p50 490 s, max 1844 s (check the slowest for
  swap-crawl before the referee; drop referee concurrency to 2 if any look like crawling).
- measure_w0 fixed: SGC is now per seed (job x attempt) and K1 reports "not measurable" below 8
  samples instead of FAIL.

### 2026-10-25 Open before the first Lambda run

- [done 2026-10-25, sampling v2] **Completion-length parity:** on the Mac the proxy caps mlx completions at 4096 tokens; on vLLM
  there is no cap. Before K0, move `max_tokens` into `config/sampling.toml` so both machines use the
  same cap (pick it from the Mac proxy logs: share of completions that hit 4096). Not changed now to
  keep the running dev re-check on one config.
- Lambda scripts (`scripts/vm_setup.sh`, `scripts/vm_serve.sh`, runbook L0-L6) are written but not
  run; VERIFY items: compose fallback package, 131072-token context fitting GPU memory, non-Qwen
  parser flags. Per AGENTS rule 7, only the 2-run referee pricing smoke runs before W0 passes
  kill.toml; K0 (9B full precision on test_normal) is the first Lambda job.
- Judge baseline and DPO exporter are written; both need the Mac free (judge: ~17 GB Qwen3-30B-A3B
  server; DPO: re-fork with sampling, since all existing forks were greedy and give no real pairs).

### 2026-10-25 Cleanup

- Archived (outside the repo) the greedy-era jobs `w0-mini-4b-mlx`, `w0-mini-9b-mlx`, `-s34`,
  `-retry-a`, `-retry-b` and their proxy logs (`local_smoke`, `mlx_4b_smoke`, `mlx_9b_mini`) to
  `~/guardfork_archive/2026-10-25-greedy-era/` (1.4 GB). Kept for guards-v1 provenance; their
  numbers above are invalid. Deleted `forks/guard_fork` (greedy 2/2 fork), `forks/replay_parity_smoke`
  (superseded by the 20/20 run), the viewer demo data, one-off queue logs, and harness-zero's
  `data/appworld-build` (regenerated by build.py).
- Label caveat: `w0-screen-9b-t06-s12` and `runs/w0/screen_nondev_tasks.txt` were named before the
  v2 split; that screen includes 7 scenarios that are now design. It is Mac pilot data only.
- The +2-seed Mac screen (`w0-screen-9b-t06-s34`) was cancelled: the Q1 failure pool and oracle
  continuations come from the GPU bf16 stack.
- Code: removed the one-off queue/resume scripts, `scripts/run_pilot.py` and `guardfork/fork/launch.py`
  (superseded by `fork/referee.py`); kept `pilot/gates.py`. Spec tests for agent-written code moved
  to `tests/test_*_spec.py`; only `bayes_prior` specs stay under `tests/handwritten/`.

### 2026-10-25 Multi-judge plumbing landed (commit 0278b57); open items

- VERIFY before the first judge call: OpenRouter id `qwen/qwen3-235b-a22b-2507` and prices; pin the
  dated Claude Sonnet snapshot via `GET /v1/models` (currently the alias `claude-sonnet-4-5`).
- [done 2026-10-25] `scripts/run_referee.py` does not yet pass judge picks as `extra_turns` (plumbing exists in
  `fork/referee.py`); wire it before the GPU referee uses judge windows.
- [done 2026-10-25: mandatory windows always run] `max_probes` bounds the total, so mandatory judge windows can be cut on long queues; raise it
  for the GPU run.
- A Claude-only probe's value does feed the oracle V̂ (it is the student's own continuation); only
  pair export excludes `eval_only` continuations.
- Judge commands now use design jobs (`jobs/w0-dev-9b-t06`); the w0-mini jobs are archived.

### 2026-10-25 Local referee smoke (diagnostic only)

- Two sampled 9B-4bit Mac failures were probed at K=4 to validate replay/referee plumbing. These
  are design diagnostics, not Q1 localization results.
- `c901732_3__8KZ792a`: first guard g=8, adjacent-rule t*=3. Values were strongly non-monotonic
  (including V(s1)=0/4 and V(s9)=3/4), demonstrating that K=4 is too noisy for scientific claims.
- `60d0b5b_1__pHRgiha`: first guard g=10, repaired adjacent-rule t*=14, largest drop=14; the
  previously interrupted turn-15 probe was deleted and rerun successfully, yielding 0/4.
- Operational result: replay-based probes and interrupted-job recovery work end to end. Scientific
  decision unchanged: use the preregistered K=8/K=16, change-point fit and split-half analysis on
  the matched bf16 GPU stack; do not interpret either smoke t* as evidence for or against guards.

### 2026-10-25 Referee smoke (Mac, 9B-4bit, K=4, 2 failures) and fixes

- Results and interpretation: see "Local referee smoke (diagnostic only)" above. For
  c901732 the K=4 curve recovers to 3/4 at t=9 after "collapsing" at t=4, so the v1
  adjacent-pair rule is unreliable at this K.
- Cost: 6157 + 4895 + 316 env-s (summed continuation wall time, ~1.6 h per failure; ~30-45 min
  elapsed per failure at concurrency 2). The ledger had
  charged 60d0b5b twice (5210 s on the rerun, incl. 4895 s of reused probes); corrected in
  `runs/cost/referee.jsonl` (backup `.bak-2026-10-25`), and `run_referee.py` no longer
  charges reused probe jobs.
- Both scenarios were evaluation-split: moved to design (PREREGISTRATION amendment 1; design
  17 / evaluation 32 scenarios).
- Referee fixes: `v_s1` pools the s_1 probe with the base runs (`v_s1_base` keeps the old
  value; for c901732 this moves tau from 0.25 to 0.107 but t*=3 stands, so the noise above,
  not the threshold, drove that label); all mandatory windows run regardless of
  `max_probes`; a probe with 0 graded continuations stops the trial with no oracle.json and
  its job is deleted for relaunch; Harbor output goes to `<forks-dir>/<job>.harbor.log`;
  judge picks are wired (`--judge-root`, `--judge-slots`); bisection plans only from
  non-Claude values and promotes Claude-only turns it asks for (no longer `eval_only`).
- Judge: non-JSON / truncated responses are recorded as errors instead of aborting the run;
  turns listed without `p` get min(leftover share, smallest stated p).
- Sampling: `config/sampling.toml` is now `qwen-thinking-code-v2` with `max_tokens = 4096` on
  both machines (0.4% of 8444 Mac 9B completions hit 4096; p99 1546). Runs before this entry
  used v1 (same values, cap applied by the proxy on mlx only).
