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
  both continuations passed (2/2). State s_3 was recoverable; the original went wrong at/after t=3.
