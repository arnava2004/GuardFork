# Reading & audit checklist

Ordered by when you need it. Tick boxes as you go. The 26xx arXiv ids came from our earlier
search; open each link once to confirm it is the paper you think it is.

## 1. Before auditing anything (≈ 1 evening)

- [ ] `AGENTS.md` and `guardfork.md` in this repo: the rules and the plan, in your own words.
- [ ] `config/kill.toml`: for each threshold, can you say what you would do if it fails?
- [ ] **AppWorld paper** — [arXiv 2407.18901](https://arxiv.org/abs/2407.18901). Sections on
      tasks, scenarios, the API docs, and TGC/SGC. Skip the leaderboard.
- [ ] **Harness-Zero paper** — [arXiv 2609.24974](https://arxiv.org/abs/2609.24974). How h, h*,
      and the reviewer work; the AppWorld numbers (26.8 bare h for 9B is our K0 anchor).

## 2. Harness-Zero code (the system we run, never modify its verifier)

Clone it and read these, in this order. For each, write one line: "what it does, what file
it writes".

- [ ] `README.md`, `envs/appworld/README.md`
- [ ] `src/harness_zero/review.py` — `AssistantResponse`, `ReviewCandidate` (`raw_original`)
- [ ] `src/harness_zero/store.py` — `TrialStore`, `trajectory.jsonl`, `result.json`
- [ ] `src/harness_zero/dataset_utils.py` — where Harbor puts `result.json`
- [ ] the student wrapper / middleware module (search for `PassthroughReviewer`,
      `ReasoningChatOpenAI`, `_preserve_history`, `student_model_requests`)
- [ ] the CLI (`run-rollout --help`): find the passthrough-reviewer flag and test split filename
      (both marked unverified in `scripts/w0_runbook.md`)
- [ ] `envs/appworld/agent.py`, `common_context.py` — the 23 demo messages, `initialize`
- [ ] `envs/appworld/harbor_environment.py` — container lifecycle, network settings
- [ ] `envs/appworld/runtime/server.py`, `client.py`, `grade.py`, `snapshot.py` — state, grading
- [ ] `envs/appworld/build_sft.py` — `canonical`, masking of demo turns / reasoning
- [ ] [Harbor](https://github.com/harbor-framework/harbor) docs: jobs, trials, `result.json` schema

## 3. Models, serving, and "is anything calling a paid API?"

- [ ] Qwen3.5-4B / 9B model cards on Hugging Face: context length, thinking mode, tool-call format.
- [ ] Their `tokenizer_config.json` `chat_template`: how thinking and tool calls are rendered.
- [ ] vLLM docs: OpenAI-compatible server, tool calling (`--tool-call-parser`), reasoning parser,
      `return_token_ids`, prefix caching. Confirm the flags in `scripts/w0_runbook.md` exist in
      the version you install.
- [ ] tinker-cookbook renderers (`renderers/qwen3_5.py`, `base.py`): `Qwen3_5Renderer`,
      `strip_thinking_from_history`, `build_generation_prompt`. This is what the parity check uses.
- [ ] mlx-lm: `mlx_lm.server` (OpenAI-compatible, used only for local smoke tests) and
      `mlx_lm.lora` (only matters if K4 says Mac training is feasible).
- [ ] **No-API audit** (rule 3). Run and read the output:
      `rg -n "OPENAI_API_KEY|api\.openai\.com|openrouter|azure|anthropic|TM_API_KEY" .`
      in both repos and in your `.env`. The student must point at your proxy/vLLM; no teacher
      model may be configured. Tinker is hosted training of open weights: allowed by rule 3, but
      decide before Weekend 5 whether you want it.

## 4. Before writing guards and the oracle (Weekends 1–2)

- [ ] **2610.11655** — first-firing signals labelling whole trajectories (our closest neighbour;
      "the route we do not take"). [arXiv](https://arxiv.org/abs/2610.11655)
- [ ] **Fission-GRPO** — branching after execution errors. [arXiv 2601.15625](https://arxiv.org/abs/2601.15625)
- [ ] **DepGPO** — verifier read-set dependency tracing. [arXiv 2610.03634](https://arxiv.org/abs/2610.03634)
- [ ] **HarnessRL** — guard penalties as reward. [arXiv 2610.26426](https://arxiv.org/abs/2610.26426)
- [ ] **ELPO** — MC rollout trees for step localization. [arXiv 2602.09598](https://arxiv.org/abs/2602.09598)
- [ ] **Math-Shepherd** — MC continuation labels for steps (the oracle idea). [arXiv 2312.08935](https://arxiv.org/abs/2312.08935)
- [ ] **VinePPO** — MC value estimates from intermediate states. [arXiv 2410.01679](https://arxiv.org/abs/2410.01679)
- [ ] Spec tests: `tests/test_guards_spec.py`, `tests/test_oracle_spec.py`, and
      `tests/test_bayes_prior_spec.py` (bayes_prior was agent-implemented on 2026-11-05 at your
      direction; read it against these specs). **I wrote these from the plan; check that each
      one says what you mean** (especially G2, G5, t*, and forkability) before implementing.

## 5. Before the Q1 statistics (Weekends 3–4)

- [ ] Brown, Cai & DasGupta (2001), *Interval Estimation for a Binomial Proportion* — Jeffreys.
- [ ] Efron & Tibshirani, *An Introduction to the Bootstrap*, ch. 1–6 (percentile intervals).
- [ ] A cluster-bootstrap reference (resample scenarios, not tasks), e.g. Field & Welsh (2007).
- [ ] Miller (2024), *Adding Error Bars to Evals* — clustered SEs for evals. [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)
- [ ] Holm (1979) step-down correction; McNemar's test (secondary only).
- [ ] Power/MDE basics: why n≈90 runs gives a ~16–20 pp MDE.

## 6. Before training (Weekends 5–7)

- [ ] **DPO** — [arXiv 2305.18290](https://arxiv.org/abs/2305.18290). Derive the loss once by hand.
- [ ] **Step-DPO** — turn/step-level DPO. [arXiv 2406.18629](https://arxiv.org/abs/2406.18629)
- [ ] **SDPO** — step-level DPO for agents with judge localization. [arXiv 2501.01821](https://arxiv.org/abs/2501.01821)
- [ ] **CSO** — PRM + expert rewrites. [arXiv 2602.03412](https://arxiv.org/abs/2602.03412)
- [ ] **SWEET-RL** — turn-level credit for multi-turn agents. [arXiv 2503.15478](https://arxiv.org/abs/2503.15478)
- [ ] **GiGPO** — step-level groups from shared states. [arXiv 2505.10978](https://arxiv.org/abs/2505.10978)
- [ ] **STaR / rejection-sampling fine-tuning** (our RFT arm). [arXiv 2203.14465](https://arxiv.org/abs/2203.14465)
- [ ] LoRA basics + whichever trainer you pick (Tinker recipe in Harness-Zero, TRL, or mlx-lm).

## 7. Auditing this repo, file by file

For each file: does it do what its docstring says, does it ever write outside `runs/`,
`results/`, `forks/`, `jobs/`, and could it leak AppWorld text into git?

- [ ] `hooks/pre-commit` — try committing a `runs/` file and a file containing task text.
- [ ] `config/dev_set.toml` + `guardfork/freeze.py` — 17 design / 32 evaluation scenarios (amendment 1)
      (the binding split, see `PREREGISTRATION.md`); `guardfork/splits.py` — `test_normal`
      needs `allow_final=True`.
- [ ] `guardfork/telemetry/trials.py` — matches `store.py` / Harbor layouts from section 2.
- [ ] `guardfork/telemetry/to_trajectory.py` — observation of turn t comes from event t+1.
- [ ] `guardfork/serve/proxy.py` — never changes the response; what it injects into requests.
- [ ] `guardfork/serve/parity.py` — message conversion matches tinker's `Message` format.
- [ ] `guardfork/measure/w0.py` — TGC/SGC definitions match the AppWorld paper.
- [ ] `tools/viewer/` — binds 127.0.0.1 only, read-only:
      `uv run python tools/viewer/server.py` (indexes `jobs/` and `forks/`).
- [ ] Forking and the referee (`fork/harbor_replay.py`, `fork/referee.py`,
      `fork/replay_checks.py`), `q1/` (localization, simulate_search, triage), `train/`
      (credit, prefix_drift, export, pairs), `judge.py`, `sampling.py`, `cost.py`, `manifest.py`.
- [ ] `uv run pytest` (all specs, bayes_prior included).
