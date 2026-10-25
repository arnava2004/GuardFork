# Rules for Agents

Plan: `guardfork.md`. Kill thresholds: `config/kill.toml`. Measurements and decisions: `NOTES.md`.

1. **Arnav hand-writes:** `guardfork/guards/rules.py` (G1–G5), `guardfork/fork/select.py`, `guardfork/fork/probe_plan.py`, `guardfork/train/losses.py`, and everything in `guardfork/stats/` (including `stats/oracle.py`). Agents write `NotImplementedError` stubs, docstrings, and failing tests (`@pytest.mark.handwritten`) only.
2. **Loader-level split lock:** `test_normal` loads only with `allow_final=True` (`--final`). Never work around it.
3. **No API teachers or API-generated data.** Open-weight models only.
4. **No Google code, data, or results. No plaintext AppWorld task data in git.** The pre-commit hook (`git config core.hooksPath hooks`) blocks `runs/`, `results/`, `forks/`, `teacher/`, `jobs/`, `appworld_initial_context.json`, `*.tar`, `*.jsonl`, and AppWorld task text. Hold any public push or post until outside-work approval clears.
5. **Reproducibility:** every trial logs `commit_sha`, `config_hash`, `harness_hash` (incl. compose hash and `PYTHONHASHSEED=0`), `model_id`, `backend`, `train_seed`, `eval_seed`, `env_seconds`, `container_limits`.
6. **Never modify AppWorld or Harbor verifiers / task checkers.**
7. **Measure before building:** no Weekend 1+ framework code until Weekend 0 numbers in `NOTES.md` pass `config/kill.toml`. Show a plan before writing code, and ask before adding dependencies.
