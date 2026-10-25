from __future__ import annotations

import json
from pathlib import Path

from guardfork.fork.referee import run_referee, write_results
from guardfork.telemetry.trials import TrialRecord, TurnRecord


def _failed_run(T: int = 8, guard_at: int = 4) -> TrialRecord:  # noqa: N803
    events, ctx = [], [{"type": "human", "content": "task"}]
    for i in range(T):
        invalid = [{"e": 1}] if i + 1 == guard_at else []
        events.append({"context": list(ctx),
                       "original": {"tool_call": {"name": "execute", "command": f"c{i}"},
                                    "content": "", "reasoning": ""},
                       "raw_original": {"tool_calls": [], "invalid_tool_calls": invalid}})
        ctx += [{"type": "ai", "content": "", "tool_calls": [
            {"name": "execute", "args": {"command": f"c{i}"}, "id": f"i{i}",
             "type": "tool_call"}]}, {"type": "tool", "content": f"o{i}", "tool_call_id": f"i{i}"}]
    turns = [TurnRecord(i, f"c{i}", False) for i in range(T)]
    return TrialRecord(Path("trialX"), "evaljob", "aaaaaaa_1", 0.0, None, 60.0, turns, events)


def _fake_launch(true_t_star: int, tmp: Path, K: int = 8):  # noqa: N803
    launched: list[int] = []

    def launch(plan_path: Path, k: int, job_name: str) -> Path:
        t = json.loads(plan_path.read_text())["fork_turn"]
        launched.append(t)
        passes = 6 if t <= true_t_star else 0
        job = tmp / job_name
        for j in range(k):
            trial = job / f"aaaaaaa_1__{j}"
            trial.mkdir(parents=True)
            (trial / "config.json").write_text("{}")
            (trial / "result.json").write_text(json.dumps(
                {"task_name": "aaaaaaa_1",
                 "verifier_result": {"rewards": {"reward": 1.0 if j < passes else 0.0}}}))
        return job

    return launch, launched


def test_referee_bisects_to_collapse_turn(tmp_path):
    rec = _failed_run(T=8, guard_at=6)
    launch, launched = _fake_launch(true_t_star=3, tmp=tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path / "plans")
    assert res.g == 6
    assert res.t_star == 3 and res.values[3] == (6, 8) and res.values[4] == (0, 8)
    assert sorted(launched) == sorted(set(launched))
    out = write_results(res, tmp_path / "results")
    oracle = json.loads((out / "oracle.json").read_text())
    assert oracle["t_star"] == 3 and oracle["guards_version"] == "v1"
    assert len(json.loads((out / "forks.json").read_text())) == 8 * len(launched)


def test_referee_skips_runs_without_nonterminal_guard(tmp_path):
    rec = _failed_run(T=5, guard_at=99)
    launch, launched = _fake_launch(1, tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path)
    assert res.skipped and launched == []
