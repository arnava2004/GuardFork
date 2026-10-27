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


def test_referee_records_probe_reasons_and_tags_eval_only(tmp_path):
    rec = _failed_run(T=12, guard_at=4)
    launch, launched = _fake_launch(true_t_star=2, tmp=tmp_path, K=4)
    res = run_referee(rec, base_pass=(4, 8), K=4, launch=launch, plans_dir=tmp_path / "plans",
                      max_probes=12, extra_turns={"judge_open": [5], "judge_claude": [10]})
    # initial window {1, g-3=1, g-1=3, g=4, g+1=5}; judge_open adds {4,5,6}; claude {9,10,11}
    assert res.probe_reasons[1] == ["initial"]
    assert res.probe_reasons[4] == ["initial", "guard", "judge_open"]
    assert res.probe_reasons[5] == ["initial", "judge_open"]
    assert res.probe_reasons[6] == ["judge_open"]
    for t in (9, 10, 11):
        assert res.probe_reasons[t] == ["judge_claude"]
    bisected = [t for t, r in res.probe_reasons.items() if r == ["bisection"]]
    assert bisected, "bisection should still refine between the mandatory windows"
    conts = res.continuations
    claude_only = [c for c in conts if c["eval_only"]]
    assert {c["turn"] for c in claude_only} == {9, 10, 11}
    assert all(c["reasons"] == ["judge_claude"] for c in claude_only)
    assert all(not c["eval_only"] for c in conts if c["turn"] not in (9, 10, 11))
    out = write_results(res, tmp_path / "results")
    oracle = json.loads((out / "oracle.json").read_text())
    assert oracle["probe_reasons"]["10"] == ["judge_claude"]
    assert oracle["n_eval_only"] == 3 * 4
    forks = json.loads((out / "forks.json").read_text())
    assert sum(f["eval_only"] for f in forks) == 12 and all("reasons" in f for f in forks)
    assert res.t_star == 2


def test_referee_extra_turns_shared_with_guard_window_are_not_eval_only(tmp_path):
    rec = _failed_run(T=8, guard_at=4)
    launch, _ = _fake_launch(true_t_star=3, tmp=tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path / "plans",
                      extra_turns={"judge_claude": [4]})
    assert res.probe_reasons[4] == ["initial", "guard", "judge_claude"]
    assert not any(c["eval_only"] for c in res.continuations)


def test_referee_flags_probes_without_graded_continuations(tmp_path):
    rec = _failed_run(T=8, guard_at=6)

    def empty_launch(plan_path, k, job_name):
        job = tmp_path / job_name
        job.mkdir(parents=True)
        return job

    res = run_referee(rec, base_pass=(4, 8), K=4, launch=empty_launch,
                      plans_dir=tmp_path / "plans")
    assert res.skipped and "no graded continuations" in res.skipped


def test_referee_skips_runs_without_nonterminal_guard(tmp_path):
    rec = _failed_run(T=5, guard_at=99)
    launch, launched = _fake_launch(1, tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path)
    assert res.skipped and launched == []


def test_referee_failed_probe_stops_without_oracle_label(tmp_path):
    rec = _failed_run(T=8, guard_at=6)
    launched = []

    def empty_launch(plan_path, k, job_name):
        launched.append(job_name)
        job = tmp_path / job_name
        job.mkdir(parents=True)
        return job

    res = run_referee(rec, base_pass=(4, 8), K=4, launch=empty_launch,
                      plans_dir=tmp_path / "plans")
    assert res.failed_probes == [1] and len(launched) == 1
    assert res.t_star is None and res.t_star_drop is None


def test_referee_pools_s1_probe_into_v_s1(tmp_path):
    rec = _failed_run(T=8, guard_at=6)
    launch, _ = _fake_launch(true_t_star=3, tmp=tmp_path, K=4)
    res = run_referee(rec, base_pass=(1, 2), K=4, launch=launch, plans_dir=tmp_path / "plans")
    assert res.v_s1_base == (1 + 0.5) / (2 + 1)
    # s_1 probe is 4/4 under the fake launcher: pooled Jeffreys (1 + 4 + 0.5) / (6 + 1)
    assert res.v_s1 == (1 + 4 + 0.5) / (2 + 4 + 1)


def test_referee_bisection_ignores_claude_values_and_promotes_requested_turns(tmp_path):
    rec = _failed_run(T=12, guard_at=4)
    launch, launched = _fake_launch(true_t_star=8, tmp=tmp_path, K=4)
    res = run_referee(rec, base_pass=(4, 8), K=4, launch=launch, plans_dir=tmp_path / "plans",
                      max_probes=12, extra_turns={"judge_claude": [9]})
    # Bisection from the non-Claude values {1, 3, 4, 5} asks for 8, then 10, then 9: all
    # already probed for Claude, so they are promoted instead of re-launched.
    assert sorted(launched) == sorted(set(launched))
    for t in (8, 9, 10):
        assert res.probe_reasons[t] == ["judge_claude", "bisection"]
    assert not any(c["eval_only"] for c in res.continuations)
    assert res.t_star == 8
