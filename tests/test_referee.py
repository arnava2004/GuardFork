from __future__ import annotations

import json
from dataclasses import dataclass, field
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


def _fake_launch(true_t_star: int, tmp: Path, alive: float = 0.75):  # noqa: N803
    """Each continuation at s_t passes with rate `alive` if t <= t*, else 0."""
    launched: list[tuple[int, int, str]] = []

    def launch(plan_path: Path, k: int, job_name: str) -> Path:
        t = json.loads(plan_path.read_text())["fork_turn"]
        launched.append((t, k, job_name))
        passes = round(alive * k) if t <= true_t_star else 0
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


@dataclass
class _Fit:
    t_star: int | None
    lo: int | None
    hi: int | None
    sensitivity: dict = field(default_factory=dict)


def _step_oracle(values, *, v_s1, tau=0.5, **_):
    """Stand-in for bayes_prior.fit_changepoint_oracle: first probed t above tau * v_s1
    with every later probed value at or below it."""
    thr = tau * v_s1
    turns = sorted(values)
    for i, t in enumerate(turns):
        s, k = values[t]
        later = turns[i + 1:]
        if s / k > thr and later and all(values[u][0] / values[u][1] <= thr for u in later):
            return _Fit(t, t, t, {0.3: t, 0.7: t})
    return _Fit(None, None, None)


def _probed(launched):
    return sorted({t for t, _, _ in launched})


def test_referee_v2_screen_grid_refine(tmp_path):
    rec = _failed_run(T=12, guard_at=6)
    launch, launched = _fake_launch(true_t_star=4, tmp=tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path / "plans",
                      grid_step=3, oracle=_step_oracle)
    # grid {1, 4, 7, 10} + G1 window {5, 6, 7}; refinement brackets t*=4 -> {4, 5}
    assert _probed(launched) == [1, 4, 5, 6, 7, 10]
    assert res.probe_reasons[1][0] == "screen" and "grid" in res.probe_reasons[1]
    assert res.probe_reasons[6][0].startswith("guard:")
    assert res.values[4][1] == 16 and res.values[5][1] == 16 and res.values[7][1] == 8
    assert "refine" in res.probe_reasons[4] and "refine" in res.probe_reasons[5]
    assert res.t_star == 4 and (res.t_star_lo, res.t_star_hi) == (4, 4)
    assert res.t_star_v1 == 4 and not res.refine_pending and not res.hopeless
    assert res.v_s1 == (4 + 6 + 0.5) / (8 + 8 + 1)
    out = write_results(res, tmp_path / "results")
    oracle = json.loads((out / "oracle.json").read_text())
    assert oracle["protocol"] == "v2" and oracle["t_star"] == 4 and oracle["guards_version"] == "v1"
    assert len(json.loads((out / "forks.json").read_text())) == sum(k for _, k, _ in launched)


def test_referee_v2_hopeless_after_confirmation(tmp_path):
    rec = _failed_run(T=12, guard_at=6)
    launch, launched = _fake_launch(true_t_star=0, tmp=tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path / "plans",
                      oracle=_step_oracle)
    assert [(t, k) for t, k, _ in launched] == [(1, 8), (1, 8)]
    assert res.hopeless and res.values[1] == (0, 16) and res.t_star is None


def test_referee_v2_probes_runs_without_guard(tmp_path):
    rec = _failed_run(T=9, guard_at=99)
    launch, launched = _fake_launch(true_t_star=5, tmp=tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path / "plans",
                      grid_step=3, oracle=_step_oracle)
    assert res.g is None and res.skipped is None
    # grid {1, 4, 7}; the step fit puts t* at the last alive probed turn, 4; refine {4, 7}
    assert _probed(launched) == [1, 4, 7]
    assert {t for t, _, j in launched if j.endswith("-refine")} == {4, 7}
    assert res.t_star == 4


def test_referee_v2_with_real_changepoint_oracle(tmp_path):
    rec = _failed_run(T=12, guard_at=6)
    launch, launched = _fake_launch(true_t_star=4, tmp=tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path / "plans")
    # grid {1, 4, 7, 10} + G1 window {5, 6, 7}: alive through 4, dead from 5; refine {4, 5}
    assert not res.refine_pending and res.t_star == 4 and res.t_star_v1 == 4
    assert {t for t, _, j in launched if j.endswith("-refine")} == {4, 5}
    assert res.t_star_lo <= 4 <= res.t_star_hi
    assert set(res.t_star_sensitivity) == {0.3, 0.7}


def test_referee_v2_refine_pending_when_oracle_unavailable(tmp_path):
    def pending(*_a, **_k):
        raise NotImplementedError

    rec = _failed_run(T=12, guard_at=6)
    launch, launched = _fake_launch(true_t_star=4, tmp=tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path / "plans",
                      oracle=pending)
    assert res.refine_pending and res.t_star is None
    assert all(k == 8 for _, k, _ in launched) and res.t_star_v1 == 4


def test_referee_v2_claude_windows_are_eval_only_and_never_refined(tmp_path):
    rec = _failed_run(T=14, guard_at=3)
    launch, launched = _fake_launch(true_t_star=8, tmp=tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path / "plans",
                      grid_step=4, extra_turns={"judge_claude": [11], "judge_open": [None]},
                      oracle=_step_oracle)
    # grid {1, 5, 9, 13}, G1 window {2, 3, 4}, Claude window {10, 11, 12}
    assert res.probe_reasons[10] == ["judge_claude"] and res.probe_reasons[12] == ["judge_claude"]
    eval_turns = {c["turn"] for c in res.continuations if c["eval_only"]}
    assert eval_turns == {10, 11, 12}
    # planning ignores Claude values: t* = 5 from {1..5 alive, 9 dead}, bracket {5, 9}
    assert {t for t, _, j in launched if j.endswith("-refine")} == {5, 9}
    assert res.t_star == 5


def test_referee_v2_failed_probe_stops_without_label(tmp_path):
    rec = _failed_run(T=8, guard_at=6)
    launched = []

    def empty_launch(plan_path, k, job_name):
        launched.append(job_name)
        job = tmp_path / job_name
        job.mkdir(parents=True)
        return job

    res = run_referee(rec, base_pass=(4, 8), K=4, launch=empty_launch,
                      plans_dir=tmp_path / "plans", oracle=_step_oracle)
    assert res.failed_probes == [1] and len(launched) == 1
    assert res.t_star is None and res.t_star_v1 is None and "no graded" in res.skipped


def test_referee_v2_dense_probes_every_forkable_turn(tmp_path):
    rec = _failed_run(T=7, guard_at=3)
    launch, launched = _fake_launch(true_t_star=4, tmp=tmp_path)
    res = run_referee(rec, base_pass=(4, 8), K=8, launch=launch, plans_dir=tmp_path / "plans",
                      dense=True, oracle=_step_oracle)
    assert _probed(launched) == [1, 2, 3, 4, 5, 6] and res.dense and res.t_star == 4
