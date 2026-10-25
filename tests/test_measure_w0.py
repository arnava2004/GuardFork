from __future__ import annotations

import json

import pytest

from guardfork.config import load_kill_config
from guardfork.measure import w0
from guardfork.telemetry.proxy_log import ProxyRecord
from guardfork.telemetry.trials import iter_trials


@pytest.fixture
def kill():
    return load_kill_config()


def test_kill_config_loads(kill):
    assert kill.weekend_0.k0_sgc_range == (18.0, 36.0)
    assert kill.weekend_1_pilot["W1a_min_replayable_fraction"] == 0.85
    assert len(kill.sha256) == 64


def test_trial_parsing(make_trial, event):
    job = make_trial("j0", "t0", "aaaaaaa_1", 1.0, [
        event("aaaaaaa_1"),
        event("aaaaaaa_1", command=None, content="<tool_call>{}</tool_call>"),
        event("aaaaaaa_1", invalid=[{"name": "execute"}]),
        event("aaaaaaa_1", command="agent 'do the thing'"),
        event("aaaaaaa_1", extra={"metadata": {"session_reset": True}}),
    ], wall_s=90)
    (t,) = list(iter_trials(job))
    assert t.task_id == "aaaaaaa_1" and t.passed and t.wall_seconds == 90
    assert [tr.parse_failure for tr in t.turns] == [False, True, True, False, False]
    assert t.turns[3].subagent_call and not t.turns[0].subagent_call
    assert t.turns[4].session_reset


def test_crashed_trial_without_teacher_logs(make_trial):
    job = make_trial("j0", "t0", "aaaaaaa_1", None, [], write_teacher=False)
    (t,) = list(iter_trials(job))
    assert t.reward is None and w0.status_of(t) == "error"


def test_tgc_sgc_and_pass_band(make_trial, event, kill):
    # job s0: scenario a all pass, scenario b one fail. job s1: everything fails.
    tids = ["aaaaaaa_1", "aaaaaaa_2", "aaaaaaa_3", "bbbbbbb_1", "bbbbbbb_2", "bbbbbbb_3"]
    job_dirs = set()
    for job, rewards in (("s0", [1, 1, 1, 1, 0, 1]), ("s1", [0] * 6)):
        for i, (tid, r) in enumerate(zip(tids, rewards, strict=True)):
            job_dirs.add(make_trial(job, f"t{i}", tid, float(r), [event(tid)]))
    trials = [t for j in sorted(job_dirs) for t in iter_trials(j)]
    assert w0.tgc(trials) == pytest.approx(100 * 5 / 12)
    assert w0.sgc(trials) == pytest.approx((50.0 + 0.0) / 2)
    counts = w0.per_task_pass(trials)
    assert counts["aaaaaaa_1"] == (1, 2)
    frac, n_in, n = w0.pass_band_fraction(counts, (0.2, 0.8), min_samples=2)
    assert (n_in, n) == (5, 6) and frac == pytest.approx(5 / 6)


def test_second_stage_selection():
    counts = {"a_1": (0, 4), "a_2": (2, 4), "a_3": (4, 4), "b_1": (3, 4), "b_2": (1, 8)}
    assert w0.second_stage_tasks(counts) == ["a_2", "b_1"]


def _proxy(n_assistant: int, prompt_len: int, tool_call: bool = True) -> ProxyRecord:
    msgs = [{"role": "assistant", "content": ""}] * n_assistant
    choice = {"finish_reason": "tool_calls" if tool_call else "stop", "message": {}}
    return ProxyRecord({"status": 200, "request": {"messages": msgs},
                        "prompt_token_ids": list(range(prompt_len)),
                        "response": {"choices": [choice]},
                        "usage": {"completion_tokens": 5}})


def test_summary_and_kill(make_trial, event, kill):
    job = make_trial("s0", "t0", "aaaaaaa_1", 0.0,
                     [event("aaaaaaa_1"), event("aaaaaaa_1", issue="bad")])
    trials = list(iter_trials(job))
    proxy = [_proxy(10, 4000), _proxy(11, 6000), _proxy(12, 9000, tool_call=False)]
    s = w0.summarize(trials, proxy, kill)
    assert s.parse_failure_rate == 0.5
    assert s.prompt_tokens_by_turn_p50 == {0: 4000, 1: 6000, 2: 9000}
    assert s.completion_tokens_total == 15
    res = {r.name.split()[0]: r for r in w0.evaluate_kill(s, kill, is_k0_run=False,
                                                          k4_prompt_tokens=5000)}
    assert res["K2"].passed is False
    assert res["K4"].passed is True
    assert res["K1"].passed is False
    md = w0.render_markdown("t", s, list(res.values()), {"commit": "x"})
    assert "**FAIL**" in md and json.dumps(md)


def test_k0_range(kill):
    s = w0.W0Summary(0, 0, None, 25.0, (None, 0, 0), None, None, None, None, {}, 0, None, {},
                     0, 0, 0, None, None)
    (k0, *_rest) = w0.evaluate_kill(s, kill, is_k0_run=True)
    assert k0.passed is True
