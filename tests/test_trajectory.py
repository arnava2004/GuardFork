from __future__ import annotations

from pathlib import Path

from guardfork.telemetry.to_trajectory import observation_after, to_trajectory
from guardfork.telemetry.trials import TrialRecord, TurnRecord, iter_trials
from guardfork.types import TrajectoryStatus


def test_observation_comes_from_next_event_context(make_trial, event):
    e1 = event("aaaaaaa_1", command="ls")
    e1["context"] = [{"type": "human", "content": "do it"}]
    e2 = event("aaaaaaa_1", command="cat")
    e2["context"] = [
        {"type": "human", "content": "do it"},
        {"type": "ai", "content": ""},
        {"type": "tool", "content": "file-list"},
    ]
    job = make_trial("j", "t0", "aaaaaaa_1", 0.0, [e1, e2])
    rec = next(iter_trials(job))
    traj = to_trajectory(rec)
    assert traj.T == 2 and traj.status is TrajectoryStatus.FAILED
    assert traj.turns[0].command == "ls"
    assert traj.turns[0].observation == "file-list"
    assert traj.turns[1].observation is None
    assert observation_after(rec.events, 0) == "file-list"


def test_session_reset_and_parse_survive_conversion(make_trial, event):
    e = event("aaaaaaa_1", extra={"metadata": {"session_reset": True}})
    job = make_trial("j", "t0", "aaaaaaa_1", 1.0, [e])
    rec = next(iter_trials(job))
    traj = to_trajectory(rec)
    assert traj.status is TrajectoryStatus.PASSED
    assert traj.turns[0].session_reset is True


def test_crashed_trial_is_error_status():
    rec = TrialRecord(Path("t"), "j", "aaaaaaa_1", None, {"type": "Boom"}, None, [], [])
    assert to_trajectory(rec).status is TrajectoryStatus.ERROR


def test_turn_limit_exception_counts_as_failure(make_trial, event):
    import json as _json

    job = make_trial("j", "t0", "aaaaaaa_1", None, [event("aaaaaaa_1")])
    result_path = job / "t0" / "result.json"
    data = _json.loads(result_path.read_text())
    data["exception_info"] = {"exception_type": "StudentTurnLimitError"}
    result_path.write_text(_json.dumps(data))
    rec = next(iter_trials(job))
    assert rec.reward == 0.0 and rec.hit_turn_limit
    assert to_trajectory(rec).status is TrajectoryStatus.MAX_TURNS


def test_max_turns_status():
    turns = [TurnRecord(i, "x", False) for i in range(40)]
    rec = TrialRecord(Path("t"), "j", "aaaaaaa_1", 0.0, None, 1.0, turns, [])
    assert to_trajectory(rec).status is TrajectoryStatus.MAX_TURNS
