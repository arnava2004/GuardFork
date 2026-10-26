from __future__ import annotations

from pathlib import Path

from guardfork.q1.localization import build_rows, eligible_failed_runs, report_table
from guardfork.telemetry.trials import TrialRecord, TurnRecord


def _rec(task: str, reward: float, n: int = 4, name="trial0", job="j",
         invalid_at: int | None = None) -> TrialRecord:
    turns = [TurnRecord(i, f"cmd{i}", False) for i in range(n)]
    events = []
    for i in range(n):
        invalid = [{"e": 1}] if invalid_at == i + 1 else []
        events.append({
            "original": {"tool_call": {"command": f"cmd{i}"}, "content": "", "reasoning": ""},
            "context": [],
            "raw_original": {"tool_calls": [], "invalid_tool_calls": invalid},
        })
    return TrialRecord(Path(name), job, task, reward, None, 1.0, turns, events)


def test_eligible_failed_runs_uses_pass_band():
    recs = [_rec("aaaaaaa_1", 0.0), _rec("aaaaaaa_1", 1.0), _rec("bbbbbbb_1", 0.0)]
    counts = {"aaaaaaa_1": (3, 8), "bbbbbbb_1": (0, 8)}
    got = eligible_failed_runs(recs, counts, band=(0.3, 0.8), min_samples=8)
    assert [r.task_id for r in got] == ["aaaaaaa_1"]


def test_report_table_headline_excludes_undefined():
    rec = _rec("aaaaaaa_1", 0.0, invalid_at=2)
    rows = build_rows([rec], {"j/trial0": {"t_star": 2, "t_star_drop": 2}})
    assert rows[0].g == 2 and rows[0].scenario == "aaaaaaa" and rows[0].t_star == 2
    md = report_table(rows)
    assert "w=1" in md and "1/1" in md


def test_dev_set_excluded_from_eligibility():
    from guardfork.freeze import DevSet

    recs = [_rec("aaaaaaa_1", 0.0), _rec("bbbbbbb_1", 0.0, job="w0-mini-4b-mlx")]
    counts = {"aaaaaaa_1": (3, 8), "bbbbbbb_1": (4, 8)}
    dev = DevSet(frozenset({"w0-mini-4b-mlx"}), frozenset({"aaaaaaa"}))
    assert eligible_failed_runs(recs, counts, dev_set=dev) == []


def test_g3_variants_and_per_guard_rows():
    from guardfork.q1.localization import localizer_turns
    from guardfork.types import ToolCall, Trajectory, TrajectoryStatus, Turn

    def turn(t, cmd, obs="ok"):
        return Turn(t, 9 + t, "", "", (ToolCall("execute", {"command": cmd}),), (), obs)

    err = "Exception: Response status code is 401:"
    recovered = Trajectory("aaaaaaa_1", (
        turn(1, "print(apis.venmo.show_balance())", err),
        turn(2, "print(apis.api_docs.show_api_doc(app_name='venmo', api_name='login'))"),
        turn(3, "x = 1"), turn(4, "y = 2"),
    ), 0.0, TrajectoryStatus.FAILED)
    from guardfork.guards.rules import evaluate
    loc = localizer_turns(recovered, evaluate(recovered))
    assert loc["G3"] == 1 and loc["all"] == 1
    assert loc["no_g3"] is None and loc["g3_unrecovered"] is None

    unrecovered = Trajectory("aaaaaaa_1", (
        turn(1, "print(apis.venmo.show_balance())", err),
        turn(2, "x = 1"), turn(3, "y = 2"), turn(4, "z = 3"),
    ), 0.0, TrajectoryStatus.FAILED)
    assert localizer_turns(unrecovered, evaluate(unrecovered))["g3_unrecovered"] == 1


def test_report_has_per_localizer_section():
    rec = _rec("aaaaaaa_1", 0.0, invalid_at=2)
    md = report_table(build_rows([rec], {"j/trial0": {"t_star": 2}}))
    for loc in ("| all |", "| no_g3 |", "| g3_unrecovered |", "| G1 |", "guards: v1"):
        assert loc in md


def test_undefined_t_star_counted():
    rec = _rec("aaaaaaa_1", 0.0)
    rows = build_rows([rec], {"j/trial0": {"t_star": None}})
    md = report_table(rows)
    assert "t* undefined: 1/1" in md


def _judge(turn: int, T: int, p: float) -> dict:  # noqa: N803
    dist = [(1 - p) / (T - 1)] * T
    dist[turn - 1] = p
    return {"turn": turn, "distribution": dist}


def test_three_judge_rows_agreement_histogram_calibration(tmp_path):
    from guardfork.judge import write_judge
    from guardfork.q1.localization import (
        judge_agreement,
        judge_calibration,
        pick_histogram,
        pick_position,
    )

    recs = [_rec("aaaaaaa_1", 0.0, n=9, name=f"trial{i}", invalid_at=3) for i in range(3)]
    oracle = {f"j/trial{i}": {"t_star": 3} for i in range(3)}
    by_slot = {
        "judge_open": {"j/trial0": _judge(3, 9, 0.95), "j/trial1": _judge(4, 9, 0.6),
                       "j/trial2": _judge(9, 9, 0.75)},
        "judge_claude": {"j/trial0": _judge(3, 9, 0.8), "j/trial1": _judge(7, 9, 0.5)},
    }
    rows = build_rows(recs, oracle, judge_results_by_slot=by_slot)
    assert [r.localizers["judge_open"] for r in rows] == [3, 4, 9]
    assert [r.localizers["judge_claude"] for r in rows] == [3, 7, None]
    assert all(r.localizers["judge_local"] is None for r in rows)
    assert rows[0].judge_dists["judge_open"][2] == 0.95 and "judge_local" not in rows[0].judge_dists

    agree = judge_agreement(rows)
    assert agree[("judge_open", "judge_claude")] == (1, 2)  # (3,3) agree; (4,7) do not
    assert agree[("judge_open", "judge_local")] == (0, 0)
    assert pick_position(3, 9) == "first" and pick_position(4, 9) == "middle"
    assert pick_position(7, 9) == "last" and pick_position(9, 9) == "last"
    hist = pick_histogram(rows)
    assert hist["judge_open"] == {"n": 3, "first": 1, "middle": 1, "last": 1, "exact_last": 1}
    assert hist["judge_claude"] == {"n": 2, "first": 1, "middle": 0, "last": 1, "exact_last": 0}
    cal = judge_calibration(rows)
    # open: p=0.95 hit, p=0.6 hit (|4-3|<=1), p=0.75 miss (9 vs 3)
    assert cal["judge_open"] == [((0.0, 0.5), 0, 0), ((0.5, 0.7), 1, 1), ((0.7, 0.9), 0, 1),
                                 ((0.9, 1.0), 1, 1)]
    assert cal["judge_claude"] == [((0.0, 0.5), 0, 0), ((0.5, 0.7), 0, 1), ((0.7, 0.9), 1, 1),
                                   ((0.9, 1.0), 0, 0)]

    md = report_table(rows)
    assert "| judge_open | 3 | 1/3" in md and "| judge_claude | 2 | 1/2" in md
    assert "| judge_local | 0 |" in md and "| judge |" not in md
    assert "| judge_open vs judge_claude | 2 | 1/2 (0.50) |" in md
    assert "| judge_open | 3 | 1 | 1 | 1 | 1 |" in md
    assert "| judge_open | – | 1/1 (1.00) | 0/1 (0.00) | 1/1 (1.00) |" in md

    # the same via judge_<slot>.json files under a root; legacy judge.json appears as `judge`
    for slot, results in by_slot.items():
        for trial_id, entry in results.items():
            write_judge({"trial_id": trial_id, "judge_id": slot, **entry}, tmp_path)
    write_judge({"trial_id": "j/trial0", "turn": 2}, tmp_path)
    rows = build_rows(recs, oracle, judge_root=tmp_path)
    assert [r.localizers["judge_open"] for r in rows] == [3, 4, 9]
    assert rows[0].localizers["judge"] == 2 and "judge" not in rows[1].localizers
    md = report_table(rows)
    assert "| judge | 1 |" in md and "| judge vs judge_open | 1 | 1/1" in md  # |2-3| <= 1
