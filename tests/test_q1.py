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
    dev = DevSet(frozenset({"w0-mini-4b-mlx"}), frozenset({"aaaaaaa_1"}))
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
