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


def test_undefined_t_star_counted():
    rec = _rec("aaaaaaa_1", 0.0)
    rows = build_rows([rec], {"j/trial0": {"t_star": None}})
    md = report_table(rows)
    assert "t* undefined: 1/1" in md
