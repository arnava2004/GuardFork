"""Specs for agent-written Weekend 1–4 stubs. Deselected by default; run with -m pending."""

from pathlib import Path

import pytest

from guardfork.config import load_kill_config
from guardfork.cost import CostEntry, CostLedger
from guardfork.fork.harbor_replay import NotForkable, build_replay_plan
from guardfork.fork.launch import ContinuationResult, value_estimate
from guardfork.pilot.gates import PilotRun, w1d_recoverable_guard_states, w1e_oracle_test_retest
from guardfork.telemetry.trials import TrialRecord, TurnRecord

pytestmark = pytest.mark.pending


def _record(n: int, reset_at: int | None = None) -> TrialRecord:
    turns = [TurnRecord(index=i, command=f"cmd{i}", parse_failure=False,
                        session_reset=(reset_at == i + 1)) for i in range(n)]
    events = [{"original": {"tool_call": {"command": f"cmd{i}"}}, "context": []}
              for i in range(n)]
    return TrialRecord(Path("t"), "j", "aaaaaaa_1", 0.0, None, 1.0, turns, events)


def test_replay_plan_has_t_minus_1_commands():
    plan = build_replay_plan(_record(6), 4)
    assert plan.commands == ("cmd0", "cmd1", "cmd2") and plan.max_turns == 37


def test_replay_plan_rejects_after_session_reset():
    with pytest.raises(NotForkable):
        build_replay_plan(_record(6, reset_at=2), 4)


def test_value_estimate_excludes_ungraded():
    res = [ContinuationResult(4, k, Path("x"), r, "ok", 1.0, None)
           for k, r in enumerate([1.0, 0.0, None, 1.0])]
    assert value_estimate(res) == (2, 3)


def test_cost_ledger_roundtrip(tmp_path):
    ledger = CostLedger(tmp_path / "cost.jsonl")
    ledger.charge(CostEntry("pilot", "t0", 60.0, 30.0, 1000, 100))
    assert CostLedger.load(tmp_path / "cost.jsonl").totals("pilot")["env_seconds"] == 60.0


def _run(values, t_star=None, retest=None, g=3):
    return PilotRun("r", "s", 8, False, g, values, t_star, retest)


def test_w1d_and_w1e():
    kill = load_kill_config()
    runs = [_run({3: (2, 8)}), _run({3: (0, 8)}), _run({3: (1, 8)})]
    assert w1d_recoverable_guard_states(runs, kill).value == pytest.approx(2 / 3)
    runs = [_run({}, 3, 4), _run({}, 3, 6), _run({}, None, 2)]
    assert w1e_oracle_test_retest(runs, kill).value == pytest.approx(0.5)
