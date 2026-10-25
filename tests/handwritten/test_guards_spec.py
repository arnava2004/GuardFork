"""Specs for guards/rules.py and fork/select.py (Arnav implements)."""

import pytest

from guardfork.fork.select import forkable_turns, guard_fork_turn
from guardfork.guards.rules import Guard, GuardFiring, check_turn, evaluate, first_nonterminal
from guardfork.types import ToolCall, Trajectory, TrajectoryStatus, Turn

pytestmark = pytest.mark.handwritten


def _turn(t: int, *, content="", reasoning="", command: str | None = "ls", invalid=(),
          observation="ok", session_reset=False) -> Turn:
    calls = (ToolCall("execute", {"command": command}),) if command else ()
    return Turn(turn=t, assistant_index=9 + t, content=content, reasoning=reasoning,
                tool_calls=calls, invalid_tool_calls=tuple(invalid), observation=observation,
                session_reset=session_reset)


def _traj(*turns: Turn) -> Trajectory:
    return Trajectory("aaaaaaa_1", tuple(turns), 0.0, TrajectoryStatus.FAILED)


def _guards(prefix, traj, **kw):
    return {f.guard for f in check_turn(prefix, traj, **kw)}


def test_g1_invalid_tool_calls():
    t = _turn(1, invalid=[{"name": "execute", "error": "bad json"}])
    assert Guard.G1_INVALID_ACTION in _guards((t,), _traj(t))


def test_g1_code_in_visible_content():
    t = _turn(1, content="```python\nprint(1)\n```", command=None)
    assert Guard.G1_INVALID_ACTION in _guards((t,), _traj(t))


def test_g1_reasoning_code_only_in_sensitivity_mode():
    t = _turn(1, reasoning="```python\nx=1\n```")
    assert Guard.G1_INVALID_ACTION not in _guards((t,), _traj(t))
    assert Guard.G1_INVALID_ACTION in _guards((t,), _traj(t), scan_reasoning_for_code=True)


def test_g1_tool_markup_in_reasoning():
    t = _turn(1, reasoning="<tool_call>{...}</tool_call>")
    assert Guard.G1_INVALID_ACTION in _guards((t,), _traj(t))


def test_g5_suppressed_by_g1():
    t = _turn(1, content="```python\nprint(1)\n```", command=None)
    got = _guards((t,), _traj(t))
    assert Guard.G5_UNVERIFIED_COMPLETION not in got


def test_demo_turns_never_fire():
    demo = Turn(turn=0, assistant_index=3, content="```x```", reasoning="", tool_calls=())
    assert check_turn((demo,), _traj(demo)) == []


def test_no_lookahead():
    a, b = _turn(1), _turn(2, invalid=[{"error": "x"}])
    assert check_turn((a,), _traj(a, b)) == check_turn((a,), _traj(a))


def test_evaluate_ordered_and_first_nonterminal():
    traj = _traj(_turn(1), _turn(2, invalid=[{"e": 1}]), _turn(3, command=None))
    firings = evaluate(traj)
    assert [f.turn for f in firings] == sorted(f.turn for f in firings)
    first = first_nonterminal(firings, traj.T)
    assert first is not None and first.turn == 2


def test_first_nonterminal_excludes_terminal_and_heldout():
    f = [GuardFiring(Guard.G3_FAILED_PRECONDITION, 2, ""),
         GuardFiring(Guard.G5_UNVERIFIED_COMPLETION, 5, "")]
    assert first_nonterminal(f, 5, exclude=frozenset({Guard.G3_FAILED_PRECONDITION})) is None


def test_forkable_stops_after_session_reset_and_subagent():
    traj = _traj(_turn(1), _turn(2, session_reset=True), _turn(3), _turn(4))
    assert forkable_turns(traj) == [1, 2]
    traj = _traj(_turn(1), _turn(2, command="agent 'sub task'"), _turn(3), _turn(4))
    assert forkable_turns(traj) == [1, 2]


def test_guard_fork_turn_is_first_nonterminal_firing():
    traj = _traj(_turn(1), _turn(2, invalid=[{"e": 1}]), _turn(3), _turn(4, command=None))
    assert guard_fork_turn(traj, evaluate(traj)) == 2
