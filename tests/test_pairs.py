from __future__ import annotations

import math

import pytest

from guardfork.guards.ast_sig import api_signature
from guardfork.train.losses import dpo_loss
from guardfork.train.pairs import (
    Action,
    gf_dpo_pairs,
    gf_eligible_failures,
    grew_pairs,
    rft_examples,
    tdpo_pairs,
)
from guardfork.types import ToolCall, Trajectory, TrajectoryStatus, Turn


def _turn(t, command="print(apis.venmo.list_transactions())", invalid=()):
    calls = (ToolCall("execute", {"command": command}),) if command else ()
    return Turn(t, 9 + t, "", "", calls, tuple(invalid), "ok")


def _traj(task, *turns, reward=0.0, status=TrajectoryStatus.FAILED):
    return Trajectory(task, tuple(turns), reward, status)


def test_api_signature_extracts_app_endpoint_kwargs():
    cmd = "print(apis.venmo.create_transaction(receiver_email='r@x.com', amount=42))"
    assert api_signature(cmd) == (("venmo", "create_transaction",
                                   ("amount", "receiver_email")),)


def test_rft_keeps_only_passes():
    win = _traj("a_1", _turn(1), reward=1.0, status=TrajectoryStatus.PASSED)
    lose = _traj("a_1", _turn(1), reward=0.0)
    assert rft_examples([win, lose]) == [win]


def test_tdpo_pairs_one_per_task():
    win = _traj("aaaaaaa_1", _turn(1), reward=1.0, status=TrajectoryStatus.PASSED)
    lose = _traj("aaaaaaa_1", _turn(1, command="print(apis.venmo.pay())"), reward=0.0)
    pairs = tdpo_pairs([win, lose])
    assert len(pairs) == 1 and pairs[0].arm == "T-DPO"


def test_grew_matches_gf_extra_count():
    win = _traj("aaaaaaa_1", _turn(1), reward=1.0, status=TrajectoryStatus.PASSED)
    a = _traj("aaaaaaa_1", _turn(1), _turn(2, invalid=[{"e": 1}]), _turn(3))
    b = _traj("aaaaaaa_1", _turn(1), _turn(2), _turn(3))
    pairs = grew_pairs([win, a, b], n_gf_pairs=1)
    assert sum(p.arm == "G-REW" for p in pairs) >= 2


def test_gf_dpo_rejects_signature_collision():
    fail = _traj("aaaaaaa_1", _turn(1), _turn(2, invalid=[{"e": 1}]), _turn(3), _turn(4))
    assert gf_eligible_failures([fail])
    same = Action("", "", fail.turns[1].command, 1.0, 2)
    pairs = gf_dpo_pairs([fail], {"aaaaaaa_1": [(same, 1.0)]})
    assert pairs == []


def test_gf_dpo_keeps_distinct_signature():
    fail = _traj("aaaaaaa_1", _turn(1), _turn(2, invalid=[{"e": 1}]), _turn(3), _turn(4))
    win = Action("", "", "print(apis.phone.search_contacts())", 1.0, 2)
    pairs = gf_dpo_pairs([fail], {"aaaaaaa_1": [(win, 1.0)]})
    assert len(pairs) == 1 and not pairs[0].signature_collision


def test_eval_only_continuations_never_pair():
    import random

    from guardfork.train.pairs import rfork_dpo_pairs, usable_continuations

    fail = _traj("aaaaaaa_1", _turn(1), _turn(2, invalid=[{"e": 1}]), _turn(3), _turn(4))
    win = Action("", "", "print(apis.phone.search_contacts())", 1.0, 2, eval_only=True)
    assert usable_continuations([(win, 1.0)]) == []
    assert gf_dpo_pairs([fail], {"aaaaaaa_1": [(win, 1.0)]}) == []
    assert rfork_dpo_pairs([fail], {"aaaaaaa_1": [(win, 1.0)]}, random.Random(0)) == []
    ok = Action("", "", "print(apis.phone.search_contacts())", 1.0, 2)
    assert len(gf_dpo_pairs([fail], {"aaaaaaa_1": [(win, 1.0), (ok, 1.0)]})) == 1


def test_dpo_length_normalize_requires_lengths():
    with pytest.raises(ValueError):
        dpo_loss(0.0, 0.0, 0.0, 0.0, length_normalize=True)
    assert math.isfinite(dpo_loss(-1, -2, -1, -1, chosen_len=4, rejected_len=2,
                                  length_normalize=True))
