from __future__ import annotations

import random

import pytest

from guardfork.q1 import simulate_search as ss
from guardfork.q1.simulate_search import (
    SearchInputs,
    SnappingProbeTable,
    beta_bootstrap_values,
    binomial_table,
    bisection_early_stop,
    bisection_fixed,
    candidate_grid,
    elpo_bel,
    entropy_peak_turn,
    guard_bayesect,
    guards_raw,
    guards_recovery_aware,
    lvrl_style_error_marker,
    simulate,
    snap,
    split_half,
    turn_surprisal,
)
from guardfork.stats.bayes_prior import LaplaceGuardPrior
from guardfork.types import ToolCall, Trajectory, TrajectoryStatus, Turn


def _turn(t: int, cmd: str = "x = 1", obs: str | None = "ok") -> Turn:
    return Turn(t, 9 + t, "", "", (ToolCall("execute", {"command": cmd}),), (), obs)


def _traj(turns: tuple[Turn, ...]) -> Trajectory:
    return Trajectory("aaaaaaa_1", turns, 0.0, TrajectoryStatus.FAILED)


def _plain(T: int) -> Trajectory:  # noqa: N803
    return _traj(tuple(_turn(t) for t in range(1, T + 1)))


# ------------------------------------------------------------------ grid / snapping


def test_candidate_grid_thinned_plus_windows_clipped():
    got = candidate_grid(10, step=3, guard_turns=[1, 9], judge_turn=None, entropy_turn=10)
    assert got == [1, 2, 4, 7, 8, 9, 10]
    assert candidate_grid(5, step=2, guard_turns=[None]) == [1, 3, 5]
    with pytest.raises(ValueError):
        candidate_grid(0, step=1)


def test_snap_prefers_nearest_then_smaller():
    assert snap(4, [1, 3, 5, 8]) == 3
    assert snap(6, [1, 3, 5, 8]) == 5
    assert snap(7, [1, 3, 5, 8]) == 8
    assert snap(3, [3]) == 3
    with pytest.raises(ValueError):
        snap(1, [])


# ------------------------------------------------------------------ entropy peak


def _proxy_record(live_turn: int, logprobs: list[float] | None, status: int = 200) -> dict:
    msgs = [{"role": "assistant"}] * (10 + live_turn - 1)
    rec = {"status": status, "request": {"messages": msgs}}
    if logprobs is not None:
        rec["completion_logprobs"] = [logprobs]
    return rec


def test_entropy_peak_from_synthetic_logprobs():
    records = [
        _proxy_record(1, [-0.1, -0.2, -0.3]),
        _proxy_record(2, [-2.0, -1.0]),
        _proxy_record(3, [-1.5, -1.5]),
        _proxy_record(4, None),
    ]
    surp = turn_surprisal(records)
    assert surp[1] == pytest.approx(0.2) and surp[2] == pytest.approx(1.5)
    assert 4 not in surp
    assert entropy_peak_turn(records) == 2  # tie 1.5 vs 1.5 -> earliest


def test_entropy_peak_ignores_failed_requests_and_empty():
    assert entropy_peak_turn([]) is None
    assert entropy_peak_turn([_proxy_record(2, [-5.0], status=500),
                              _proxy_record(1, [-0.5])]) == 1


# ------------------------------------------------------------------ resampling


def test_beta_bootstrap_deterministic_with_seed_and_in_unit_interval():
    values = {1: (7, 8), 3: (4, 8), 5: (0, 8)}
    a = beta_bootstrap_values(values, random.Random(3))
    b = beta_bootstrap_values(values, random.Random(3))
    c = beta_bootstrap_values(values, random.Random(4))
    assert a == b and a != c
    assert set(a) == {1, 3, 5} and all(0.0 < p < 1.0 for p in a.values())
    with pytest.raises(ValueError):
        beta_bootstrap_values({1: (9, 8)}, random.Random(0))


def test_binomial_table_shapes():
    tbl = binomial_table({1: 1.0, 2: 0.0}, 8, random.Random(0))
    assert tbl[1] == [1] * 8 and tbl[2] == [0] * 8


def test_split_half_sizes_and_conservation():
    outcomes = {1: [1] * 10 + [0] * 6, 2: [0] * 16, 5: [1, 0] * 8}
    a, b = split_half(outcomes, random.Random(0))
    for t in outcomes:
        assert a[t][1] == 8 and b[t][1] == 8
        assert a[t][0] + b[t][0] == sum(outcomes[t])
    with pytest.raises(ValueError):
        split_half({1: [1, 0, 1]}, random.Random(0))


# ------------------------------------------------------------------ probe table


def test_probe_table_snaps_and_charges_once_per_rollout():
    tbl = SnappingProbeTable(10, values={1: (7, 8), 5: (6, 8), 7: (0, 8)})
    assert tbl.value(6) == (6, 8)  # 6 -> tie between 5 and 7 -> 5
    assert tbl.continuations_used == 8
    assert tbl.value(5) == (6, 8) and tbl.continuations_used == 8  # already paid
    s, k = tbl.value(8, k_max=4)
    assert k == 4 and s == 0 and tbl.continuations_used == 12
    assert tbl.value(7) == (0, 8) and tbl.continuations_used == 16
    tbl.reset()
    assert tbl.continuations_used == 0


def test_probe_table_from_outcomes_prefix_is_ordered():
    tbl = SnappingProbeTable(4, outcomes={2: [1, 1, 0, 0]})
    assert tbl.value(2, k_max=2) == (2, 2)
    assert tbl.value(2) == (2, 4)
    with pytest.raises(ValueError):
        SnappingProbeTable(4)


# ------------------------------------------------------------------ zero-rollout localizers


def test_error_marker_floor_uses_exact_strings():
    traj = _traj((
        _turn(1, obs="fine"),
        _turn(2, obs="warning: something"),
        _turn(3, obs="Exception: Response status code is 404"),
        _turn(4, obs="Traceback (most recent call last)"),
    ))
    inputs = SearchInputs(traj)
    tbl = SnappingProbeTable(4, values={1: (1, 1)})
    assert lvrl_style_error_marker(inputs, tbl) == 3
    assert lvrl_style_error_marker(SearchInputs(_plain(3)), tbl) is None
    assert tbl.continuations_used == 0


def test_guards_use_frozen_rules_on_trajectory_types():
    err = "Exception: Response status code is 401:"
    recovered = _traj((
        _turn(1, "print(apis.venmo.show_balance())", err),
        _turn(2, "print(apis.api_docs.show_api_doc(app_name='venmo', api_name='login'))"),
        _turn(3), _turn(4),
    ))
    inputs = SearchInputs.from_trajectory(recovered)
    tbl = SnappingProbeTable(4, values={1: (1, 1)})
    assert guards_raw(inputs, tbl) == 1
    assert guards_recovery_aware(inputs, tbl) is None
    unrecovered = _traj((_turn(1, "print(apis.venmo.show_balance())", err),
                         _turn(2), _turn(3), _turn(4)))
    assert guards_recovery_aware(SearchInputs.from_trajectory(unrecovered), tbl) == 1
    invalid = Trajectory("aaaaaaa_1", (
        _turn(1), Turn(2, 11, "", "", (), ({"e": 1},), "ok"), _turn(3), _turn(4),
    ), 0.0, TrajectoryStatus.FAILED)
    assert guards_raw(SearchInputs.from_trajectory(invalid), tbl) == 2
    assert tbl.continuations_used == 0


# ------------------------------------------------------------------ search policies


STAIR = {t: (6, 8) for t in range(1, 6)} | {t: (0, 8) for t in range(6, 11)}


def test_bisection_fixed_finds_collapse_and_accounts_budget():
    tbl = SnappingProbeTable(10, values=STAIR)
    inputs = SearchInputs(_plain(10), v_s1=0.75)
    pick = bisection_fixed(8)(inputs, tbl)
    assert pick == 5
    assert tbl.continuations_used % 8 == 0
    assert 16 <= tbl.continuations_used <= 8 * 6
    assert tbl.continuations_used == 8 * len(tbl.asked)


def test_bisection_early_stop_is_cheaper_on_clear_cut_states():
    inputs = SearchInputs(_plain(10), v_s1=0.75)
    fixed_tbl = SnappingProbeTable(10, values=STAIR)
    early_tbl = SnappingProbeTable(10, values=STAIR)
    assert bisection_fixed(8)(inputs, fixed_tbl) == 5
    assert bisection_early_stop(8)(inputs, early_tbl) == 5
    # collapsed turns (0/4) stop at half budget, alive ones always top up to 8
    assert fixed_tbl.continuations_used == 32  # turns 1, 5, 7, 6
    assert early_tbl.continuations_used == 24  # 8 + 8 + 4 + 4


def test_bisection_respects_max_probes_budget():
    tbl = SnappingProbeTable(10, values=STAIR)
    inputs = SearchInputs(_plain(10), v_s1=0.75)
    bisection_fixed(8, max_probes=2)(inputs, tbl)
    assert tbl.continuations_used == 16


def test_elpo_bel_bisects_with_small_budget_then_confirms():
    tbl = SnappingProbeTable(10, values=STAIR)
    inputs = SearchInputs(_plain(10), v_s1=0.75)
    assert elpo_bel(8)(inputs, tbl) == 5
    # bracket (0, 11): mids 5, 8, 6 at 4 rollouts each, then confirm turn 6 at 8
    assert tbl.continuations_used == 4 + 4 + 8 + 0
    assert set(tbl.asked) == {5, 6, 8}


def test_elpo_bel_undefined_when_s1_collapsed():
    tbl = SnappingProbeTable(4, values={t: (0, 8) for t in range(1, 5)})
    assert elpo_bel(8)(SearchInputs(_plain(4), v_s1=0.75), tbl) is None


def test_guard_bayesect_raises_until_arnav_implements():
    tbl = SnappingProbeTable(10, values=STAIR)
    inputs = SearchInputs(_plain(10), v_s1=0.75)
    with pytest.raises(NotImplementedError):
        guard_bayesect(LaplaceGuardPrior(), 8)(inputs, tbl)


def test_guard_bayesect_with_mocked_prior(monkeypatch):
    def fake_pmf(self, cands, g, T):  # noqa: N803
        centre = g if g is not None else (T + 1) // 2
        w = [1.0 / (1 + abs(c - centre)) for c in cands]
        z = sum(w)
        return [x / z for x in w]

    def fake_next(posterior, probed, T):  # noqa: N803
        free = [t for t in range(1, T + 1) if t not in probed]
        return max(free, key=lambda t: (posterior[t - 1], -t)) if free else None

    def fake_stop(s, k, *, tau, delta=0.05):
        return s == 0 or s == k or k >= 8

    monkeypatch.setattr(LaplaceGuardPrior, "pmf", fake_pmf)
    monkeypatch.setattr(ss.bayes_prior, "next_bayes_probe", fake_next)
    monkeypatch.setattr(ss.bayes_prior, "sequential_beta_stop", fake_stop)

    traj = Trajectory("aaaaaaa_1", (
        *(_turn(t) for t in range(1, 7)), Turn(7, 16, "", "", (), ({"e": 1},), "ok"),
        *(_turn(t) for t in range(8, 11)),
    ), 0.0, TrajectoryStatus.FAILED)
    inputs = SearchInputs.from_trajectory(traj, v_s1=0.75)
    tbl = SnappingProbeTable(10, values=STAIR)
    pick = guard_bayesect(LaplaceGuardPrior(), 8, max_probes=6)(inputs, tbl)
    assert pick == 5
    assert 0 < tbl.continuations_used <= 6 * 8
    assert tbl.continuations_used % 4 == 0


def test_simulate_reports_hits_and_costs_per_policy():
    tbl = SnappingProbeTable(10, values=STAIR)
    inputs = SearchInputs(_plain(10), v_s1=0.75, judge_turn=3, entropy_turn=9)
    policies = {
        "open_judge": ss.open_judge,
        "entropy_peak": ss.entropy_peak,
        "last_turn": ss.last_turn,
        "bisection_fixed": bisection_fixed(8),
    }
    res = simulate(policies, tbl, t_star=5, inputs=inputs)
    assert res["open_judge"]["pick"] == 3 and res["open_judge"]["continuations"] == 0
    assert res["open_judge"]["hit"] == {0: False, 1: False, 2: True, 3: True}
    assert res["last_turn"]["pick"] == 10 and not any(res["last_turn"]["hit"].values())
    assert res["bisection_fixed"]["hit"][0] is True
    assert res["bisection_fixed"]["continuations"] > 0
    assert tbl.continuations_used == 0  # reset after the run
    none = simulate({"entropy_peak": ss.entropy_peak}, tbl, t_star=None, inputs=inputs)
    assert not any(none["entropy_peak"]["hit"].values())


def test_default_policies_ladder_names():
    names = set(ss.default_policies(8))
    assert {"lvrl_style_error_marker", "random", "last_turn", "entropy_peak", "guards_raw",
            "guards_recovery_aware", "open_judge", "claude_judge", "bisection_fixed",
            "bisection_early_stop", "elpo_bel"} == names
    with_prior = ss.default_policies(8, prior=LaplaceGuardPrior())
    assert {"guard_bayesect", "guard_bayesect_open_prior",
            "guard_bayesect_claude_prior"} <= set(with_prior)
    assert ss.POLICY_FLAGS["guard_bayesect_claude_prior"] == {"eval_only": True}
    assert ss.eval_only_policies(with_prior) == {"claude_judge", "guard_bayesect_claude_prior"}
    assert "guard_bayesect_open_prior" not in ss.eval_only_policies(with_prior)


def test_candidate_grid_accepts_multiple_judge_turns():
    got = candidate_grid(12, step=4, guard_turns=[], judge_turns=[3, None, 11])
    assert got == [1, 2, 3, 4, 5, 9, 10, 11, 12]
    assert candidate_grid(12, step=4, judge_turn=3, judge_turns=[11]) == got


def test_judge_slot_inputs_and_fallbacks():
    inputs = SearchInputs(_plain(6), judge_turn=2)
    assert ss.open_judge(inputs, None) == 2 and ss.claude_judge(inputs, None) is None
    inputs = SearchInputs(_plain(6), judge_turn=2,
                          judge_turns={"judge_open": 4, "judge_claude": 5},
                          judge_dists={"judge_open": [0, 0, 0, 1, 0, 0], "judge_local": [1]})
    assert ss.open_judge(inputs, None) == 4 and ss.claude_judge(inputs, None) == 5
    assert ss.judge_slot_pick("judge_claude")(inputs, None) == 5
    assert inputs.judge_dist("judge_open") == [0, 0, 0, 1, 0, 0]
    assert inputs.judge_dist("judge_local") is None  # wrong length
    lik = ss.judge_likelihood([0, 0, 0, 1, 0, 0], 6, floor=0.06)
    assert lik == pytest.approx([0.01, 0.01, 0.01, 0.95, 0.01, 0.01])
    assert ss.judge_likelihood(None, 6) is None and ss.judge_likelihood([0, 0], 6) is None


def test_guard_bayesect_judge_prior_shifts_posterior(monkeypatch):
    def fake_pmf(self, cands, g, T):  # noqa: N803
        return [1.0 / len(cands)] * len(cands)

    monkeypatch.setattr(LaplaceGuardPrior, "pmf", fake_pmf)
    monkeypatch.setattr(ss.bayes_prior, "next_bayes_probe", lambda post, probed, T: None)
    inputs = SearchInputs(_plain(10), v_s1=0.75,
                          judge_dists={"judge_open": [0] * 6 + [1.0] + [0] * 3,
                                       "judge_claude": [1.0] + [0] * 9})
    tbl = SnappingProbeTable(10, values=STAIR)
    # no probes (fake next returns None): the pick is the prior x judge-likelihood mode
    assert guard_bayesect(LaplaceGuardPrior(), 8)(inputs, tbl) == 1  # uniform: ties -> 1
    assert guard_bayesect(LaplaceGuardPrior(), 8, judge_slot="judge_open")(inputs, tbl) == 7
    assert guard_bayesect(LaplaceGuardPrior(), 8, judge_slot="judge_claude")(inputs, tbl) == 1
    assert guard_bayesect(LaplaceGuardPrior(), 8, judge_slot="judge_local")(inputs, tbl) == 1
    assert tbl.continuations_used == 0


def test_random_turn_is_seeded_and_non_terminal():
    inputs = SearchInputs(_plain(10), seed=5)
    tbl = SnappingProbeTable(10, values={1: (1, 1)})
    a = ss.random_turn(inputs, tbl)
    assert a == ss.random_turn(inputs, tbl) and 1 <= a <= 9
