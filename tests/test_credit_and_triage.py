from __future__ import annotations

from pathlib import Path

import pytest

from guardfork.guards.rules import Guard, GuardFiring
from guardfork.q1.triage import (
    AUDIT_NOTE,
    abort_savings,
    abort_turn,
    hopeless_label,
    hopeless_screen,
    silent_harm_table,
)
from guardfork.telemetry.trials import TrialRecord
from guardfork.train.credit import (
    blame_onset,
    c3_rule_flip,
    delta_mc,
    hard_mask_prefix,
    open_judge,
    pitfall_recall,
    prefix_preservation,
    rank_corr,
    score_by_third,
    soft_posterior_pw,
    spearman,
    thirds,
    to_blame,
    uniform_grpo,
)

# V_hat staircase: alive through s_5, collapses at s_6, T = 8, failed run.
VALUES = {1: 0.75, 2: 0.75, 3: 0.7, 4: 0.75, 5: 0.75, 6: 0.0, 7: 0.0, 8: 0.0}
T_STAR = 5
T = 8


# ------------------------------------------------------------------ delta and rules


def test_delta_mc_adjacent_pairs_with_terminal_reward():
    d = delta_mc(VALUES, 0.0, T)
    assert set(d) == set(range(1, 9))
    assert d[5] == pytest.approx(0.75) and d[8] == pytest.approx(0.0)
    assert d[2] == pytest.approx(0.05) and d[3] == pytest.approx(-0.05)
    sparse = delta_mc({1: 0.5, 3: 0.5, 4: 0.0}, 0.0, 4)
    assert set(sparse) == {3, 4}


def test_rules_shapes_and_ranges():
    assert uniform_grpo(4) == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
    assert hard_mask_prefix(5, 3) == {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.0}
    assert hard_mask_prefix(3, None) == uniform_grpo(3)
    assert open_judge(4, 4) == {1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0}
    firings = [GuardFiring(Guard.G1_INVALID_ACTION, 2, "x"),
               GuardFiring(Guard.G3_FAILED_PRECONDITION, 3, "y"),
               GuardFiring(Guard.G2_LOOP, 4, "z")]
    assert c3_rule_flip(4, firings) == {1: 1.0, 2: -1.0, 3: 1.0, 4: -1.0}
    assert to_blame(c3_rule_flip(4, firings)) == {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0}
    with pytest.raises(ValueError):
        uniform_grpo(0)


def test_soft_posterior_is_cumulative_and_normalised():
    w = soft_posterior_pw(5, {2: 1.0, 3: 1.0, 4: 2.0})
    assert w == pytest.approx({1: 0.0, 2: 0.25, 3: 0.5, 4: 1.0, 5: 1.0})
    seq = soft_posterior_pw(4, [0.0, 0.5, 0.5, 0.0])
    assert seq[1] == 0.0 and seq[2] == pytest.approx(0.5) and seq[4] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        soft_posterior_pw(3, [1.0, 0.0])
    with pytest.raises(ValueError):
        soft_posterior_pw(3, {1: 0.0})


# ------------------------------------------------------------------ scoring


def test_blame_onset_is_posterior_for_soft_and_spike_for_hard():
    soft = soft_posterior_pw(5, {2: 0.25, 3: 0.75})
    assert blame_onset(soft) == pytest.approx({1: 0.0, 2: 0.25, 3: 0.75, 4: 0.0, 5: 0.0})
    assert blame_onset(hard_mask_prefix(4, 3)) == {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0}
    assert blame_onset(uniform_grpo(3)) == {1: 1.0, 2: 0.0, 3: 0.0}
    flips = [GuardFiring(Guard.G2_LOOP, 2, ""), GuardFiring(Guard.G1_INVALID_ACTION, 4, "")]
    assert blame_onset(c3_rule_flip(4, flips)) == {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0}


def test_pitfall_recall_and_prefix_preservation():
    d = delta_mc(VALUES, 0.0, T)
    soft = soft_posterior_pw(T, {4: 0.2, 5: 0.6, 6: 0.2})
    assert pitfall_recall(soft, d, top_k=1) == 1.0  # turn 5 is the big drop
    assert pitfall_recall(uniform_grpo(T), d, top_k=1) == 0.0  # ties -> turn 1
    assert pitfall_recall(hard_mask_prefix(T, 5), d, top_k=1) == 1.0
    assert pitfall_recall(hard_mask_prefix(T, 7), d, top_k=1) == 0.0
    assert pitfall_recall(soft, {}, top_k=1) is None and pitfall_recall(soft, d, 0) is None

    assert prefix_preservation(uniform_grpo(T), T_STAR) == 0.0
    assert prefix_preservation(hard_mask_prefix(T, 5), T_STAR) == 1.0
    assert prefix_preservation(hard_mask_prefix(T, 3), T_STAR) == pytest.approx(0.5)
    assert prefix_preservation(soft, T_STAR) == pytest.approx(1 - 0.2 / 4)
    assert prefix_preservation(soft, 1) == 1.0 and prefix_preservation(soft, None) is None


def test_spearman_with_ties_and_rank_corr():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert spearman([1], [2]) is None
    assert spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486832980505138)
    d = delta_mc(VALUES, 0.0, T)
    assert rank_corr(uniform_grpo(T), d) is None  # constant blame
    assert rank_corr(hard_mask_prefix(T, 5), d) > 0
    assert rank_corr(hard_mask_prefix(T, 5), d) > rank_corr(hard_mask_prefix(T, 7), d)


def test_thirds_and_score_by_third():
    assert thirds(8) == {"first": range(1, 3), "middle": range(3, 6), "last": range(6, 9)}
    assert thirds(9) == {"first": range(1, 4), "middle": range(4, 7), "last": range(7, 10)}
    assert thirds(2) == {"first": range(1, 1), "middle": range(1, 2), "last": range(2, 3)}
    d = delta_mc(VALUES, 0.0, T)
    res = score_by_third(hard_mask_prefix(T, 5), d, T_STAR, T, top_k=1)
    assert set(res) == {"overall", "first", "middle", "last"}
    assert res["overall"]["n"] == 8 and res["overall"]["t_star_third"] == "middle"
    assert res["overall"]["pitfall_recall"] == 1.0
    assert res["overall"]["prefix_preservation"] == 1.0
    assert res["middle"]["pitfall_recall"] == 1.0 and res["middle"]["n"] == 3
    assert res["first"]["rank_corr"] is None  # blame constant 0 in the first third
    assert res["last"]["n"] == 3


# ------------------------------------------------------------------ triage: s1 screen


def test_hopeless_labels():
    assert hopeless_label([0] * 8) == "pending"
    assert hopeless_label([0] * 16) == "hopeless"
    assert hopeless_label([0] * 7 + [1]) == "eligible"
    assert hopeless_label([0] * 15 + [1]) == "eligible"
    assert hopeless_label([0, 0]) == "pending"


def test_hopeless_screen_counts_and_confirmation_queue():
    res = hopeless_screen({
        "a": [0] * 8, "b": [0] * 16, "c": [1, 0, 0, 0, 0, 0, 0, 0], "d": [0] * 4,
        "e": [0] * 12,
    })
    assert res["labels"] == {"a": "pending", "b": "hopeless", "c": "eligible", "d": "pending",
                             "e": "pending"}
    assert res["counts"] == {"eligible": 1, "hopeless": 1, "pending": 3}
    assert res["needs_confirmation"] == ["a", "e"]
    assert res["n"] == 5


# ------------------------------------------------------------------ triage: silent harm


def _rec(name: str, reward: float, job: str = "j") -> TrialRecord:
    return TrialRecord(Path(name), job, "aaaaaaa_1", reward, None, 1.0, [], [])


def test_silent_harm_table_selects_passed_runs_with_g4_g5():
    firings = {
        "p1": [GuardFiring(Guard.G5_UNVERIFIED_COMPLETION, 6, "complete_task without read"),
               GuardFiring(Guard.G4_CONSTRAINT_VIOLATION, 3, "destructive write/delete")],
        "p2": [GuardFiring(Guard.G3_FAILED_PRECONDITION, 2, "HTTP 404")],
        "p3": [],
        "f1": [GuardFiring(Guard.G4_CONSTRAINT_VIOLATION, 1, "apps outside task")],
        "p4": [GuardFiring(Guard.G4_CONSTRAINT_VIOLATION, 2, "apps outside task")],
    }
    recs = [_rec("p1", 1.0), _rec("p2", 1.0), _rec("p3", 1.0), _rec("f1", 0.0),
            _rec("p4", 1.0)]
    rows = silent_harm_table(recs, lambda r: firings[r.trial_dir.name])
    assert [(r["trial"], r["guard"], r["turn"]) for r in rows] == [
        ("j/p1", "G4", 3), ("j/p1", "G5", 6), ("j/p4", "G4", 2)]
    assert all(r["manual_verdict"] == "" for r in rows)
    assert rows[0]["note"] == AUDIT_NOTE and rows[1]["note"] == ""
    assert "changed_model_names" in AUDIT_NOTE
    assert set(rows[0]) == {"trial", "guard", "turn", "evidence", "manual_verdict", "note"}
    capped = silent_harm_table(recs, lambda r: firings[r.trial_dir.name], n_runs=1)
    assert {r["trial"] for r in capped} == {"j/p1"}


# ------------------------------------------------------------------ triage: abort savings


def test_abort_turn_threshold():
    assert abort_turn({3: 0.5, 4: 0.3, 5: 0.2}, 0.9) == 5
    assert abort_turn({3: 0.5, 4: 0.3, 5: 0.2}, 0.7) == 4
    assert abort_turn([0.1, 0.1, 0.1], 0.9) == 3  # normalised: certain by the last turn
    assert abort_turn([0.1, 0.1, 0.1], 0.9, T=3) is None  # ...but nothing left to abort
    assert abort_turn({3: 0.5, 4: 0.3, 5: 0.2}, 0.9, T=5) is None
    with pytest.raises(ValueError):
        abort_turn({1: 0.0})


def test_abort_savings_tokens_and_recall():
    posteriors = {
        "r1": {5: 0.95, 6: 0.05},  # abort after turn 5 -> saves turns 6..8
        "r2": {2: 0.95, 3: 0.05},  # premature: t* = 4
        "r3": {t: 0.1 for t in range(1, 9)},  # never decisive -> no abort
        "r4": {1: 1.0},  # no token counts -> skipped
    }
    tokens = {
        "r1": dict.fromkeys(range(1, 9), 100),
        "r2": dict.fromkeys(range(1, 9), 100),
        "r3": [100] * 8,
    }
    res = abort_savings(posteriors, tokens, 0.9, t_star_by_run={"r1": 5, "r2": 4, "r3": 6})
    assert res["n_runs"] == 3 and res["n_aborted"] == 2
    assert res["tokens_total"] == 2400 and res["tokens_saved"] == 300 + 600
    assert res["tokens_saved_frac"] == pytest.approx(900 / 2400)
    assert res["abort_turns"] == {"r1": 5, "r2": 2, "r3": None}
    assert res["recall"] == pytest.approx(2 / 3)
    assert res["premature_frac"] == pytest.approx(1 / 3)
    assert res["mean_lag"] == 0.0
    plain = abort_savings(posteriors, tokens)
    assert plain["recall"] is None and plain["tokens_saved_frac"] == pytest.approx(900 / 2400)
    with pytest.raises(ValueError):
        abort_savings(posteriors, tokens, 1.5)
