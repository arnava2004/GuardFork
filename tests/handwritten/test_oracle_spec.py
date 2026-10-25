"""Specs for stats/oracle.py, stats/bootstrap.py, fork/probe_plan.py (Arnav implements)."""

import math

import pytest

from guardfork.fork.probe_plan import initial_probes, next_probe
from guardfork.stats.bootstrap import (
    analytic_random_hit,
    clustered_bootstrap_ci,
    hit_minus_random,
    holm_adjust,
)
from guardfork.stats.oracle import argmax_drop_turn, derailment_turn, jeffreys_v1


def test_jeffreys():
    assert jeffreys_v1(3, 8) == pytest.approx(3.5 / 9)


def test_derailment_turn_first_qualifying_adjacent_pair():
    # K=8, V~(s1)=0.6 -> need V(s_{t+1}) <= 0.3 and drop >= 0.25
    values = {1: 0.625, 3: 0.5, 4: 0.125, 5: 0.0}
    assert derailment_turn(values, reward=0.0, v_s1=0.6, K=8, T=6) == 3


def test_derailment_uses_terminal_reward_as_s_T_plus_1():
    values = {5: 0.5, 6: 0.5}
    assert derailment_turn(values, reward=0.0, v_s1=0.6, K=8, T=6) == 6


def test_derailment_undefined_when_flat():
    values = {1: 0.25, 2: 0.25, 3: 0.25}
    assert derailment_turn(values, reward=0.25, v_s1=0.6, K=8, T=3) is None


def test_non_adjacent_pairs_ignored():
    values = {1: 0.625, 4: 0.0}
    assert derailment_turn(values, reward=0.0, v_s1=0.6, K=8, T=6) is None


def test_argmax_drop():
    values = {1: 0.75, 2: 0.625, 3: 0.125}
    assert argmax_drop_turn(values, reward=0.0, T=3) == 2


def test_initial_probes():
    assert initial_probes(5, 10) == [1, 2, 4, 5, 6]
    assert initial_probes(2, 3) == [1, 2, 3]


def test_analytic_random_hit():
    assert analytic_random_hit(11, 1, 1) == pytest.approx(2 / 10)
    assert analytic_random_hit(11, 5, 1) == pytest.approx(3 / 10)
    assert hit_minus_random(5, 5, 11, 1) == pytest.approx(1 - 0.3)


def test_clustered_bootstrap_constant():
    mean, lo, hi = clustered_bootstrap_ci([0.5] * 6, ["a", "a", "b", "b", "c", "c"], n_boot=200)
    assert mean == lo == hi == 0.5
    assert not math.isnan(mean)


def test_next_probe_bisects_crossing_gap():
    values = {1: 0.6, 2: 0.6, 4: 0.1, 5: 0.0, 6: 0.0}
    assert next_probe(values, v_s1=0.6, K=8, T=10) == 3
    filled = {**values, 3: 0.35}
    assert next_probe(filled, v_s1=0.6, K=8, T=10) is None


def test_holm_adjust_monotone():
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert adj[0] <= adj[2] <= adj[1]
    assert adj[0] == pytest.approx(0.03)
