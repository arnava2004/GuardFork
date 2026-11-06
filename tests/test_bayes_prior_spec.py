"""Specs for stats/bayes_prior.py, written before the implementation (originally for
Arnav to hand-write against; implemented by an agent at Arnav's direction on 2026-11-05)."""

from __future__ import annotations

import pytest

from guardfork.stats.bayes_prior import (
    ChangePointResult,
    LaplaceGuardPrior,
    fit_changepoint_oracle,
    next_bayes_probe,
    sequential_beta_stop,
)

# Staircase V-hat at K = 8: alive through s_5, collapsed from s_6 on. V~(s1) = 0.8.
STAIRCASE = {1: (7, 8), 2: (7, 8), 3: (6, 8), 4: (6, 8), 5: (6, 8), 6: (0, 8), 7: (0, 8),
             8: (0, 8), 9: (0, 8)}


def test_changepoint_staircase_collapse_at_5():
    res = fit_changepoint_oracle(STAIRCASE, v_s1=0.8, n_boot=200, seed=0)
    assert isinstance(res, ChangePointResult)
    assert res.t_star == 5
    assert res.lo is not None and res.hi is not None and res.lo <= 5 <= res.hi
    assert res.tau_abs == pytest.approx(0.4)
    assert res.t_star_drop == 5
    assert set(res.sensitivity) == {0.3, 0.7}
    assert res.sensitivity[0.3] == 5 and res.sensitivity[0.7] == 5


def test_changepoint_interval_covers_noisy_boundary():
    noisy = {**STAIRCASE, 5: (3, 8), 6: (2, 8)}
    res = fit_changepoint_oracle(noisy, v_s1=0.8, n_boot=300, seed=1)
    assert res.t_star in (4, 5, 6)
    assert res.lo <= res.t_star <= res.hi
    assert res.hi - res.lo >= 1


def test_changepoint_undefined_when_flat():
    flat = {1: (6, 8), 2: (6, 8), 3: (5, 8)}
    res = fit_changepoint_oracle(flat, v_s1=0.8, n_boot=50)
    assert res.t_star is None and res.lo is None and res.hi is None


def test_changepoint_deterministic_in_seed():
    a = fit_changepoint_oracle(STAIRCASE, v_s1=0.8, n_boot=100, seed=7)
    b = fit_changepoint_oracle(STAIRCASE, v_s1=0.8, n_boot=100, seed=7)
    assert (a.lo, a.hi) == (b.lo, b.hi)


def test_changepoint_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_changepoint_oracle({}, v_s1=0.8)
    with pytest.raises(ValueError):
        fit_changepoint_oracle({1: (0, 0)}, v_s1=0.8)


def test_laplace_pmf_sums_to_one_and_peaks_at_g_minus_delta():
    prior = LaplaceGuardPrior(b=1.5, pi0=0.0, delta=1.0, lam=0.0)
    cands = list(range(1, 11))
    pmf = prior.pmf(cands, g=6, T=10)
    assert len(pmf) == 10
    assert sum(pmf) == pytest.approx(1.0)
    assert cands[max(range(10), key=pmf.__getitem__)] == 5
    assert pmf[4] > pmf[5] > pmf[6] and pmf[4] > pmf[3] > pmf[2]


def test_laplace_pmf_floor_and_uniform_without_guard():
    prior = LaplaceGuardPrior(b=1.0, pi0=0.1, delta=0.0, lam=0.2)
    cands = list(range(1, 9))
    pmf = prior.pmf(cands, g=2, T=8)
    assert sum(pmf) == pytest.approx(1.0)
    assert min(pmf) >= 0.2 / 8 - 1e-12
    uniform = prior.pmf(cands, g=None, T=8)
    assert all(p == pytest.approx(1 / 8) for p in uniform)


def test_laplace_fit_recovers_shift_and_scale():
    pairs = [(t + 2, t, 20) for t in range(3, 15)] + [(t + 2, t, 20) for t in range(3, 15)]
    fitted = LaplaceGuardPrior(lam=0.05).fit(pairs)
    assert fitted.delta == pytest.approx(2.0)
    assert fitted.b < 1.0
    assert fitted.pi0 < 0.2
    assert fitted.lam == 0.05
    with pytest.raises(ValueError):
        LaplaceGuardPrior().fit([])


def test_sequential_beta_stop_decisive_cases():
    assert sequential_beta_stop(0, 16, tau=0.3) is True  # P(p <= 0.3 | Beta(1, 17)) > 0.99
    assert sequential_beta_stop(4, 8, tau=0.3) is False  # Beta(5, 5) straddles 0.3
    assert sequential_beta_stop(8, 8, tau=0.3) is True  # decisively recoverable
    assert sequential_beta_stop(0, 4, tau=0.3) is False  # 1 - 0.7^5 = 0.83 < 0.95


def test_sequential_beta_stop_validates():
    with pytest.raises(ValueError):
        sequential_beta_stop(5, 4, tau=0.3)
    with pytest.raises(ValueError):
        sequential_beta_stop(0, 4, tau=1.5)


def test_next_bayes_probe_never_returns_probed_turn():
    posterior = [0.1, 0.2, 0.4, 0.2, 0.1]
    for probed in ({3}, {1, 2, 3}, {2, 4}, set()):
        nxt = next_bayes_probe(posterior, probed, T=5)
        assert nxt is not None and 1 <= nxt <= 5 and nxt not in probed
    assert next_bayes_probe(posterior, {1, 2, 3, 4, 5}, T=5) is None


def test_next_bayes_probe_bisects_mass_and_handles_point_mass():
    # Mass split between early and late turns: the entropy-minimising probe sits between.
    posterior = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0]
    nxt = next_bayes_probe(posterior, set(), T=6)
    assert nxt in (2, 3, 4, 5)
    assert next_bayes_probe([0.0, 1.0, 0.0], set(), T=3) is None
    with pytest.raises(ValueError):
        next_bayes_probe([0.5, 0.5], set(), T=3)
