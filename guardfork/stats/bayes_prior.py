"""Change-point oracle, Laplace guard prior, sequential Beta stopping, Bayes next probe.

HAND-WRITTEN BY ARNAV (AGENTS.md rule 1). Agents may only edit the stubs, docstrings and
type signatures in this file and the failing specs in
`tests/handwritten/test_bayes_prior_spec.py`. Every body below raises `NotImplementedError`
until Arnav fills it in.

Conventions shared with `guardfork/stats/oracle.py` and `guardfork/fork/probe_plan.py`:

- Turns are 1-based over live turns; `s_t` is the state *before* turn `t`, so `values[t]`
  is the Monte Carlo estimate of `V(s_t)` as `(successes, rollouts)`.
- A failed run has `V(s_{T+1}) = reward = 0`; callers that want the terminal state in the
  fit pass `{T + 1: (0, K)}` explicitly.
- `t*` is the first irrecoverable turn: the first turn `t` after which the fitted value
  stays at or below the threshold, i.e. `V(s_u) <= tau_abs` for every probed `u > t` and
  `V(s_t) > tau_abs`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChangePointResult:
    """Output of `fit_changepoint_oracle`.

    `t_star`: change-point first-irrecoverable turn at `tau_abs = tau * v_s1`, or None when
    no probed turn is above the threshold with a later probed turn at or below it.
    `lo`, `hi`: 90% bootstrap credible interval for `t_star` (both None when `t_star` is
    None). `tau_abs`: the absolute threshold used. `t_star_drop`: largest-drop sensitivity
    turn (argmax over adjacent probed pairs of `V(s_t) - V(s_{t+1})`). `sensitivity`: the
    change-point turn at alternative relative thresholds, keyed by the relative `tau`
    (pre-registered: 0.3 and 0.7).
    """

    t_star: int | None
    lo: int | None
    hi: int | None
    tau_abs: float
    t_star_drop: int | None
    sensitivity: dict[float, int | None] = field(default_factory=dict)


def fit_changepoint_oracle(values: dict[int, tuple[int, int]], *, v_s1: float,
                           tau: float = 0.5, n_boot: int = 1000, seed: int = 0,
                           ) -> ChangePointResult:
    """Primary Q1 oracle (PREREGISTRATION.md, "Primary oracle").

    `values[t] = (s_t, k_t)` are the Monte Carlo continuation outcomes at every probed state
    (thinned grid, candidate windows, K = 16 refinements, optionally `{T+1: (0, K)}`).
    `v_s1` is the Jeffreys-smoothed base value `V~(s_1)` (`stats.oracle.jeffreys_v1`).

    Fit: a single change-point step model over all `sum(k_t)` outcomes, with
    `V(s_u) = p_hi` for `u <= c` and `V(s_u) = p_lo` for `u > c`, `c` ranging over the
    probed turns; `t_star` is the change point `c` whose fitted post-change level satisfies
    `p_lo <= tau * v_s1` and that first makes every later probed value stay at or below the
    threshold. The 90% interval `[lo, hi]` comes from `n_boot` Beta-posterior parametric
    bootstrap replicates (`p_t ~ Beta(1 + s_t, 1 + k_t - s_t)` at every probed `t`, re-fit,
    5th / 95th percentile of the replicate `t_star`), seeded by `seed` through
    `random.Random`. `t_star_drop` and `sensitivity` (relative thresholds 0.3 and 0.7) are
    computed from the same point estimates; `sensitivity` keys are the relative `tau`.

    Raises `ValueError` if `values` is empty, any `k_t <= 0`, or `v_s1 <= 0`.
    """
    raise NotImplementedError("Arnav hand-writes fit_changepoint_oracle (AGENTS.md rule 1)")


@dataclass(frozen=True)
class LaplaceGuardPrior:
    """Prior over `t*` given the first non-terminal guard firing `g`.

    `b`: Laplace scale (turns). `pi0`: false-fire probability, the chance that `g` carries
    no information about `t*` (that branch is uniform over the candidates). `delta`: shift
    of the mode, `t* = g - delta` (guards fire *after* the derailment; calibrated per model,
    the 9B shift is expected to differ from the 4B one). `lam`: a hand-set uniform-mixture
    floor so no candidate turn ever gets zero prior mass; it is not fitted.

    `pmf(c, g, T) ∝ lam / |c| + (1 - lam) * (pi0 / |c| + (1 - pi0) * Laplace(t; g - delta, b))`
    for each `t` in the candidate list `c`, normalised to sum to 1 over `c`. With `g` None
    (no non-terminal guard fired) the pmf is uniform over `c`.
    """

    b: float = 2.0
    pi0: float = 0.1
    delta: float = 0.0
    lam: float = 0.05

    def pmf(self, t_star_candidates: list[int], g: int | None, T: int,  # noqa: N803
            ) -> list[float]:
        """Prior probability of each candidate turn in `t_star_candidates` (same order),
        summing to 1. With `pi0 = lam = 0` the mode is `g - delta` (ties to the nearest
        integer candidate on the earlier side). Raises `ValueError` on an empty candidate
        list or a candidate outside `[1, T]`."""
        raise NotImplementedError("Arnav hand-writes LaplaceGuardPrior.pmf (AGENTS.md rule 1)")

    def fit(self, pairs: list[tuple[int, int, int]]) -> LaplaceGuardPrior:
        """Return a new prior calibrated on design-split `(g, t_star, T)` triples (never
        evaluation scenarios). `delta` is the median of `g - t_star`; `b` is the Laplace
        maximum-likelihood scale, the mean of `|g - delta - t_star|` over the non-false-fire
        responsibilities; `pi0` comes from a few EM steps over the uniform/Laplace mixture.
        `lam` is carried over unchanged. Raises `ValueError` on an empty list."""
        raise NotImplementedError("Arnav hand-writes LaplaceGuardPrior.fit (AGENTS.md rule 1)")


def sequential_beta_stop(successes: int, trials: int, *, tau: float, delta: float = 0.05,
                         ) -> bool:
    """Sequential stopping rule for continuations at one probed state.

    With the Beta(1 + successes, 1 + trials - successes) posterior on `p = V(s_t)`, stop
    (return True) when the posterior is decisive about the threshold `tau` either way:
    `P(p <= tau) >= 1 - delta` (collapsed) or `P(p <= tau) <= delta` (recoverable).
    Return False while the posterior straddles `tau`, so the caller launches more
    continuations. `tau` is absolute (already multiplied by `V~(s_1)`). Raises `ValueError`
    unless `0 <= successes <= trials`, `0 < tau < 1` and `0 < delta < 0.5`.
    """
    raise NotImplementedError("Arnav hand-writes sequential_beta_stop (AGENTS.md rule 1)")


def next_bayes_probe(posterior: list[float], probed: set[int], T: int,  # noqa: N803
                     ) -> int | None:
    """Turn to probe next by expected posterior-entropy minimisation.

    `posterior[t - 1]` is the current `P(t* = t)` for `t` in `1..T` (length `T`, sums to
    1). Probing state `s_t` reveals whether `t` is before or after the change point
    (`t <= t*` recoverable, `t > t*` collapsed, treated as a noiseless binary outcome for
    the planning step); choose the unprobed `t` whose outcome minimises the expected
    entropy of the updated posterior, breaking ties toward the smaller turn. Return None
    when every turn in `1..T` is in `probed` or the posterior is already a point mass.
    Raises `ValueError` if `len(posterior) != T`.
    """
    raise NotImplementedError("Arnav hand-writes next_bayes_probe (AGENTS.md rule 1)")
