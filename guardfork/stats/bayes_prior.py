"""Change-point oracle, Laplace guard prior, sequential Beta stopping, Bayes next probe.

Originally reserved for Arnav to hand-write (AGENTS.md rule 1); implemented by an agent at
Arnav's direction on 2026-11-05 against the specs in
`tests/test_bayes_prior_spec.py`. Stdlib only.

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

import math
import random
from dataclasses import dataclass, field

SENSITIVITY_TAUS = (0.3, 0.7)


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


def _bernoulli_loglik(s: float, k: float, p: float) -> float:
    """log p^s (1-p)^(k-s), with 0 log 0 = 0 (counts may be fractional in the bootstrap)."""
    out = 0.0
    if s > 0:
        out += s * math.log(p)
    if k - s > 0:
        out += (k - s) * math.log(1 - p)
    return out


def _change_point(counts: list[tuple[int, float, float]], tau_abs: float) -> int | None:
    """Maximum-likelihood single step over turn-sorted `(t, s, k)` counts, restricted to
    steps whose pooled pre-change level is above `tau_abs` and post-change level at or below
    it. Returns the last pre-change turn, or None when no step qualifies. Ties go to the
    earlier turn."""
    best: tuple[float, int] | None = None
    for i in range(len(counts) - 1):
        s_hi = sum(s for _, s, _ in counts[: i + 1])
        k_hi = sum(k for _, _, k in counts[: i + 1])
        s_lo = sum(s for _, s, _ in counts[i + 1:])
        k_lo = sum(k for _, _, k in counts[i + 1:])
        p_hi, p_lo = s_hi / k_hi, s_lo / k_lo
        if not (p_hi > tau_abs >= p_lo):
            continue
        ll = _bernoulli_loglik(s_hi, k_hi, p_hi) + _bernoulli_loglik(s_lo, k_lo, p_lo)
        if best is None or ll > best[0]:
            best = (ll, counts[i][0])
    return best[1] if best else None


def _percentile(sorted_xs: list[int], q: float) -> int:
    """Nearest-rank percentile of an ascending list."""
    return sorted_xs[min(len(sorted_xs) - 1, max(0, math.ceil(q * len(sorted_xs)) - 1))]


def fit_changepoint_oracle(values: dict[int, tuple[int, int]], *, v_s1: float,
                           tau: float = 0.5, n_boot: int = 1000, seed: int = 0,
                           ) -> ChangePointResult:
    """Primary Q1 oracle (PREREGISTRATION.md, "Primary oracle").

    `values[t] = (s_t, k_t)` are the Monte Carlo continuation outcomes at every probed state
    (thinned grid, candidate windows, K = 16 refinements, optionally `{T+1: (0, K)}`).
    `v_s1` is the Jeffreys-smoothed base value `V~(s_1)` (`stats.oracle.jeffreys_v1`).

    Fit: a single change-point step model over all `sum(k_t)` outcomes, with
    `V(s_u) = p_hi` for `u <= c` and `V(s_u) = p_lo` for `u > c`, `c` ranging over the
    probed turns except the last; `p_hi`, `p_lo` are the pooled maximum-likelihood levels.
    Only steps with `p_hi > tau * v_s1 >= p_lo` qualify (the fitted value is above the
    threshold up to `c` and stays at or below it after), and `t_star` is the qualifying `c`
    with the highest binomial likelihood (ties to the earlier turn). The 90% interval
    `[lo, hi]` comes from `n_boot` Beta-posterior parametric bootstrap replicates
    (`p_t ~ Beta(1 + s_t, 1 + k_t - s_t)` at every probed `t`, re-fit on the fractional
    counts `(p_t * k_t, k_t)`, nearest-rank 5th / 95th percentile of the replicate `t_star`),
    seeded by `seed` through `random.Random`. Replicates with no qualifying step are
    dropped; the interval is widened to contain the point estimate if the percentiles
    exclude it. `t_star_drop` and `sensitivity` (relative thresholds 0.3 and 0.7) are
    computed from the same point estimates; `sensitivity` keys are the relative `tau`.

    Raises `ValueError` if `values` is empty, any `k_t <= 0`, any `s_t` outside `[0, k_t]`,
    or `v_s1 <= 0`.
    """
    if not values:
        raise ValueError("no probed values")
    if v_s1 <= 0:
        raise ValueError(f"v_s1 must be positive, got {v_s1}")
    for t, (s, k) in values.items():
        if k <= 0 or not 0 <= s <= k:
            raise ValueError(f"bad outcome count at turn {t}: {s}/{k}")
    counts = [(t, float(s), float(k)) for t, (s, k) in sorted(values.items())]
    tau_abs = tau * v_s1
    t_star = _change_point(counts, tau_abs)

    v_hat = [(t, s / k) for t, s, k in counts]
    drops = [(v_hat[i][1] - v_hat[i + 1][1], v_hat[i][0]) for i in range(len(v_hat) - 1)]
    t_star_drop = min(drops, key=lambda d: (-d[0], d[1]))[1] if drops else None
    sensitivity = {rel: _change_point(counts, rel * v_s1) for rel in SENSITIVITY_TAUS}

    lo = hi = None
    if t_star is not None:
        rng = random.Random(seed)
        reps = []
        for _ in range(n_boot):
            boot = [(t, rng.betavariate(1 + s, 1 + k - s) * k, k) for t, s, k in counts]
            c = _change_point(boot, tau_abs)
            if c is not None:
                reps.append(c)
        reps.sort()
        lo, hi = t_star, t_star
        if reps:
            lo = min(_percentile(reps, 0.05), t_star)
            hi = max(_percentile(reps, 0.95), t_star)
    return ChangePointResult(t_star, lo, hi, tau_abs, t_star_drop, sensitivity)


@dataclass(frozen=True)
class LaplaceGuardPrior:
    """Prior over `t*` given the first non-terminal guard firing `g`.

    `b`: Laplace scale (turns). `pi0`: false-fire probability, the chance that `g` carries
    no information about `t*` (that branch is uniform over the candidates). `delta`: shift
    of the mode, `t* = g - delta` (guards fire *after* the derailment; calibrated per model,
    the 9B shift is expected to differ from the 4B one). `lam`: a hand-set uniform-mixture
    floor so no candidate turn ever gets zero prior mass; it is not fitted.

    `pmf(c, g, T) = lam / |c| + (1 - lam) * (pi0 / |c| + (1 - pi0) * L(t))` for each `t` in
    the candidate list `c`, where `L` is `Laplace(t; g - delta, b)` renormalised over `c`,
    so the pmf sums to 1 and never drops below `lam / |c|`. With `g` None (no non-terminal
    guard fired) the pmf is uniform over `c`.
    """

    b: float = 2.0
    pi0: float = 0.1
    delta: float = 0.0
    lam: float = 0.05

    def pmf(self, t_star_candidates: list[int], g: int | None, T: int,  # noqa: N803
            ) -> list[float]:
        """Prior probability of each candidate turn in `t_star_candidates` (same order),
        summing to 1. With `pi0 = lam = 0` the mode is `g - delta`. Raises `ValueError` on
        an empty candidate list or a candidate outside `[1, T]`."""
        if not t_star_candidates:
            raise ValueError("no candidate turns")
        if any(not 1 <= t <= T for t in t_star_candidates):
            raise ValueError(f"candidate outside [1, {T}]")
        n = len(t_star_candidates)
        if g is None:
            return [1 / n] * n
        laplace = _laplace_over(t_star_candidates, g - self.delta, self.b)
        return [self.lam / n + (1 - self.lam) * (self.pi0 / n + (1 - self.pi0) * q)
                for q in laplace]

    def fit(self, pairs: list[tuple[int, int, int]], n_em: int = 20) -> LaplaceGuardPrior:
        """Return a new prior calibrated on design-split `(g, t_star, T)` triples (never
        evaluation scenarios). `delta` is the median of `g - t_star`. `b` and `pi0` come
        from `n_em` EM steps over the uniform/Laplace mixture (Laplace renormalised over
        `1..T` per triple): `pi0` is one minus the mean Laplace responsibility, `b` the
        responsibility-weighted mean of `|g - delta - t_star|`, floored at `B_MIN` so a
        perfectly consistent design set cannot collapse the scale to zero. `lam` is carried
        over unchanged. Raises `ValueError` on an empty list."""
        if not pairs:
            raise ValueError("no (g, t_star, T) pairs to fit")
        shifts = sorted(g - t for g, t, _ in pairs)
        mid = len(shifts) // 2
        delta = shifts[mid] if len(shifts) % 2 else (shifts[mid - 1] + shifts[mid]) / 2
        b, pi0 = self.b, self.pi0
        for _ in range(n_em):
            resp = []
            for g, t, T in pairs:  # noqa: N806
                cands = list(range(1, T + 1))
                lap = _laplace_over(cands, g - delta, b)[t - 1]
                num = (1 - pi0) * lap
                resp.append(num / (num + pi0 / T))
            pi0 = 1 - sum(resp) / len(resp)
            weighted = sum(r * abs(g - delta - t) for r, (g, t, _) in zip(resp, pairs,
                                                                         strict=True))
            b = max(B_MIN, weighted / sum(resp)) if sum(resp) > 0 else self.b
        return LaplaceGuardPrior(b=b, pi0=pi0, delta=delta, lam=self.lam)


B_MIN = 0.1  # turns; lower bound on the fitted Laplace scale


def _laplace_over(cands: list[int], mode: float, b: float) -> list[float]:
    """Laplace(t; mode, b) at each candidate, renormalised to sum to 1 over `cands`."""
    w = [math.exp(-abs(t - mode) / b) for t in cands]
    z = sum(w)
    return [x / z for x in w]


def _beta_cdf_int(x: float, a: int, b: int) -> float:
    """CDF of Beta(a, b) at x for integer a, b >= 1: P(Binomial(a + b - 1, x) >= a)."""
    n = a + b - 1
    return sum(math.comb(n, j) * x**j * (1 - x) ** (n - j) for j in range(a, n + 1))


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
    if not 0 <= successes <= trials:
        raise ValueError(f"need 0 <= successes <= trials, got {successes}/{trials}")
    if not 0 < tau < 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    if not 0 < delta < 0.5:
        raise ValueError(f"delta must be in (0, 0.5), got {delta}")
    p_below = _beta_cdf_int(tau, 1 + successes, 1 + trials - successes)
    return p_below >= 1 - delta or p_below <= delta


def _entropy(ps: list[float]) -> float:
    z = sum(ps)
    return -sum(p / z * math.log(p / z) for p in ps if p > 0) if z > 0 else 0.0


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
    if len(posterior) != T:
        raise ValueError(f"posterior has length {len(posterior)}, expected T = {T}")
    if max(posterior) >= sum(posterior) - 1e-12:
        return None
    best: tuple[float, int] | None = None
    for t in range(1, T + 1):
        if t in probed:
            continue
        early, late = posterior[: t - 1], posterior[t - 1:]  # t* < t collapsed, t* >= t alive
        p_late = sum(late)
        p_early = sum(early)
        expected = p_early * _entropy(early) + p_late * _entropy(late)
        if best is None or expected < best[0] - 1e-12:
            best = (expected, t)
    return best[1] if best else None
