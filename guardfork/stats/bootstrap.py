"""Localization inference: analytic random hit, clustered bootstrap, Holm."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence


def analytic_random_hit(T: int, t_star: int, w: int) -> float:  # noqa: N803
    """|{t in [1, T-1] : |t - t*| <= w}| / (T - 1)."""
    if T <= 1:
        raise ValueError("T must be > 1 for a non-terminal horizon")
    n = sum(1 for t in range(1, T) if abs(t - t_star) <= w)
    return n / (T - 1)


def hit_minus_random(g: int, t_star: int, T: int, w: int) -> float:  # noqa: N803
    """d_i(w) = 1(|g - t*| <= w) - analytic_random_hit(T, t*, w)."""
    return float(abs(g - t_star) <= w) - analytic_random_hit(T, t_star, w)


def _percentile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    if not ys:
        raise ValueError("empty sample")
    pos = q * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def clustered_bootstrap_ci(values: Sequence[float], clusters: Sequence[str], *,
                           n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
                           ) -> tuple[float, float, float]:
    """(mean, lo, hi), resampling whole scenarios with replacement."""
    if len(values) != len(clusters) or not values:
        raise ValueError("values and clusters must be non-empty and aligned")
    mean = sum(values) / len(values)
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(clusters):
        by_cluster[c].append(i)
    keys = list(by_cluster)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        drawn = [rng.choice(keys) for _ in keys]
        idx = [i for k in drawn for i in by_cluster[k]]
        samples.append(sum(values[i] for i in idx) / len(idx))
    lo = _percentile(samples, alpha / 2)
    hi = _percentile(samples, 1 - alpha / 2)
    return mean, lo, hi


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the original order."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (n - rank) * pvalues[i])
        running = max(running, adj)
        adjusted[i] = running
    return adjusted
