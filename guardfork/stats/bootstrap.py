"""Localization inference. HAND-WRITTEN BY ARNAV (AGENTS.md rule 1)."""

from __future__ import annotations

from collections.abc import Sequence


def analytic_random_hit(T: int, t_star: int, w: int) -> float:  # noqa: N803
    """|{t in [1, T-1] : |t - t*| <= w}| / (T - 1)."""
    raise NotImplementedError("Arnav: stats/bootstrap.py")


def hit_minus_random(g: int, t_star: int, T: int, w: int) -> float:  # noqa: N803
    """d_i(w) = 1(|g - t*| <= w) - analytic_random_hit(T, t*, w)."""
    raise NotImplementedError("Arnav: stats/bootstrap.py")


def clustered_bootstrap_ci(values: Sequence[float], clusters: Sequence[str], *,
                           n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
                           ) -> tuple[float, float, float]:
    """(mean, lo, hi), resampling whole scenarios with replacement."""
    raise NotImplementedError("Arnav: stats/bootstrap.py")
