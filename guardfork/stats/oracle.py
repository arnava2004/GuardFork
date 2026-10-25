"""Monte Carlo derailment oracle. HAND-WRITTEN BY ARNAV (AGENTS.md rule 1)."""

from __future__ import annotations


def jeffreys_v1(passes: int, n: int) -> float:
    """Jeffreys-smoothed base value V~(s_1) = (passes + 0.5) / (n + 1)."""
    raise NotImplementedError("Arnav: stats/oracle.py")


def derailment_turn(values: dict[int, float], *, reward: float, v_s1: float, K: int,  # noqa: N803
                    T: int) -> int | None:  # noqa: N803
    """t* = min{t : V(s_{t+1}) <= 0.5 V~(s_1) and V(s_t) - V(s_{t+1}) >= 2/K} over
    adjacent probed pairs, with V(s_{T+1}) = reward. None if undefined."""
    raise NotImplementedError("Arnav: stats/oracle.py")


def argmax_drop_turn(values: dict[int, float], *, reward: float, T: int) -> int | None:  # noqa: N803
    """Sensitivity: t maximizing V(s_t) - V(s_{t+1}) over adjacent probed pairs."""
    raise NotImplementedError("Arnav: stats/oracle.py")
