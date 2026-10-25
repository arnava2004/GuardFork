"""Monte Carlo probe schedule. HAND-WRITTEN BY ARNAV (AGENTS.md rule 1)."""

from __future__ import annotations


def initial_probes(g: int, T: int) -> list[int]:  # noqa: N803
    """{0, g-3, g-1, g, g+1} clipped to [1, T], deduplicated, ascending."""
    raise NotImplementedError("Arnav: fork/probe_plan.py")


def next_probe(values: dict[int, float], v_s1: float, K: int, T: int) -> int | None:  # noqa: N803
    """Next state to probe by bisection until the crossing interval is adjacent, or None
    when done. `values[t]` is V_hat(s_t); V_hat(s_{T+1}) is the run's reward (0)."""
    raise NotImplementedError("Arnav: fork/probe_plan.py")
