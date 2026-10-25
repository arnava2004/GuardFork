"""Monte Carlo probe schedule around the first non-terminal guard."""

from __future__ import annotations


def initial_probes(g: int, T: int) -> list[int]:  # noqa: N803
    """{0, g-3, g-1, g, g+1} clipped to [1, T], deduplicated, ascending."""
    raw = {0, g - 3, g - 1, g, g + 1}
    return sorted({max(1, min(T, t)) for t in raw})


def next_probe(values: dict[int, float], v_s1: float, K: int, T: int) -> int | None:  # noqa: N803
    """Next state to probe by bisection until the crossing interval is adjacent, or None
    when done. `values[t]` is V_hat(s_t); V_hat(s_{T+1}) is the run's reward (0)."""
    if K <= 0:
        raise ValueError("K must be positive")
    states = sorted(t for t in values if 1 <= t <= T)
    if 1 not in values:
        return 1
    threshold = 0.5 * v_s1
    min_drop = 2 / K
    for lo, hi in zip(states, states[1:], strict=False):
        if hi - lo <= 1:
            continue
        if values[hi] <= threshold or (values[lo] - values[hi]) >= min_drop:
            return (lo + hi) // 2
    last = states[-1]
    if last < T and values[last] > threshold:
        return (last + T) // 2
    return None
