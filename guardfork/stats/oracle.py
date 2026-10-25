"""Monte Carlo derailment oracle: Jeffreys V~(s1), t*, argmax-drop sensitivity."""

from __future__ import annotations


def jeffreys_v1(passes: int, n: int) -> float:
    """Jeffreys-smoothed base value V~(s_1) = (passes + 0.5) / (n + 1)."""
    if n < 0 or passes < 0 or passes > n:
        raise ValueError(f"need 0 <= passes <= n, got {passes}/{n}")
    return (passes + 0.5) / (n + 1)


def _adjacent_pairs(values: dict[int, float], *, reward: float, T: int  # noqa: N803
                    ) -> list[tuple[int, float, float]]:
    """(t, V(s_t), V(s_{t+1})) over adjacent probed pairs, using V(s_{T+1}) = reward."""
    v = dict(values)
    v[T + 1] = reward
    states = sorted(v)
    out = []
    for t, nxt in zip(states, states[1:], strict=False):
        if nxt == t + 1:
            out.append((t, v[t], v[nxt]))
    return out


def derailment_turn(values: dict[int, float], *, reward: float, v_s1: float, K: int,  # noqa: N803
                    T: int) -> int | None:  # noqa: N803
    """t* = min{t : V(s_{t+1}) <= 0.5 V~(s_1) and V(s_t) - V(s_{t+1}) >= 2/K} over
    adjacent probed pairs, with V(s_{T+1}) = reward. None if undefined."""
    if K <= 0:
        raise ValueError("K must be positive")
    threshold = 0.5 * v_s1
    min_drop = 2 / K
    qualifying = [
        t for t, v_t, v_next in _adjacent_pairs(values, reward=reward, T=T)
        if v_next <= threshold and (v_t - v_next) >= min_drop and 1 <= t <= T
    ]
    return min(qualifying) if qualifying else None


def argmax_drop_turn(values: dict[int, float], *, reward: float, T: int) -> int | None:  # noqa: N803
    """Sensitivity: t maximizing V(s_t) - V(s_{t+1}) over adjacent probed pairs."""
    pairs = [(t, v_t - v_next) for t, v_t, v_next in _adjacent_pairs(values, reward=reward, T=T)
             if 1 <= t <= T]
    if not pairs:
        return None
    best_drop = max(drop for _, drop in pairs)
    return min(t for t, drop in pairs if drop == best_drop)
