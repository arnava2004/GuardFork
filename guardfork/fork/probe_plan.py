"""Monte Carlo probe schedules: the v2 candidate-augmented grid (`candidate_grid`, used by
the referee) and the v1 guard window + bisection (`initial_probes`, `next_probe`; kept for
the simulated `bisection_*` search policies and the v1 sensitivity analysis)."""

from __future__ import annotations

from collections.abc import Iterable


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


def candidate_grid(T: int, *, step: int, guard_turns: Iterable[int | None] = (),  # noqa: N803
                   judge_turn: int | None = None, entropy_turn: int | None = None,
                   judge_turns: Iterable[int | None] = ()) -> list[int]:
    """Thinned grid every `step` turns from 1, plus {x-1, x, x+1} windows around every guard
    first firing, every judge pick (`judge_turns`, one per judge slot; `judge_turn` kept for
    compatibility) and the entropy peak, clipped to [1, T], sorted."""
    if T < 1 or step < 1:
        raise ValueError("need T >= 1 and step >= 1")
    grid = set(range(1, T + 1, step))
    for x in (*guard_turns, judge_turn, *judge_turns, entropy_turn):
        if x is None:
            continue
        grid.update(u for u in (x - 1, x, x + 1) if 1 <= u <= T)
    return sorted(grid)
