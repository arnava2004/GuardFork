"""Oracle-scored step credit (FRONTIER_TANGENTS.md, follow-up plan item 1).

For a failed run with per-turn Monte Carlo values V_hat(s_t) and oracle t*, every
zero-rollout credit rule produces per-turn weights, scored against Delta_t^MC =
V_hat(s_t) - V_hat(s_{t+1}) on pitfall recall, prefix preservation and Spearman rank
correlation, overall and by trajectory third.

Weight conventions. Rules return `dict[int, float]` over turns 1..T. For the [0, 1] rules
(`uniform_grpo`, `hard_mask_prefix`, `open_judge`, `soft_posterior_pw`) a weight is the
*blame* multiplier applied to the failed run's sequence-level advantage: 1 = fully blamed,
0 = spared. `c3_rule_flip` is signed: +1 keeps the sequence advantage, -1 flips it at
rule-detected invalid-action / loop turns. `to_blame` maps any rule output into [0, 1]
blame for scoring (signed rules: flipped turns -> 1, others -> 0, i.e. the rule points at
the flagged turns; non-negative rules are clipped to [0, 1]). All stdlib.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from guardfork.guards.rules import Guard, GuardFiring

FLIP_GUARDS = frozenset({Guard.G1_INVALID_ACTION, Guard.G2_LOOP})


# ---------------------------------------------------------------------------- ground truth


def delta_mc(values: Mapping[int, float], reward: float, T: int) -> dict[int, float]:  # noqa: N803
    """Delta_t = V_hat(s_t) - V_hat(s_{t+1}) over adjacent probed turns, with
    V_hat(s_{T+1}) = reward. Turns whose successor was not probed are absent."""
    if T < 1:
        raise ValueError("T must be >= 1")
    v = {t: float(x) for t, x in values.items() if 1 <= t <= T}
    v[T + 1] = float(reward)
    return {t: v[t] - v[t + 1] for t in sorted(v) if t <= T and (t + 1) in v}


# ---------------------------------------------------------------------------- credit rules


def _check_T(T: int) -> None:  # noqa: N802, N803
    if T < 1:
        raise ValueError("T must be >= 1")


def uniform_grpo(T: int) -> dict[int, float]:  # noqa: N803
    """GRPO-style: every turn carries the whole sequence advantage."""
    _check_T(T)
    return dict.fromkeys(range(1, T + 1), 1.0)


def hard_mask_prefix(T: int, t_cut: int | None) -> dict[int, float]:  # noqa: N803
    """Critical-State / PivoARL style: weight 0 before `t_cut`, 1 from `t_cut` on.
    `t_cut` None (no localizer pick) falls back to uniform."""
    _check_T(T)
    if t_cut is None:
        return uniform_grpo(T)
    return {t: (1.0 if t >= t_cut else 0.0) for t in range(1, T + 1)}


def c3_rule_flip(T: int, firings: Iterable[GuardFiring],  # noqa: N803
                 flip: frozenset[Guard] = FLIP_GUARDS) -> dict[int, float]:
    """C^3 style signed multipliers: +1 everywhere, -1 (advantage sign flip) at turns where
    a rule-detected invalid action (G1) or loop (G2) fired.

    # VERIFY: re-read C^3 (arXiv 2610.19695) for whether the flip applies to the per-turn
    # advantage on failed runs too (here it marks the flagged turns; `to_blame` scores the
    # rule as pointing at them) before quoting it as "C^3" in a table.
    """
    _check_T(T)
    flipped = {f.turn for f in firings if f.guard in flip and 1 <= f.turn <= T}
    return {t: (-1.0 if t in flipped else 1.0) for t in range(1, T + 1)}


def open_judge(T: int, judge_turn: int | None) -> dict[int, float]:  # noqa: N803
    """ECHO-style: the open-weight judge's pick j plays the role of the cut; turns from j on
    are blamed, the prefix is spared. Uniform when the judge returned nothing.

    # VERIFY: ECHO (arXiv 2610.16027) credit shape before labelling this "ECHO-style".
    """
    return hard_mask_prefix(T, judge_turn)


def soft_posterior_pw(T: int, posterior_over_t_star: Mapping[int, float] | Sequence[float],  # noqa: N803
                      ) -> dict[int, float]:
    """GuardFork: weight_t = P(t* <= t) under the (Laplace guard) posterior over t*.
    Accepts a dict keyed by turn or a length-T sequence (`posterior[t-1]`); renormalised."""
    _check_T(T)
    if isinstance(posterior_over_t_star, Mapping):
        mass = {int(t): float(p) for t, p in posterior_over_t_star.items()}
    else:
        if len(posterior_over_t_star) != T:
            raise ValueError("posterior sequence must have length T")
        mass = {t: float(p) for t, p in enumerate(posterior_over_t_star, start=1)}
    if any(p < 0 for p in mass.values()):
        raise ValueError("posterior mass must be non-negative")
    z = sum(mass.values())
    if z <= 0:
        raise ValueError("posterior has no mass")
    out, cum = {}, 0.0
    for t in range(1, T + 1):
        cum += mass.get(t, 0.0) / z
        out[t] = min(1.0, cum)
    return out


def to_blame(weights: Mapping[int, float]) -> dict[int, float]:
    """Blame in [0, 1] for scoring (see module docstring)."""
    if any(w < 0 for w in weights.values()):
        return {t: (1.0 if w < 0 else 0.0) for t, w in weights.items()}
    return {t: min(1.0, max(0.0, float(w))) for t, w in weights.items()}


# ---------------------------------------------------------------------------- scoring


def _top_k(scores: Mapping[int, float], k: int) -> set[int]:
    """The `k` turns with the largest score; ties resolve toward earlier turns."""
    order = sorted(scores, key=lambda t: (-scores[t], t))
    return set(order[:k])


def blame_onset(weights: Mapping[int, float]) -> dict[int, float]:
    """Where blame *starts*: `max(0, blame_t - blame_{t-1})` with `blame_0 = 0`. For a
    hard mask this is a unit spike at the cut, for `soft_posterior_pw` it is the posterior
    pmf itself, for uniform it is a spike at turn 1."""
    blame = to_blame(weights)
    out, prev = {}, 0.0
    for t in sorted(blame):
        out[t] = max(0.0, blame[t] - prev)
        prev = blame[t]
    return out


def pitfall_recall(weights: Mapping[int, float], delta: Mapping[int, float], top_k: int,
                   ) -> float | None:
    """Fraction of the `top_k` largest-drop turns (by Delta_t^MC) that are also among the
    `top_k` largest blame onsets (`blame_onset`; cumulative weights would otherwise always
    point at the tail). Compared over turns with a defined Delta only; ties resolve toward
    earlier turns. None when no turn is comparable or `top_k < 1`."""
    if top_k < 1:
        return None
    onset = blame_onset(weights)
    common = {t: delta[t] for t in delta if t in onset}
    if not common:
        return None
    k = min(top_k, len(common))
    pitfalls = _top_k(common, k)
    blamed = _top_k({t: onset[t] for t in common}, k)
    return len(pitfalls & blamed) / len(pitfalls)


def prefix_preservation(weights: Mapping[int, float], t_star: int | None) -> float | None:
    """1 - mean blame over the pre-derailment prefix (turns < t*): 1.0 when the prefix is
    fully spared, 0.0 when it is fully blamed. 1.0 for t* == 1 (empty prefix); None when
    t* is undefined."""
    if t_star is None:
        return None
    blame = to_blame(weights)
    prefix = [blame[t] for t in blame if t < t_star]
    if not prefix:
        return 1.0
    return 1.0 - sum(prefix) / len(prefix)


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation with average ranks for ties; None when fewer than 2 pairs
    or either side is constant."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    return sxy / (sxx * syy) ** 0.5


def rank_corr(weights: Mapping[int, float], delta: Mapping[int, float]) -> float | None:
    """Spearman correlation between blame and Delta_t^MC over the common turns."""
    blame = to_blame(weights)
    turns = sorted(t for t in delta if t in blame)
    return spearman([blame[t] for t in turns], [delta[t] for t in turns])


def thirds(T: int) -> dict[str, range]:  # noqa: N803
    """Contiguous first / middle / last third of turns 1..T (remainder goes to later
    thirds, so the last third is never shorter than the first)."""
    _check_T(T)
    base, extra = divmod(T, 3)
    n1 = base
    n2 = base + (1 if extra == 2 else 0)
    return {
        "first": range(1, n1 + 1),
        "middle": range(n1 + 1, n1 + n2 + 1),
        "last": range(n1 + n2 + 1, T + 1),
    }


def score_by_third(weights: Mapping[int, float], delta: Mapping[int, float],
                   t_star: int | None, T: int, *, top_k: int = 2) -> dict[str, dict]:  # noqa: N803
    """`pitfall_recall`, `rank_corr` and the number of comparable turns, overall and
    restricted to each trajectory third; `prefix_preservation` and `t_star_third`
    (which third t* falls in, None when undefined) at the top level."""
    out: dict[str, dict] = {
        "overall": {
            "n": len([t for t in delta if t in weights]),
            "pitfall_recall": pitfall_recall(weights, delta, top_k),
            "rank_corr": rank_corr(weights, delta),
            "prefix_preservation": prefix_preservation(weights, t_star),
            "t_star_third": None,
        },
    }
    for name, turns in thirds(T).items():
        # Restrict the ground truth to the third; keep the full weights so blame onsets
        # at the third's first turn are measured against the previous turn, not zero.
        d = {t: delta[t] for t in turns if t in delta and t in weights}
        out[name] = {
            "n": len(d),
            "pitfall_recall": pitfall_recall(weights, d, top_k) if d else None,
            "rank_corr": rank_corr(weights, d) if d else None,
        }
        if t_star is not None and t_star in turns:
            out["overall"]["t_star_third"] = name
    return out
