"""Offline search-policy simulator over a probed value table (PREREGISTRATION.md,
"Screening and probing" / "Localizers compared").

Every localizer is a pure function `(SearchInputs, ProbeTable) -> int | None`. Zero-rollout
localizers (guards, judge, entropy peak, error marker, random, last turn) never touch the
table; search policies (bisection, ELPO-style, guard_bayesect) ask the table for
continuation outcomes and are charged for them. A `ProbeTable` only knows the probed set P;
asking for an unmeasured turn snaps to `argmin_u (|u - t|, u)` (deterministic snapping).

Simulator unbiasedness (pre-registered): on K = 16 states use `split_half` (8 rollouts define
t*, 8 independent ones drive the search); elsewhere draw `beta_bootstrap_values` and fresh
binomial outcomes with `binomial_table`.

`guard_bayesect` calls `guardfork.stats.bayes_prior`, which Arnav hand-writes; it raises
`NotImplementedError` until then (tests mock the module).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from guardfork.fork.probe_plan import candidate_grid, next_probe  # noqa: F401 (re-export)
from guardfork.guards.rules import GuardFiring, evaluate
from guardfork.q1.localization import localizer_turns
from guardfork.stats import bayes_prior
from guardfork.telemetry.proxy_log import ProxyRecord
from guardfork.types import Trajectory

ERROR_MARKERS = ("Traceback", "Execution failed", "Response status code is 4", "Error:")
WINDOWS = (0, 1, 2, 3)


# ---------------------------------------------------------------------------- grid and snapping


def snap(t: int, probed: Iterable[int]) -> int:
    """Probed turn closest to `t`; ties go to the smaller turn."""
    pool = list(probed)
    if not pool:
        raise ValueError("no probed turns to snap to")
    return min(pool, key=lambda u: (abs(u - t), u))


# ---------------------------------------------------------------------------- entropy peak


def _completion_logprobs(record: ProxyRecord | Mapping[str, Any]) -> list[float] | None:
    raw = record.raw if isinstance(record, ProxyRecord) else record
    per_choice = raw.get("completion_logprobs")
    if not isinstance(per_choice, list) or not per_choice:
        return None
    first = per_choice[0]
    if not isinstance(first, list):
        return None
    vals = [float(x) for x in first if isinstance(x, int | float) and not isinstance(x, bool)]
    return vals or None


def turn_surprisal(records: Iterable[ProxyRecord | Mapping[str, Any]]) -> dict[int, float]:
    """Per-turn mean token surprisal `mean(-logprob)` over the completion tokens of the
    sampled continuation, from the proxy's logged `completion_logprobs`.

    This is a *surprisal proxy*, not the full-distribution token entropy: the proxy logs
    only the sampled token's logprob (`logprobs=True`, no `top_logprobs`), so the
    expectation over the vocabulary is unavailable. If a later proxy version logs
    `top_logprobs`, replace this with the truncated-distribution entropy. The latest
    successful record for a turn wins (retries overwrite)."""
    out: dict[int, float] = {}
    for rec in records:
        pr = rec if isinstance(rec, ProxyRecord) else ProxyRecord(dict(rec))
        if pr.raw.get("status") not in (None, 200):
            continue
        turn = pr.live_turn_index + 1
        if turn < 1:
            continue
        lps = _completion_logprobs(pr)
        if lps is None:
            continue
        out[turn] = -sum(lps) / len(lps)
    return out


def entropy_peak_turn(proxy_records_for_run: Iterable[ProxyRecord | Mapping[str, Any]],
                      ) -> int | None:
    """Turn with the highest mean token surprisal (see `turn_surprisal`); ties go to the
    earliest turn. None when no record carries completion logprobs."""
    surprisal = turn_surprisal(proxy_records_for_run)
    if not surprisal:
        return None
    return min(surprisal, key=lambda t: (-surprisal[t], t))


# ---------------------------------------------------------------------------- resampling


def beta_bootstrap_values(values: Mapping[int, tuple[int, int]], rng: random.Random,
                          ) -> dict[int, float]:
    """`p_t ~ Beta(1 + s_t, 1 + k_t - s_t)` for every probed turn (stdlib betavariate)."""
    out = {}
    for t in sorted(values):
        s, k = values[t]
        if not 0 <= s <= k or k <= 0:
            raise ValueError(f"bad outcome count at turn {t}: {s}/{k}")
        out[t] = rng.betavariate(1 + s, 1 + k - s)
    return out


def binomial_table(p_by_turn: Mapping[int, float], K: int, rng: random.Random,  # noqa: N803
                   ) -> dict[int, list[int]]:
    """Fresh per-rollout 0/1 outcomes, `K` per turn, drawn from `p_by_turn`."""
    if K <= 0:
        raise ValueError("K must be positive")
    return {t: [int(rng.random() < p_by_turn[t]) for _ in range(K)] for t in sorted(p_by_turn)}


def split_half(outcomes_k16: Mapping[int, list[int]], rng: random.Random,
               ) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """Split the 16 per-rollout outcomes at every turn into two independent 8/8 halves
    `(half_a, half_b)` as `(successes, rollouts)` counts. Requires per-rollout 0/1 outcomes
    (aggregate counts cannot be split); every turn must have an even count (16 by
    pre-registration; other even counts split in half)."""
    half_a: dict[int, tuple[int, int]] = {}
    half_b: dict[int, tuple[int, int]] = {}
    for t in sorted(outcomes_k16):
        xs = list(outcomes_k16[t])
        if len(xs) < 2 or len(xs) % 2:
            raise ValueError(f"turn {t}: need an even number (>= 2) of outcomes, got {len(xs)}")
        idx = list(range(len(xs)))
        rng.shuffle(idx)
        half = len(xs) // 2
        a = [xs[i] for i in idx[:half]]
        b = [xs[i] for i in idx[half:]]
        half_a[t] = (sum(a), half)
        half_b[t] = (sum(b), half)
    return half_a, half_b


# ---------------------------------------------------------------------------- probe table


class ProbeTable(Protocol):
    """What a search policy may see: `T`, snapped continuation outcomes, and its bill."""

    @property
    def T(self) -> int: ...  # noqa: N802

    @property
    def continuations_used(self) -> int: ...

    def value(self, t: int, k_max: int | None = None) -> tuple[int, int]:
        """`(successes, rollouts)` at the probed turn nearest to `t` (deterministic
        snapping), using at most the first `k_max` rollouts (all when None). Charges the
        continuation budget once per newly consumed rollout."""
        ...

    def reset(self) -> None: ...


class SnappingProbeTable:
    """`ProbeTable` over a probed set P.

    Built from aggregate counts `values[t] = (s, k)` or per-rollout `outcomes[t] = [0/1,
    ...]`. With aggregates only, the per-rollout order is a seeded shuffle of `s` ones and
    `k - s` zeros so `k_max` prefixes are well defined and reproducible."""

    def __init__(self, T: int, *, values: Mapping[int, tuple[int, int]] | None = None,  # noqa: N803
                 outcomes: Mapping[int, list[int]] | None = None, seed: int = 0) -> None:
        if (values is None) == (outcomes is None):
            raise ValueError("give exactly one of values= or outcomes=")
        if T < 1:
            raise ValueError("T must be >= 1")
        self._T = T
        rng = random.Random(seed)
        self._outcomes: dict[int, list[int]] = {}
        if outcomes is not None:
            for t, xs in outcomes.items():
                self._outcomes[t] = [int(bool(x)) for x in xs]
        else:
            assert values is not None
            for t, (s, k) in values.items():
                if not 0 <= s <= k or k <= 0:
                    raise ValueError(f"bad outcome count at turn {t}: {s}/{k}")
                xs = [1] * s + [0] * (k - s)
                rng.shuffle(xs)
                self._outcomes[t] = xs
        if not self._outcomes:
            raise ValueError("empty probe table")
        self._consumed: dict[int, int] = {}

    @property
    def T(self) -> int:  # noqa: N802
        return self._T

    @property
    def probed(self) -> list[int]:
        return sorted(self._outcomes)

    @property
    def continuations_used(self) -> int:
        return sum(self._consumed.values())

    @property
    def asked(self) -> list[int]:
        """Snapped turns whose outcomes were consumed so far, ascending."""
        return sorted(self._consumed)

    def value(self, t: int, k_max: int | None = None) -> tuple[int, int]:
        u = snap(t, self._outcomes)
        xs = self._outcomes[u]
        k = len(xs) if k_max is None else min(len(xs), max(0, k_max))
        used = self._consumed.get(u, 0)
        if k > used:
            self._consumed[u] = k
        k = max(k, used)
        return sum(xs[:k]), k

    def reset(self) -> None:
        self._consumed.clear()


# ---------------------------------------------------------------------------- policies

OPEN_JUDGE_SLOT = "judge_open"
CLAUDE_JUDGE_SLOT = "judge_claude"
JUDGE_LIKELIHOOD_FLOOR = 0.05  # uniform mixture so no turn gets zero judge likelihood
# Rows that may never feed training or checkpoint selection (AGENTS.md rule 3).
POLICY_FLAGS: dict[str, dict[str, bool]] = {
    "claude_judge": {"eval_only": True},
    "guard_bayesect_claude_prior": {"eval_only": True},
}


@dataclass
class SearchInputs:
    """Everything a localizer may know about one failed run besides the probe table."""

    traj: Trajectory
    firings: list[GuardFiring] = field(default_factory=list)
    judge_turn: int | None = None  # legacy single judge pick (open judge)
    entropy_turn: int | None = None
    v_s1: float = 0.5
    seed: int = 0
    judge_turns: dict[str, int | None] = field(default_factory=dict)  # slot -> argmax pick
    judge_dists: dict[str, list[float]] = field(default_factory=dict)  # slot -> P(t*) 1..T

    @classmethod
    def from_trajectory(cls, traj: Trajectory, **kw: Any) -> SearchInputs:
        """Evaluate the frozen guards (`guards-v1`) on `traj` and keep the firings."""
        return cls(traj, list(evaluate(traj)), **kw)

    @property
    def T(self) -> int:  # noqa: N802
        return self.traj.T

    def judge_pick(self, slot: str) -> int | None:
        """Argmax pick of a judge slot; `judge_open` falls back to the legacy `judge_turn`."""
        if slot in self.judge_turns:
            return self.judge_turns[slot]
        if slot == OPEN_JUDGE_SLOT:
            return self.judge_turn
        return None

    def judge_dist(self, slot: str) -> list[float] | None:
        """Distribution of a judge slot when it has length T (None otherwise)."""
        dist = self.judge_dists.get(slot)
        return list(dist) if dist is not None and len(dist) == self.T else None


Policy = Callable[[SearchInputs, ProbeTable], int | None]


def random_turn(inputs: SearchInputs, table: ProbeTable) -> int | None:
    """Uniform non-terminal turn in [1, T-1] (T itself when T == 1), seeded."""
    T = inputs.T  # noqa: N806
    return random.Random(inputs.seed).randint(1, max(1, T - 1))


def last_turn(inputs: SearchInputs, table: ProbeTable) -> int | None:
    return inputs.T


def entropy_peak(inputs: SearchInputs, table: ProbeTable) -> int | None:
    return inputs.entropy_turn


def guards_raw(inputs: SearchInputs, table: ProbeTable) -> int | None:
    """First non-terminal G1–G5 firing (`localizer_turns(...)['all']`, frozen rules)."""
    return localizer_turns(inputs.traj, inputs.firings)["all"]


def guards_recovery_aware(inputs: SearchInputs, table: ProbeTable) -> int | None:
    """`g3_unrecovered`: G3 counts only if the next turns neither read docs nor log in."""
    return localizer_turns(inputs.traj, inputs.firings)["g3_unrecovered"]


def open_judge(inputs: SearchInputs, table: ProbeTable) -> int | None:
    """Argmax pick of `judge_open` (legacy `judge_turn` when no slot dict is given)."""
    return inputs.judge_pick(OPEN_JUDGE_SLOT)


def judge_slot_pick(slot: str) -> Policy:
    """Zero-rollout policy returning the argmax pick of one judge slot."""

    def policy(inputs: SearchInputs, table: ProbeTable) -> int | None:
        return inputs.judge_pick(slot)

    policy.__name__ = f"{slot}_pick"
    return policy


def claude_judge(inputs: SearchInputs, table: ProbeTable) -> int | None:
    """Eval-only row (`POLICY_FLAGS`): argmax pick of `judge_claude`."""
    return inputs.judge_pick(CLAUDE_JUDGE_SLOT)


def lvrl_style_error_marker(inputs: SearchInputs, table: ProbeTable) -> int | None:
    """Floor: the first live turn whose observation contains any of `ERROR_MARKERS`
    (our own explicit strings; LVRL's exact strings are not claimed)."""
    for turn in inputs.traj.turns:
        obs = turn.observation or ""
        if any(m in obs for m in ERROR_MARKERS):
            return turn.turn
    return None


def _threshold(inputs: SearchInputs, tau: float) -> float:
    return tau * inputs.v_s1


def _first_collapsed(v_hat: Mapping[int, float], tau_abs: float, T: int) -> int | None:  # noqa: N803
    """t* read off a probed table: the last alive turn before the first collapsed probed
    turn such that every later probed turn stays collapsed. T when only the terminal
    reward is below the threshold."""
    turns = sorted(t for t in v_hat if 1 <= t <= T)
    collapsed = [t for t in turns if v_hat[t] <= tau_abs]
    for c in collapsed:
        if all(v_hat[u] <= tau_abs for u in turns if u >= c):
            alive = [t for t in turns if t < c and v_hat[t] > tau_abs]
            return alive[-1] if alive else None
    return T if any(v_hat[t] > tau_abs for t in turns) else None


def bisection_fixed(K: int, *, tau: float = 0.5, max_probes: int = 12) -> Policy:  # noqa: N803
    """Probe s_1 with K rollouts, then bisect with `probe_plan.next_probe` until the crossing
    interval is adjacent or `max_probes` turns were asked. Always spends the full K per
    asked turn; returns the first-collapsed turn read off the probed values."""
    if K <= 0:
        raise ValueError("K must be positive")

    def policy(inputs: SearchInputs, table: ProbeTable) -> int | None:
        T = inputs.T  # noqa: N806
        v_hat: dict[int, float] = {}
        asked: set[int] = set()

        def ask(t: int) -> None:
            s, k = table.value(t, k_max=K)
            u = snap(t, _probed_from(table, t))
            asked.add(u)
            v_hat[u] = s / k if k else 0.0

        ask(1)
        while len(asked) < max_probes:
            nxt = next_probe(v_hat, inputs.v_s1, K, T)
            if nxt is None:
                break
            before = len(asked)
            ask(nxt)
            if len(asked) == before:  # snapped onto an already-asked turn: no progress
                break
        return _first_collapsed(v_hat, _threshold(inputs, tau), T)

    return policy


def _probed_from(table: ProbeTable, t: int) -> list[int]:
    probed = getattr(table, "probed", None)
    return list(probed) if probed else [t]


def bisection_early_stop(K: int, *, tau: float = 0.5, max_probes: int = 12) -> Policy:  # noqa: N803
    """Bisection with per-probe early stopping on the collapsed side: ask for K // 2
    rollouts first and stop at that turn when none succeeds (0/h, clearly collapsed);
    otherwise top up to K. Alive states always get the full K so the half-sample does not
    inflate V_hat and trigger spurious drops in `next_probe`. Same bracket logic as
    `bisection_fixed`, cheaper when collapsed states are clear-cut."""
    if K <= 1:
        raise ValueError("K must be >= 2")
    half = K // 2

    def policy(inputs: SearchInputs, table: ProbeTable) -> int | None:
        T = inputs.T  # noqa: N806
        v_hat: dict[int, float] = {}
        asked: set[int] = set()

        def ask(t: int) -> None:
            s, k = table.value(t, k_max=half)
            if s > 0:
                s, k = table.value(t, k_max=K)
            u = snap(t, _probed_from(table, t))
            asked.add(u)
            v_hat[u] = s / k if k else 0.0

        ask(1)
        while len(asked) < max_probes:
            nxt = next_probe(v_hat, inputs.v_s1, K, T)
            if nxt is None:
                break
            before = len(asked)
            ask(nxt)
            if len(asked) == before:
                break
        return _first_collapsed(v_hat, _threshold(inputs, tau), T)

    return policy


def elpo_bel(K: int, *, tau: float = 0.5, max_probes: int = 12) -> Policy:  # noqa: N803
    """Simplified ELPO-style binary search with early stop.

    Treats each probed state as a binary oracle (`V_hat <= tau * v_s1` -> collapsed) with a
    small budget `K // 2` per probe, bisects the bracket [last alive, first collapsed]
    starting from (1, T+1), stops as soon as the bracket is adjacent, then confirms the
    collapsed side with the full K. Returns the last alive turn of the bracket.

    # VERIFY: against ELPO (arXiv 2602.09598) before any table calls this "ELPO"; the
    # paper's exact rollout tree, budget schedule and stopping rule were not re-read here.
    """
    if K <= 1:
        raise ValueError("K must be >= 2")
    small = max(1, K // 2)

    def policy(inputs: SearchInputs, table: ProbeTable) -> int | None:
        T = inputs.T  # noqa: N806
        tau_abs = _threshold(inputs, tau)
        lo, hi = 0, T + 1  # lo: known alive (0 = virtual start), hi: known collapsed
        probes = 0
        seen: set[int] = set()
        while hi - lo > 1 and probes < max_probes:
            mid = (lo + hi) // 2
            u = snap(mid, _probed_from(table, mid))
            if u in seen or not (lo < u < hi):
                break
            seen.add(u)
            probes += 1
            s, k = table.value(u, k_max=small)
            if k and s / k <= tau_abs:
                hi = u
            else:
                lo = u
        if 1 <= hi <= T:
            table.value(hi, k_max=K)  # confirmation at full K
        return lo or None  # lo == 0: s_1 already collapsed, t* undefined

    return policy


def _binomial_pmf(s: int, k: int, p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.comb(k, s) * p**s * (1 - p) ** (k - s)


def judge_likelihood(dist: list[float] | None, T: int,  # noqa: N803
                     floor: float = JUDGE_LIKELIHOOD_FLOOR) -> list[float] | None:
    """Judge distribution as a likelihood over t* = 1..T: `(1 - floor) * p_t + floor / T`,
    so a confident judge never zeroes a candidate. None when `dist` is absent/mis-sized."""
    if dist is None or len(dist) != T or T <= 0:
        return None
    z = sum(dist)
    if z <= 0:
        return None
    return [(1 - floor) * p / z + floor / T for p in dist]


def guard_bayesect(prior: bayes_prior.LaplaceGuardPrior, K: int, *,  # noqa: N803
                   tau: float = 0.5, step: int = 4, max_probes: int = 8,
                   stop_delta: float = 0.05, p_dead: float | None = None,
                   judge_slot: str | None = None) -> Policy:
    """Laplace guard prior + sequential Beta stopping + entropy-minimising next probe.

    Posterior over t* in 1..T starts from `prior.pmf(.., g, T)` with `g` the first
    non-terminal guard firing. With `judge_slot` (e.g. `judge_open`, `judge_claude`) the
    judge's distribution from `inputs.judge_dists` multiplies that prior as a zero-rollout
    likelihood (`judge_likelihood`, floored mixture) before any probe; the multiplication
    is done here, so Arnav's `bayes_prior` stubs keep their signatures. Each probe asks
    `next_bayes_probe`; at the chosen turn, rollouts are consumed `step` at a time until
    `sequential_beta_stop` says the state is decisive (or K is spent). The binomial
    likelihood of `(s, k)` under candidate c uses `p = v_s1` for `t <= c` (alive) and
    `p = p_dead` (default `tau * v_s1 / 2`) for `t > c`. Returns the posterior mode.
    Raises `NotImplementedError` until Arnav implements `guardfork.stats.bayes_prior`.
    """
    if K <= 0 or step <= 0:
        raise ValueError("K and step must be positive")

    def policy(inputs: SearchInputs, table: ProbeTable) -> int | None:
        T = inputs.T  # noqa: N806
        tau_abs = _threshold(inputs, tau)
        dead = tau_abs / 2 if p_dead is None else p_dead
        alive = max(inputs.v_s1, tau_abs + 1e-6)
        g = localizer_turns(inputs.traj, inputs.firings)["all"]
        cands = list(range(1, T + 1))
        post = list(prior.pmf(cands, g, T))
        lik = judge_likelihood(inputs.judge_dist(judge_slot), T) if judge_slot else None
        if lik is not None:
            post = [p * q for p, q in zip(post, lik, strict=True)]
            z = sum(post)
            post = [p / z for p in post] if z > 0 else [1 / T] * T
        probed: set[int] = set()
        for _ in range(max_probes):
            t = bayes_prior.next_bayes_probe(post, probed, T)
            if t is None:
                break
            u = snap(t, _probed_from(table, t))
            if u in probed:
                probed.add(t)
                continue
            probed.update({t, u})
            s = k = 0
            for budget in range(step, K + 1, step):
                s, k = table.value(u, k_max=budget)
                if k < budget or bayes_prior.sequential_beta_stop(s, k, tau=tau_abs,
                                                                   delta=stop_delta):
                    break
            if k == 0:
                continue
            post = [p * _binomial_pmf(s, k, alive if u <= c else dead)
                    for c, p in zip(cands, post, strict=True)]
            z = sum(post)
            post = [p / z for p in post] if z > 0 else [1 / T] * T
        return cands[max(range(T), key=lambda i: (post[i], -i))]

    return policy


# ---------------------------------------------------------------------------- simulate


def simulate(policies: Mapping[str, Policy], table: ProbeTable, t_star: int | None,
             windows: tuple[int, ...] = WINDOWS, *, inputs: SearchInputs) -> dict[str, dict]:
    """Run every policy on a fresh copy of the table's budget; per policy report the pick,
    `hit[w] = |pick - t*| <= w` for each window (False when either is None) and the
    continuations used. Zero-rollout localizers report 0 continuations."""
    out: dict[str, dict] = {}
    for name, policy in policies.items():
        table.reset()
        pick = policy(inputs, table)
        hit = {w: (pick is not None and t_star is not None and abs(pick - t_star) <= w)
               for w in windows}
        out[name] = {"pick": pick, "hit": hit, "continuations": table.continuations_used}
    table.reset()
    return out


def default_policies(K: int = 8, *, prior: bayes_prior.LaplaceGuardPrior | None = None,  # noqa: N803
                     ) -> dict[str, Policy]:
    """The pre-registered localizer ladder (guard_bayesect rows only when a prior is given:
    plain, `_open_prior` with the `judge_open` distribution, `_claude_prior` with the
    `judge_claude` distribution). `claude_judge` and `guard_bayesect_claude_prior` are
    eval-only rows (`POLICY_FLAGS`)."""
    ladder: dict[str, Policy] = {
        "lvrl_style_error_marker": lvrl_style_error_marker,
        "random": random_turn,
        "last_turn": last_turn,
        "entropy_peak": entropy_peak,
        "guards_raw": guards_raw,
        "guards_recovery_aware": guards_recovery_aware,
        "open_judge": open_judge,
        "claude_judge": claude_judge,
        "bisection_fixed": bisection_fixed(K),
        "bisection_early_stop": bisection_early_stop(K),
        "elpo_bel": elpo_bel(K),
    }
    if prior is not None:
        ladder["guard_bayesect"] = guard_bayesect(prior, K)
        ladder["guard_bayesect_open_prior"] = guard_bayesect(prior, K,
                                                             judge_slot=OPEN_JUDGE_SLOT)
        ladder["guard_bayesect_claude_prior"] = guard_bayesect(prior, K,
                                                               judge_slot=CLAUDE_JUDGE_SLOT)
    return ladder


def eval_only_policies(policies: Mapping[str, Policy]) -> set[str]:
    return {name for name in policies if POLICY_FLAGS.get(name, {}).get("eval_only")}
