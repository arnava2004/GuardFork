"""Monte Carlo referee for one failed run, v2 protocol (PREREGISTRATION.md "Screening and
probing" and "Primary oracle").

1. **s1 screen.** `screen_k` continuations from s_1; a 0/`screen_k` run gets
   `confirm_k - screen_k` more, and 0/`confirm_k` is `hopeless` (no further probes, no t*).
   Every run with at least one success enters the pool, whether or not a guard fired.
2. **Candidate grid.** `K` continuations at every turn of `probe_plan.candidate_grid`: the
   thinned grid (every `grid_step` turns) plus {x-1, x, x+1} around the first firing of
   each guard and around every `extra_turns` pick (judge slots, entropy peak), clipped to
   forkable turns. `dense=True` probes every forkable turn (the 10-15 dense runs that bound
   snapping bias).
3. **Refinement.** The primary oracle (`oracle`, default
   `stats.bayes_prior.fit_changepoint_oracle`) is fit to the non-eval-only values; the two
   probed turns bracketing its t* get `refine_k` more continuations (K = 16 there), then the
   oracle is refit on all values. Until `bayes_prior` is implemented the oracle raises
   `NotImplementedError`: the result then has `refine_pending = True` and no primary t*,
   and a rerun after implementation reuses every finished probe job.

The oracle sees the probed values plus the run's own end state `{T + 1: (0, K)}` (a failed
run, reward 0). `t_star_v1` (adjacent-pair rule, `stats.oracle`) and `t_star_drop` are
sensitivity labels from the same values.

`launch(plan, K, job_name) -> job_dir` is injected so the loop is testable offline; the
real launcher runs one Harbor job with `n_attempts = K` from a ReplayPlan.

Every probed turn records why it was probed (`screen`, `grid`, `guard:<name>`, `refine`,
`dense`, plus every `extra_turns` source such as `judge_open` / `judge_claude`).
Continuations whose turn reasons are only `{"judge_claude"}` are tagged `eval_only` in
forks.json (AGENTS.md rule 3: they never enter pair export); their outcomes still count for
the oracle. Refinement is planned from non-eval-only values, so no `refine` probe depends
on a Claude pick.

`v_s1` pools the s_1 continuations with the base runs (exchangeable draws of V(s_1));
`v_s1_base` keeps the base-only Jeffreys value. A probe with no graded continuation
(Harbor or proxy failure) stops the referee: `failed_probes` lists it and callers must not
treat the result as an oracle label.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from guardfork.fork.harbor_replay import build_replay_plan, save_plan
from guardfork.fork.probe_plan import candidate_grid
from guardfork.fork.select import forkable_turns
from guardfork.guards.rules import GUARDS_VERSION, evaluate, first_nonterminal
from guardfork.stats import bayes_prior
from guardfork.stats.oracle import argmax_drop_turn, derailment_turn, jeffreys_v1
from guardfork.telemetry.to_trajectory import to_trajectory
from guardfork.telemetry.trials import TrialRecord, iter_trials
from guardfork.types import Trajectory

Launch = Callable[[Path, int, str], Path]
Oracle = Callable[..., Any]  # fit_changepoint_oracle(values, *, v_s1, tau, ...) -> result
EVAL_ONLY_SOURCES = frozenset({"judge_claude"})
PROTOCOL = "v2"


def is_eval_only(reasons: Iterable[str]) -> bool:
    """True when every reason for probing a turn is an eval-only source."""
    reasons = set(reasons)
    return bool(reasons) and reasons <= EVAL_ONLY_SOURCES


@dataclass
class RefereeResult:
    trial_id: str
    task_id: str
    T: int  # noqa: N815
    g: int | None
    K: int  # noqa: N815
    v_s1: float
    values: dict[int, tuple[int, int]] = field(default_factory=dict)
    t_star: int | None = None
    t_star_lo: int | None = None
    t_star_hi: int | None = None
    t_star_sensitivity: dict[float, int | None] = field(default_factory=dict)
    t_star_v1: int | None = None
    t_star_drop: int | None = None
    continuations: list[dict] = field(default_factory=list)
    env_seconds: float = 0.0
    skipped: str | None = None
    probe_reasons: dict[int, list[str]] = field(default_factory=dict)
    v_s1_base: float | None = None
    failed_probes: list[int] = field(default_factory=list)
    hopeless: bool = False
    refine_pending: bool = False
    dense: bool = False

    def to_json(self) -> dict:
        return {
            "trial_id": self.trial_id, "task_id": self.task_id, "T": self.T, "g": self.g,
            "K": self.K, "v_s1": self.v_s1, "v_s1_base": self.v_s1_base,
            "protocol": PROTOCOL, "dense": self.dense,
            "values": {str(t): list(v) for t, v in sorted(self.values.items())},
            "t_star": self.t_star, "t_star_lo": self.t_star_lo, "t_star_hi": self.t_star_hi,
            "t_star_sensitivity": {str(k): v for k, v in self.t_star_sensitivity.items()},
            "t_star_v1": self.t_star_v1, "t_star_drop": self.t_star_drop,
            "hopeless": self.hopeless, "refine_pending": self.refine_pending,
            "guards_version": GUARDS_VERSION, "env_seconds": self.env_seconds,
            "n_continuations": len(self.continuations), "skipped": self.skipped,
            "probe_reasons": {str(t): list(r) for t, r in sorted(self.probe_reasons.items())},
            "n_eval_only": sum(bool(c.get("eval_only")) for c in self.continuations),
            "failed_probes": list(self.failed_probes),
        }


def collect_job(job_dir: Path, turn: int) -> tuple[tuple[int, int], list[dict], float]:
    passes = graded = 0
    conts, secs = [], 0.0
    for k, rec in enumerate(iter_trials(job_dir, job_dir.name)):
        first = rec.turns[0].command if rec.turns else None
        conts.append({"turn": turn, "k": k, "reward": rec.reward, "first_action": first,
                      "trial_dir": str(rec.trial_dir), "job": job_dir.name})
        secs += rec.wall_seconds or 0.0
        if rec.reward is not None:
            graded += 1
            passes += int(rec.passed)
    return (passes, graded), conts, secs


def first_firing_per_guard(traj: Trajectory) -> dict[str, int]:
    """Guard name -> turn of its first firing (any turn, terminal included)."""
    out: dict[str, int] = {}
    for f in evaluate(traj):
        out.setdefault(f.guard.value, f.turn)
    return out


def bracket(t_star: int, probed: Iterable[int]) -> list[int]:
    """t* and the next probed turn after it (the two states the change-point separates)."""
    later = [u for u in probed if u > t_star]
    return [t_star, min(later)] if later else [t_star]


def run_referee(rec: TrialRecord, *, base_pass: tuple[int, int], K: int,  # noqa: N803
                launch: Launch, plans_dir: Path, grid_step: int = 3,
                extra_turns: Mapping[str, Iterable[int | None]] | None = None,
                screen_k: int = 8, confirm_k: int = 16, refine_k: int = 8,
                dense: bool = False, tau: float = 0.5,
                oracle: Oracle | None = None) -> RefereeResult:
    """Run the v2 protocol on one failed run (see the module docstring). `extra_turns` maps a
    source name to picked turns (None entries are ignored)."""
    oracle = oracle or bayes_prior.fit_changepoint_oracle
    traj = to_trajectory(rec)
    first = first_nonterminal(evaluate(traj), traj.T)
    trial_id = f"{rec.job}/{rec.trial_dir.name}"
    passes, n = base_pass
    v_base = jeffreys_v1(passes, n) if n else 0.5
    result = RefereeResult(trial_id, rec.task_id or "", traj.T,
                           first.turn if first else None, K, v_base, v_s1_base=v_base,
                           dense=dense)
    allowed = forkable_turns(traj)
    if 1 not in allowed:
        result.skipped = "s_1 not forkable"
        return result
    wanted: dict[int, list[str]] = {}

    def want(t: int, reason: str) -> None:
        if t in allowed and reason not in wanted.setdefault(t, []):
            wanted[t].append(reason)

    def tag(t: int) -> None:
        eval_only = is_eval_only(result.probe_reasons[t])
        for c in result.continuations:
            if c["turn"] == t:
                c["reasons"] = list(result.probe_reasons[t])
                c["eval_only"] = eval_only

    def probe(t: int, k: int, suffix: str, reason: str) -> bool:
        """Launch k continuations at s_t (added to any earlier ones); False if none graded."""
        reasons = result.probe_reasons.setdefault(t, [])
        for r in [*wanted.get(t, []), reason]:
            if r not in reasons:
                reasons.append(r)
        plan = build_replay_plan(rec, t)
        path = save_plan(plan, plans_dir / f"{rec.trial_dir.name}_t{t}.json")
        job_dir = launch(path, k, f"ref-{rec.trial_dir.name}-t{t}{suffix}")
        (s, g), conts, secs = collect_job(job_dir, t)
        ps, pg = result.values.get(t, (0, 0))
        result.values[t] = (ps + s, pg + g)
        result.continuations += conts
        result.env_seconds += secs
        tag(t)
        if g == 0:
            result.failed_probes.append(t)
            note = f"probe t={t}{suffix} produced no graded continuations"
            result.skipped = f"{result.skipped}; {note}" if result.skipped else note
            return False
        if t == 1:
            s1, n1 = result.values[1]
            result.v_s1 = jeffreys_v1(passes + s1, n + n1)
        return True

    # 1. s1 screen
    if not probe(1, screen_k, "", "screen"):
        return result
    if result.values[1][0] == 0:
        if not probe(1, confirm_k - screen_k, "-confirm", "screen"):
            return result
        if result.values[1][0] == 0:
            result.hopeless = True
            result.skipped = f"hopeless: 0/{result.values[1][1]} from s_1"
            return result

    # 2. candidate grid
    for name, t in first_firing_per_guard(traj).items():
        for u in (t - 1, t, t + 1):
            want(u, f"guard:{name}")
    for source, turns in (extra_turns or {}).items():
        for x in turns:
            if x is not None:
                for u in (x - 1, x, x + 1):
                    want(u, source)
    T = traj.T  # noqa: N806
    if dense:
        for t in allowed:
            want(t, "dense")
    else:
        for t in candidate_grid(T, step=grid_step):
            want(t, "grid")
    for t in sorted(wanted):
        if t == 1:
            result.probe_reasons[1] += [r for r in wanted[1] if r not in result.probe_reasons[1]]
            tag(1)
            continue
        if not probe(t, K, "", wanted[t][0]):
            return result

    # 3. refinement around the primary change-point
    terminal = {T + 1: (0, max(K, 1))}

    def fit(include_eval_only: bool) -> Any:
        vals = {t: v for t, v in result.values.items()
                if include_eval_only or not is_eval_only(result.probe_reasons[t])}
        return oracle({**vals, **terminal}, v_s1=result.v_s1, tau=tau)

    try:
        plan_fit = fit(include_eval_only=False)
    except NotImplementedError:
        result.refine_pending = True
    else:
        if plan_fit.t_star is not None:
            planning_turns = [t for t in result.values
                              if not is_eval_only(result.probe_reasons[t])]
            for t in bracket(plan_fit.t_star, planning_turns):
                if t in allowed and not probe(t, refine_k, "-refine", "refine"):
                    return result
        final = fit(include_eval_only=True)
        result.t_star, result.t_star_lo, result.t_star_hi = final.t_star, final.lo, final.hi
        result.t_star_sensitivity = dict(final.sensitivity)

    v_hat = {t: p / g for t, (p, g) in result.values.items() if g}
    result.t_star_v1 = derailment_turn(v_hat, reward=rec.reward or 0.0, v_s1=result.v_s1,
                                       K=K, T=T)
    result.t_star_drop = argmax_drop_turn(v_hat, reward=rec.reward or 0.0, T=T)
    return result


def write_results(result: RefereeResult, results_root: Path) -> Path:
    out = results_root / result.trial_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "oracle.json").write_text(json.dumps(result.to_json(), indent=2))
    (out / "forks.json").write_text(json.dumps(result.continuations, indent=2))
    return out
