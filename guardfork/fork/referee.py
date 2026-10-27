"""Monte Carlo referee for one failed run: probe states around the first guard with K
continuations each, bisect to the turn where success odds collapse, and record t*.

`launch(plan, K, job_name) -> job_dir` is injected so the loop is testable offline; the
real launcher runs one Harbor job with `n_attempts = K` from a ReplayPlan.

Every probed turn records why it was probed (`reasons`: `initial`, `guard`, `bisection`,
plus any named `extra_turns` source such as `judge_open` / `judge_claude`). Continuations
whose probe reasons are only `{"judge_claude"}` are tagged `eval_only: true` in forks.json
(AGENTS.md rule 3: launched only because of the proprietary eval-only judge, so they never
enter pair export); their outcomes still count for the Q1 oracle value table. Bisection
plans its next probe only from non-eval-only values, so no `bisection` probe depends on a
Claude pick; if bisection asks for a turn that was probed only for `judge_claude`, that turn
gains the `bisection` reason and loses its `eval_only` tag (it would have been launched
anyway).

`v_s1` starts from the base runs' Jeffreys estimate and, once s_1 is probed, pools the s_1
continuations with the base runs (they are exchangeable draws of V(s_1)); `v_s1_base`
keeps the base-only value. All mandatory probes (initial window, guard, `extra_turns`
windows) always run; `max_probes` bounds the total only for bisection. A probe with no
graded continuation (Harbor or proxy failure) stops the referee: `failed_probes` lists it,
`t_star` stays None, and callers must not treat the result as an oracle label.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from guardfork.fork.harbor_replay import NotForkable, build_replay_plan, save_plan
from guardfork.fork.probe_plan import initial_probes, next_probe
from guardfork.fork.select import forkable_turns
from guardfork.guards.rules import GUARDS_VERSION, evaluate, first_nonterminal
from guardfork.stats.oracle import argmax_drop_turn, derailment_turn, jeffreys_v1
from guardfork.telemetry.to_trajectory import to_trajectory
from guardfork.telemetry.trials import TrialRecord, iter_trials

Launch = Callable[[Path, int, str], Path]
EVAL_ONLY_SOURCES = frozenset({"judge_claude"})


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
    t_star_drop: int | None = None
    continuations: list[dict] = field(default_factory=list)
    env_seconds: float = 0.0
    skipped: str | None = None
    probe_reasons: dict[int, list[str]] = field(default_factory=dict)
    v_s1_base: float | None = None
    failed_probes: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "trial_id": self.trial_id, "task_id": self.task_id, "T": self.T, "g": self.g,
            "K": self.K, "v_s1": self.v_s1, "v_s1_base": self.v_s1_base,
            "values": {str(t): list(v) for t, v in sorted(self.values.items())},
            "t_star": self.t_star, "t_star_drop": self.t_star_drop,
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
                      "trial_dir": str(rec.trial_dir)})
        secs += rec.wall_seconds or 0.0
        if rec.reward is not None:
            graded += 1
            passes += int(rec.passed)
    return (passes, graded), conts, secs


def run_referee(rec: TrialRecord, *, base_pass: tuple[int, int], K: int,  # noqa: N803
                launch: Launch, plans_dir: Path, max_probes: int = 10,
                extra_turns: Mapping[str, Iterable[int]] | None = None) -> RefereeResult:
    """`extra_turns` maps a source name (e.g. `judge_open`, `judge_claude`) to turns whose
    {x-1, x, x+1} windows are mandatory probes after the initial guard window and before
    bisection (clipped to [1, T] and to forkable turns). Probe reasons accumulate per turn;
    see the module docstring for `eval_only`, `v_s1` pooling and `failed_probes`."""
    traj = to_trajectory(rec)
    first = first_nonterminal(evaluate(traj), traj.T)
    trial_id = f"{rec.job}/{rec.trial_dir.name}"
    passes, n = base_pass
    v_base = jeffreys_v1(passes, n) if n else 0.5
    result = RefereeResult(trial_id, rec.task_id or "", traj.T,
                           first.turn if first else None, K, v_base, v_s1_base=v_base)
    if first is None:
        result.skipped = "no non-terminal guard"
        return result
    allowed = set(forkable_turns(traj))
    wanted: dict[int, list[str]] = {}

    def want(t: int, reason: str) -> None:
        if 1 <= t <= traj.T and t in allowed and reason not in wanted.setdefault(t, []):
            wanted[t].append(reason)

    for t in initial_probes(first.turn, traj.T):
        want(t, "initial")
    want(first.turn, "guard")
    for source, turns in (extra_turns or {}).items():
        for x in turns:
            if x is None:
                continue
            for u in (x - 1, x, x + 1):
                want(u, source)
    mandatory = [t for t in initial_probes(first.turn, traj.T) if t in allowed]
    mandatory += [t for t in sorted(wanted) if t not in mandatory]

    def tag(t: int) -> None:
        reasons = result.probe_reasons[t]
        eval_only = is_eval_only(reasons)
        for c in result.continuations:
            if c["turn"] == t:
                c["reasons"] = list(reasons)
                c["eval_only"] = eval_only

    def probe(t: int, reason: str | None = None) -> bool:
        """Launch K continuations at s_t; False when none was graded."""
        reasons = list(wanted.get(t, []))
        if reason and reason not in reasons:
            reasons.append(reason)
        result.probe_reasons[t] = reasons
        plan = build_replay_plan(rec, t)
        path = save_plan(plan, plans_dir / f"{rec.trial_dir.name}_t{t}.json")
        job_dir = launch(path, K, f"ref-{rec.trial_dir.name}-t{t}")
        value, conts, secs = collect_job(job_dir, t)
        result.values[t] = value
        result.continuations += conts
        result.env_seconds += secs
        tag(t)
        if value[1] == 0:
            result.failed_probes.append(t)
            note = f"probe t={t} produced no graded continuations"
            result.skipped = f"{result.skipped}; {note}" if result.skipped else note
            return False
        if t == 1:
            result.v_s1 = jeffreys_v1(passes + value[0], n + value[1])
        return True

    def planning_values() -> dict[int, float]:
        return {t: p / g for t, (p, g) in result.values.items()
                if g and not is_eval_only(result.probe_reasons[t])}

    for t in mandatory:
        if not probe(t):
            return result
    while len(result.values) < max_probes:
        nxt = next_probe(planning_values(), result.v_s1, K, traj.T)
        if nxt is None:
            break
        if nxt in result.values:
            if not is_eval_only(result.probe_reasons[nxt]):
                break
            result.probe_reasons[nxt].append("bisection")
            tag(nxt)
            continue
        if nxt not in allowed:
            result.skipped = f"bisection needs non-forkable turn {nxt}"
            break
        try:
            if not probe(nxt, "bisection"):
                return result
        except NotForkable:
            break
    v_hat = {t: p / g for t, (p, g) in result.values.items() if g}
    result.t_star = derailment_turn(v_hat, reward=rec.reward or 0.0, v_s1=result.v_s1,
                                    K=K, T=traj.T)
    result.t_star_drop = argmax_drop_turn(v_hat, reward=rec.reward or 0.0, T=traj.T)
    return result


def write_results(result: RefereeResult, results_root: Path) -> Path:
    out = results_root / result.trial_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "oracle.json").write_text(json.dumps(result.to_json(), indent=2))
    (out / "forks.json").write_text(json.dumps(result.continuations, indent=2))
    return out
