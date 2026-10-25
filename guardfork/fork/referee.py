"""Monte Carlo referee for one failed run: probe states around the first guard with K
continuations each, bisect to the turn where success odds collapse, and record t*.

`launch(plan, K, job_name) -> job_dir` is injected so the loop is testable offline; the
real launcher runs one Harbor job with `n_attempts = K` from a ReplayPlan.
"""

from __future__ import annotations

import json
from collections.abc import Callable
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

    def to_json(self) -> dict:
        return {
            "trial_id": self.trial_id, "task_id": self.task_id, "T": self.T, "g": self.g,
            "K": self.K, "v_s1": self.v_s1,
            "values": {str(t): list(v) for t, v in sorted(self.values.items())},
            "t_star": self.t_star, "t_star_drop": self.t_star_drop,
            "guards_version": GUARDS_VERSION, "env_seconds": self.env_seconds,
            "n_continuations": len(self.continuations), "skipped": self.skipped,
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
                launch: Launch, plans_dir: Path, max_probes: int = 10) -> RefereeResult:
    traj = to_trajectory(rec)
    first = first_nonterminal(evaluate(traj), traj.T)
    trial_id = f"{rec.job}/{rec.trial_dir.name}"
    passes, n = base_pass
    result = RefereeResult(trial_id, rec.task_id or "", traj.T,
                           first.turn if first else None, K,
                           jeffreys_v1(passes, n) if n else 0.5)
    if first is None:
        result.skipped = "no non-terminal guard"
        return result
    allowed = set(forkable_turns(traj))
    queue = [t for t in initial_probes(first.turn, traj.T) if t in allowed]

    def probe(t: int) -> None:
        plan = build_replay_plan(rec, t)
        path = save_plan(plan, plans_dir / f"{rec.trial_dir.name}_t{t}.json")
        job_dir = launch(path, K, f"ref-{rec.trial_dir.name}-t{t}")
        value, conts, secs = collect_job(job_dir, t)
        result.values[t] = value
        result.continuations += conts
        result.env_seconds += secs

    for t in queue:
        if len(result.values) >= max_probes:
            break
        probe(t)
    while len(result.values) < max_probes:
        v_hat = {t: p / g for t, (p, g) in result.values.items() if g}
        nxt = next_probe(v_hat, result.v_s1, K, traj.T)
        if nxt is None or nxt in result.values:
            break
        if nxt not in allowed:
            result.skipped = f"bisection needs non-forkable turn {nxt}"
            break
        try:
            probe(nxt)
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
