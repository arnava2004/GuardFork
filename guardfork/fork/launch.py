"""Launch K fork continuations as Harbor trials and collect verifier rewards."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from guardfork.fork.harbor_replay import ReplayPlan
from guardfork.telemetry.trials import iter_trials

HarborRunner = Callable[[ReplayPlan, int, Path], Path]


@dataclass(frozen=True)
class ContinuationResult:
    plan_turn: int
    k: int
    trial_dir: Path
    reward: float | None
    status: str
    env_seconds: float | None
    first_action: str | None


def value_estimate(results: list[ContinuationResult]) -> tuple[int, int]:
    """(passes, graded) for V_hat(s_t); errors/ungraded continuations are excluded and
    counted separately by the caller."""
    graded = [r for r in results if r.reward is not None]
    return sum(r.reward >= 1.0 for r in graded), len(graded)


def _collect(job_dir: Path, plan: ReplayPlan, k: int) -> ContinuationResult:
    records = list(iter_trials(job_dir, job_dir.name))
    if not records:
        return ContinuationResult(plan.fork_turn, k, job_dir, None, "missing", None, None)
    rec = records[0]
    first = rec.turns[0].command if rec.turns else None
    status = "ok" if rec.reward is not None else "error"
    return ContinuationResult(plan.fork_turn, k, rec.trial_dir, rec.reward, status,
                              rec.wall_seconds, first)


def launch_continuations(plan: ReplayPlan, K: int, *, job_name: str, output_dir: Path,  # noqa: N803
                         concurrency: int, eval_seed: int,
                         runner: HarborRunner | None = None) -> list[ContinuationResult]:
    """Run K isolated Harbor trials from `plan`; each graded by the official verifier.

    Writes results under `output_dir/job_name` (forks/ — gitignored). Pass `runner` in tests;
    the default runner is a Harbor subprocess and requires the AppWorld images.
    """
    if K <= 0:
        raise ValueError("K must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / f"{job_name}_plan.json"
    plan_path.write_text(json.dumps({**plan.to_json(), "K": K, "eval_seed": eval_seed,
                                     "concurrency": concurrency}, indent=2))
    results = []
    for k in range(K):
        trial_out = output_dir / job_name / f"k{k}"
        trial_out.mkdir(parents=True, exist_ok=True)
        if runner is None:
            raise RuntimeError(
                "launch_continuations needs a Harbor runner (or inject runner= in tests); "
                "do not launch forks until Weekend 0 gates pass"
            )
        produced = runner(plan, k, trial_out)
        results.append(_collect(produced, plan, k))
    return results
