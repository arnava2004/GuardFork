"""Launch K fork continuations as Harbor trials and collect verifier rewards (Weekends 1–2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from guardfork.fork.harbor_replay import ReplayPlan


@dataclass(frozen=True)
class ContinuationResult:
    plan_turn: int
    k: int
    trial_dir: Path
    reward: float | None
    status: str
    env_seconds: float | None
    first_action: str | None


def launch_continuations(plan: ReplayPlan, K: int, *, job_name: str, output_dir: Path,  # noqa: N803
                         concurrency: int, eval_seed: int) -> list[ContinuationResult]:
    """Run K isolated Harbor trials from `plan`; each graded by the official verifier.

    Writes results under `output_dir/job_name` (forks/ — gitignored) and charges every trial
    to the CostLedger.
    """
    raise NotImplementedError("W1: fork.launch.launch_continuations")


def value_estimate(results: list[ContinuationResult]) -> tuple[int, int]:
    """(passes, graded) for V_hat(s_t); errors/ungraded continuations are excluded and
    counted separately by the caller."""
    raise NotImplementedError("W1: fork.launch.value_estimate")
