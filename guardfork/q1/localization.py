"""Q1 study assembly (Weekends 3–4): eligibility, per-run rows, and the report table.

Inference (clustered bootstrap, analytic random) lives in `guardfork.stats` (Arnav's); this
module only selects runs and builds rows.
"""

from __future__ import annotations

from dataclasses import dataclass

from guardfork.telemetry.trials import TrialRecord


@dataclass(frozen=True)
class Q1Row:
    trial_id: str
    scenario: str
    T: int  # noqa: N815
    g: int | None          # first non-terminal guard turn (None: no non-terminal firing)
    g_terminal: bool       # a guard fired only at g == T
    t_star: int | None     # None: undefined (excluded, rate reported)
    t_star_drop: int | None  # argmax-drop sensitivity


def eligible_failed_runs(records: list[TrialRecord], task_pass: dict[str, tuple[int, int]],
                         band: tuple[float, float] = (0.3, 0.8), min_samples: int = 8
                         ) -> list[TrialRecord]:
    """Failed runs on tasks with p_hat in `band` over >= `min_samples` base rollouts."""
    raise NotImplementedError("W3: q1.localization.eligible_failed_runs")


def build_rows(records: list[TrialRecord], oracle_results: dict[str, dict]) -> list[Q1Row]:
    raise NotImplementedError("W3: q1.localization.build_rows")


def report_table(rows: list[Q1Row], windows: tuple[int, ...] = (0, 1, 2, 3)) -> str:
    """Markdown: non-terminal headline, terminal column, undefined-t* rate, Hit(w) per w."""
    raise NotImplementedError("W3: q1.localization.report_table")
