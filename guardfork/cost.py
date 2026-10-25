"""CostLedger: env-seconds, GPU-hours and tokens per phase, for P2 and the K3 re-check."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CostEntry:
    phase: str
    trial_dir: str
    env_seconds: float
    gpu_seconds: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class CostLedger:
    path: Path
    entries: list[CostEntry] = field(default_factory=list)

    def charge(self, entry: CostEntry) -> None:
        """Append to memory and to `path` (under runs/, gitignored)."""
        raise NotImplementedError("W1: cost.CostLedger.charge")

    def totals(self, phase: str | None = None) -> dict[str, float]:
        raise NotImplementedError("W1: cost.CostLedger.totals")

    def cost_per_usable_pair(self, phase: str, usable_pairs: int, gpu_usd_per_hour: float
                             ) -> float | None:
        raise NotImplementedError("W1: cost.CostLedger.cost_per_usable_pair")

    @classmethod
    def load(cls, path: Path) -> CostLedger:
        raise NotImplementedError("W1: cost.CostLedger.load")
