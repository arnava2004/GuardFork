"""CostLedger: env-seconds, GPU-hours and tokens per phase, for P2 and the K3 re-check."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(entry)) + "\n")

    def totals(self, phase: str | None = None) -> dict[str, float]:
        rows = [e for e in self.entries if phase is None or e.phase == phase]
        return {
            "env_seconds": sum(e.env_seconds for e in rows),
            "gpu_seconds": sum(e.gpu_seconds for e in rows),
            "prompt_tokens": float(sum(e.prompt_tokens for e in rows)),
            "completion_tokens": float(sum(e.completion_tokens for e in rows)),
            "n": float(len(rows)),
        }

    def cost_per_usable_pair(self, phase: str, usable_pairs: int, gpu_usd_per_hour: float
                             ) -> float | None:
        if usable_pairs <= 0:
            return None
        tot = self.totals(phase)
        usd = tot["gpu_seconds"] / 3600.0 * gpu_usd_per_hour
        return usd / usable_pairs

    @classmethod
    def load(cls, path: Path) -> CostLedger:
        ledger = cls(path)
        if not path.is_file():
            return ledger
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ledger.entries.append(CostEntry(**json.loads(line)))
        return ledger
