"""Kill-criteria config (`config/kill.toml`)."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guardfork import REPO_ROOT

DEFAULT_KILL_PATH = REPO_ROOT / "config" / "kill.toml"


@dataclass(frozen=True)
class Weekend0Thresholds:
    k0_sgc_range: tuple[float, float]
    k1_min_pass_band_fraction: float
    k1_pass_band: tuple[float, float]
    k1_min_samples_per_task: int
    k2_max_parse_failure_rate: float
    k3_max_cost_usd: float
    k4_max_prompt_tokens: int


@dataclass(frozen=True)
class CostInputs:
    gpu_usd_per_hour: float
    episodes_per_gpu_concurrently: int
    planned_q1_episodes: int
    planned_q2_episodes: int


@dataclass(frozen=True)
class KillConfig:
    weekend_0: Weekend0Thresholds
    weekend_1_pilot: dict[str, float]
    cost: CostInputs
    sha256: str


def load_kill_config(path: Path = DEFAULT_KILL_PATH) -> KillConfig:
    raw_bytes = path.read_bytes()
    raw: dict[str, Any] = tomllib.loads(raw_bytes.decode("utf-8"))
    w0 = raw["weekend_0"]
    lo, hi = w0["K0_9b_b0_test_normal_sgc_range"]
    band_lo, band_hi = w0["K1_pass_band"]
    return KillConfig(
        weekend_0=Weekend0Thresholds(
            k0_sgc_range=(float(lo), float(hi)),
            k1_min_pass_band_fraction=float(w0["K1_min_pass_band_fraction"]),
            k1_pass_band=(float(band_lo), float(band_hi)),
            k1_min_samples_per_task=int(w0["K1_min_samples_per_task"]),
            k2_max_parse_failure_rate=float(w0["K2_max_parse_failure_rate_per_turn"]),
            k3_max_cost_usd=float(w0["K3_max_projected_q1_q2_cost_usd"]),
            k4_max_prompt_tokens=int(w0["K4_max_mac_prompt_tokens_at_nonterminal_guard"]),
        ),
        weekend_1_pilot={k: float(v) for k, v in raw["weekend_1_pilot"].items()},
        cost=CostInputs(**raw["cost"]),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
