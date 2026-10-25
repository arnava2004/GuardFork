"""Per-trial reproducibility manifest (AGENTS.md rule 5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Manifest:
    commit_sha: str
    config_hash: str
    harness_hash: str
    model_id: str
    backend: str
    train_seed: int | None
    eval_seed: int
    env_seconds: float | None
    container_limits: dict[str, str]
    pythonhashseed: str = "0"


def harness_hash(harness_zero_dir: Path, compose_file: Path) -> str:
    """sha256 over the Harness-Zero commit, the AppWorld agent/env sources, the compose
    file, and PYTHONHASHSEED."""
    raise NotImplementedError("W1: manifest.harness_hash")


def write_manifest(trial_dir: Path, manifest: Manifest) -> Path:
    """Write `guardfork_manifest.json` beside the trial (never into the verifier dir)."""
    raise NotImplementedError("W1: manifest.write_manifest")
