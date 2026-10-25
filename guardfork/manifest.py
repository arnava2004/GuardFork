"""Per-trial reproducibility manifest (AGENTS.md rule 5)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

PYTHONHASHSEED = "0"
_SOURCE_REL = (
    "envs/appworld/agent.py",
    "envs/appworld/harbor_environment.py",
    "envs/appworld/common_context.py",
    "src/harness_zero/student.py",
    "src/harness_zero/harness.py",
)


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
    pythonhashseed: str = PYTHONHASHSEED


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def harness_hash(harness_zero_dir: Path, compose_file: Path) -> str:
    """sha256 over the Harness-Zero commit, the AppWorld agent/env sources, the compose
    file, and PYTHONHASHSEED."""
    digest = hashlib.sha256()
    digest.update(_git_sha(harness_zero_dir).encode())
    digest.update(f"PYTHONHASHSEED={PYTHONHASHSEED}".encode())
    if compose_file.is_file():
        digest.update(compose_file.read_bytes())
    for rel in _SOURCE_REL:
        path = harness_zero_dir / rel
        if path.is_file():
            digest.update(rel.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def write_manifest(trial_dir: Path, manifest: Manifest) -> Path:
    """Write `guardfork_manifest.json` beside the trial (never into the verifier dir)."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    path = trial_dir / "guardfork_manifest.json"
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    return path
