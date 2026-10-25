"""Guard freeze and dev/eval separation.

`config/guards_freeze.toml` pins the sha256 of the guard sources for each released
version; `config/dev_set.toml` lists the jobs and tasks the guards were tuned on.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from guardfork import REPO_ROOT

GUARD_SOURCES = ("guardfork/guards/rules.py", "guardfork/guards/ast_sig.py")
FREEZE_PATH = REPO_ROOT / "config" / "guards_freeze.toml"
DEV_SET_PATH = REPO_ROOT / "config" / "dev_set.toml"


def guard_sources_sha256(root: Path = REPO_ROOT) -> str:
    digest = hashlib.sha256()
    for rel in GUARD_SOURCES:
        digest.update(rel.encode())
        digest.update((root / rel).read_bytes())
    return digest.hexdigest()


def frozen_versions(path: Path = FREEZE_PATH) -> dict[str, str]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {name: entry["sha256"] for name, entry in data.get("versions", {}).items()}


@dataclass(frozen=True)
class DevSet:
    """Design side of the scenario-level design/evaluation split."""

    jobs: frozenset[str]
    scenarios: frozenset[str]

    @property
    def tasks(self) -> frozenset[str]:
        return frozenset(f"{s}_{i}" for s in self.scenarios for i in (1, 2, 3))

    def excludes(self, job: str, task_id: str | None) -> bool:
        if job in self.jobs:
            return True
        if task_id is None:
            return False
        scenario, sep, _ = task_id.rpartition("_")
        return bool(sep) and scenario in self.scenarios


def load_dev_set(path: Path = DEV_SET_PATH) -> DevSet:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    scenarios = set(data.get("design_scenarios", []))
    for task in data.get("dev_tasks", []):  # legacy key
        scenarios.add(task.rpartition("_")[0])
    return DevSet(frozenset(data.get("dev_jobs", [])), frozenset(scenarios))
