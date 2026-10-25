"""Scenario-grouped AppWorld splits with a loader-level lock on test_normal.

Task ids look like ``<scenario>_<n>`` with three tasks per scenario. All tasks of a
scenario land in the same split so scenario-clustered inference stays valid.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SPLIT_SALT = "guardfork-dev-heldout-v1"
N_DEV_HELDOUT_SCENARIOS = 9
EXPECTED_TRAIN147 = (147, 49)
EXPECTED_TEST168 = (168, 56)


class FinalSplitLocked(PermissionError):
    """Raised when test_normal is requested without allow_final=True."""


def scenario_of(task_id: str) -> str:
    scenario, sep, idx = task_id.rpartition("_")
    if not sep or not scenario or not idx.isdigit():
        raise ValueError(f"not an AppWorld task id: {task_id!r}")
    return scenario


def group_by_scenario(task_ids: Iterable[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for tid in task_ids:
        groups.setdefault(scenario_of(tid), []).append(tid)
    return {s: sorted(ts) for s, ts in sorted(groups.items())}


def read_task_ids(path: Path) -> list[str]:
    ids = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task ids in {path}")
    return ids


def _rank(scenario: str) -> str:
    return hashlib.sha256(f"{SPLIT_SALT}:{scenario}".encode()).hexdigest()


def split_train147(task_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """Deterministically split train147 into (train, dev_heldout) by scenario."""
    groups = group_by_scenario(task_ids)
    ranked = sorted(groups, key=_rank)
    dev = set(ranked[:N_DEV_HELDOUT_SCENARIOS])
    train = [t for s, ts in groups.items() if s not in dev for t in ts]
    heldout = [t for s, ts in groups.items() if s in dev for t in ts]
    return train, heldout


@dataclass(frozen=True)
class Splits:
    train: list[str]
    dev_heldout: list[str]
    _test_normal: list[str]

    def test_normal(self, *, allow_final: bool = False) -> list[str]:
        if not allow_final:
            raise FinalSplitLocked("test_normal is locked; pass allow_final=True (--final)")
        return list(self._test_normal)


def _check_shape(name: str, ids: list[str], expected: tuple[int, int], strict: bool) -> None:
    got = (len(ids), len(group_by_scenario(ids)))
    if strict and got != expected:
        raise ValueError(f"{name}: expected (tasks, scenarios)={expected}, got {got}")


def load_splits(train147_path: Path, test168_path: Path, *, strict: bool = True) -> Splits:
    train147 = read_task_ids(train147_path)
    test168 = read_task_ids(test168_path)
    _check_shape("train147", train147, EXPECTED_TRAIN147, strict)
    _check_shape("test168", test168, EXPECTED_TEST168, strict)
    overlap = {scenario_of(t) for t in train147} & {scenario_of(t) for t in test168}
    if overlap:
        raise ValueError(f"scenarios shared between train147 and test168: {sorted(overlap)[:5]}")
    train, dev = split_train147(train147)
    return Splits(train=train, dev_heldout=dev, _test_normal=test168)
