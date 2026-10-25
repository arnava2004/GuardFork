from __future__ import annotations

from pathlib import Path

import pytest

from guardfork.splits import (
    FinalSplitLocked,
    group_by_scenario,
    load_splits,
    scenario_of,
    split_train147,
)


def _ids(n_scenarios: int, offset: int) -> list[str]:
    return [f"{i + offset:07x}_{k}" for i in range(n_scenarios) for k in (1, 2, 3)]


def test_scenario_of():
    assert scenario_of("50e1ac9_2") == "50e1ac9"
    with pytest.raises(ValueError):
        scenario_of("nounderscore")


def test_split_train147_is_scenario_grouped_and_deterministic():
    ids = _ids(49, 0)
    train, dev = split_train147(ids)
    assert (len(train), len(dev)) == (120, 27)
    assert not {scenario_of(t) for t in train} & {scenario_of(t) for t in dev}
    assert split_train147(list(reversed(ids))) == (train, dev)
    assert all(len(v) == 3 for v in group_by_scenario(dev).values())


def test_test_normal_locked(tmp_path: Path):
    tr, te = tmp_path / "train.txt", tmp_path / "test.txt"
    tr.write_text("\n".join(_ids(49, 0)))
    te.write_text("\n".join(_ids(56, 1000)))
    splits = load_splits(tr, te)
    with pytest.raises(FinalSplitLocked):
        splits.test_normal()
    assert len(splits.test_normal(allow_final=True)) == 168


def test_overlap_rejected(tmp_path: Path):
    tr, te = tmp_path / "train.txt", tmp_path / "test.txt"
    tr.write_text("\n".join(_ids(49, 0)))
    te.write_text("\n".join(_ids(56, 10)))
    with pytest.raises(ValueError, match="shared"):
        load_splits(tr, te)
