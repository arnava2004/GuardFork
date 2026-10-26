from guardfork.train.prefix_drift import (
    DriftRun,
    TurnLogProbs,
    compare_checkpoints,
    group_of,
    run_group_means,
    summarize,
)


def _run(trial, scenario, lo, hi, deltas, n=1):
    turns = [TurnLogProbs(t, -10.0, -10.0 + d, n) for t, d in enumerate(deltas, start=1)]
    return DriftRun(trial, scenario, lo, hi, turns)


def test_grouping():
    assert [group_of(t, 4, 6) for t in (1, 4, 6, 7)] == ["pre", "at", "at", "post"]


def test_group_means_and_per_token():
    run = _run("a", "s1", 3, 3, [0.2, 0.4, -1.0, -0.5], n=2)
    means = run_group_means(run)
    assert means["pre"] == 0.30000000000000004 or abs(means["pre"] - 0.3) < 1e-9
    assert means["at"] == -1.0 and means["post"] == -0.5
    assert abs(run_group_means(run, per_token=True)["at"] + 0.5) < 1e-9


def test_prefix_protection_gap_positive_when_prefix_spared():
    # Soft rule: pre-mistake turns barely move, the at-turn is pushed down.
    runs = [_run(f"r{i}", f"sc{i % 3}", 3, 3, [0.0, 0.05, -1.2, -0.8]) for i in range(6)]
    s = summarize(runs, n_boot=300)
    assert s.n_runs == 6 and s.n_turns == {"pre": 12, "at": 6, "post": 6}
    assert s.gap_pre_minus_at is not None and s.gap_pre_minus_at > 1.0
    assert s.gap_ci is not None and s.gap_ci[0] > 0


def test_gap_near_zero_when_uniform_penalty():
    runs = [_run(f"r{i}", f"sc{i % 2}", 2, 2, [-0.7, -0.7, -0.7]) for i in range(4)]
    s = summarize(runs, n_boot=200)
    assert abs(s.gap_pre_minus_at) < 1e-9 and s.gap_pre_minus_post == 0.0


def test_runs_without_pre_turns_are_kept_but_not_paired():
    runs = [_run("a", "s", 1, 1, [-1.0, 0.1]), _run("b", "s", 2, 2, [0.3, -1.0])]
    s = summarize(runs, n_boot=100)
    assert s.n_runs == 2 and len([r for r in s.per_run if "gap_pre_minus_at" in r]) == 1


def test_compare_table():
    runs = [_run("a", "s", 2, 2, [0.1, -1.0, -0.2])]
    md = compare_checkpoints({"uniform": summarize(runs, n_boot=50)})
    assert "| uniform | 1 |" in md and "gap pre-at" in md
