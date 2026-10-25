"""Offline preference-pair builders for RFT, T-DPO, G-REW, R-FORK-DPO, GF-DPO."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from guardfork.fork.select import guard_fork_turn, random_fork_turn
from guardfork.guards.ast_sig import api_signature
from guardfork.guards.rules import Guard, evaluate
from guardfork.types import N_DEMO_ASSISTANT_TURNS, Trajectory, TrajectoryStatus


@dataclass(frozen=True)
class Action:
    content: str
    reasoning: str
    command: str | None
    reward: float | None = None
    turn: int | None = None


@dataclass(frozen=True)
class Pair:
    task_id: str
    arm: str
    chosen: Action
    rejected: Action
    think_only_diff: bool
    signature_collision: bool
    masked_assistant_turns: tuple[int, ...] = tuple(range(N_DEMO_ASSISTANT_TURNS))


def _action(traj: Trajectory, turn: int | None = None) -> Action:
    if turn is None:
        t = traj.turns[-1] if traj.turns else None
    else:
        t = next((x for x in traj.turns if x.turn == turn), None)
    if t is None:
        return Action("", "", None, traj.reward, turn)
    return Action(t.content, t.reasoning, t.command, traj.reward, t.turn)


def _flags(chosen: Action, rejected: Action) -> tuple[bool, bool]:
    sig_c, sig_r = api_signature(chosen.command), api_signature(rejected.command)
    collision = bool(sig_c) and sig_c == sig_r
    think_only = (chosen.command == rejected.command
                  and (chosen.reasoning or "") != (rejected.reasoning or ""))
    return think_only, collision


def rft_examples(trajectories: Sequence[Trajectory]) -> list[Trajectory]:
    """Successful bare-h trajectories (demo turns masked at train time)."""
    return [t for t in trajectories if t.status is TrajectoryStatus.PASSED]


def tdpo_pairs(trajectories: Sequence[Trajectory]) -> list[Pair]:
    """Whole-trajectory DPO: one success vs one failure per task."""
    by_task: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        by_task.setdefault(traj.task_id, []).append(traj)
    pairs = []
    for task_id, group in by_task.items():
        wins = [t for t in group if t.status is TrajectoryStatus.PASSED]
        losses = [t for t in group if t.status is not TrajectoryStatus.PASSED]
        if not wins or not losses:
            continue
        chosen, rejected = _action(wins[0]), _action(losses[0])
        think, coll = _flags(chosen, rejected)
        pairs.append(Pair(task_id, "T-DPO", chosen, rejected, think, coll))
    return pairs


def _n_nonterminal_guards(traj: Trajectory) -> int:
    firings = evaluate(traj)
    return sum(1 for f in firings if f.turn < traj.T)


def grew_pairs(trajectories: Sequence[Trajectory], n_gf_pairs: int) -> list[Pair]:
    """T-DPO plus failure-vs-failure ranked by outcome - 0.1 * #nonterminal_guards,
    truncated to `n_gf_pairs` extra pairs so the total extra count matches GF-DPO."""
    pairs = tdpo_pairs(trajectories)
    by_task: dict[str, list[Trajectory]] = {}
    for traj in trajectories:
        if traj.status is not TrajectoryStatus.PASSED:
            by_task.setdefault(traj.task_id, []).append(traj)
    extra: list[Pair] = []
    for task_id, group in by_task.items():
        if len(group) < 2:
            continue
        ranked = sorted(group, key=lambda t: (t.reward or 0.0) - 0.1 * _n_nonterminal_guards(t),
                        reverse=True)
        chosen, rejected = _action(ranked[0]), _action(ranked[-1])
        think, coll = _flags(chosen, rejected)
        extra.append(Pair(task_id, "G-REW", chosen, rejected, think, coll))
    extra = extra[:n_gf_pairs]
    tagged = [Pair(p.task_id, "G-REW", p.chosen, p.rejected, p.think_only_diff,
                   p.signature_collision) for p in pairs]
    return tagged + extra


def gf_dpo_pairs(failures: Sequence[Trajectory],
                 continuations: dict[str, list[tuple[Action, float]]],
                 exclude: frozenset[Guard] = frozenset()) -> list[Pair]:
    """Fork at first non-terminal guard; keep AST-distinct win vs loss first actions."""
    pairs = []
    for traj in failures:
        firings = evaluate(traj)
        g = guard_fork_turn(traj, firings, exclude)
        if g is None:
            continue
        original = _action(traj, g)
        cont = continuations.get(traj.task_id, [])
        wins = [a for a, r in cont if r >= 1.0]
        losses = [a for a, r in cont if r < 1.0]
        if not wins:
            continue
        rejected = original if (original.reward or 0) < 1.0 else (losses[0] if losses else None)
        if rejected is None:
            continue
        for chosen in wins:
            think, coll = _flags(chosen, rejected)
            if coll:
                continue
            pairs.append(Pair(traj.task_id, "GF-DPO", chosen, rejected, think, coll))
            break
    return pairs


def rfork_dpo_pairs(failures: Sequence[Trajectory],
                    continuations: dict[str, list[tuple[Action, float]]],
                    rng: random.Random) -> list[Pair]:
    """Random forkable turn on the same failed trajectory set as GF-DPO."""
    pairs = []
    for traj in failures:
        t = random_fork_turn(traj, rng)
        if t is None:
            continue
        original = _action(traj, t)
        cont = continuations.get(traj.task_id, [])
        wins = [a for a, r in cont if r >= 1.0]
        if not wins:
            continue
        chosen = wins[0]
        think, coll = _flags(chosen, original)
        if coll:
            continue
        pairs.append(Pair(traj.task_id, "R-FORK-DPO", chosen, original, think, coll))
    return pairs


def gf_eligible_failures(trajectories: Sequence[Trajectory],
                         exclude: frozenset[Guard] = frozenset()) -> list[Trajectory]:
    out = []
    for traj in trajectories:
        if traj.status is TrajectoryStatus.PASSED:
            continue
        firings = evaluate(traj)
        if guard_fork_turn(traj, firings, exclude) is not None:
            out.append(traj)
    return out
