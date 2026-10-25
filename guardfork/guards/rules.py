"""Offline harness guards G1–G5, evaluated on live-turn prefixes only (no lookahead)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from guardfork.guards.ast_sig import api_signature, is_read, is_write
from guardfork.types import N_DEMO_ASSISTANT_TURNS, Trajectory, Turn

CODE_FENCE = "```"
APPWORLD_EXEC = "appworld exec"
TOOL_MARKUP = ("<tool_call>", "<function=")
HTTP_ERROR = re.compile(r"(?:status code is|HTTP)\s*(401|403|404|422)\b", re.IGNORECASE)
ENTITY_ID = re.compile(
    r"""(?:['"][0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}['"]"""
    r"""|[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,})""",
    re.IGNORECASE,
)
DESTRUCTIVE = re.compile(
    r"\b(?:delete_|remove_|drop_|destroy_|unlink)\w*\s*\(", re.IGNORECASE
)


class Guard(StrEnum):
    G1_INVALID_ACTION = "G1"
    G2_LOOP = "G2"
    G3_FAILED_PRECONDITION = "G3"
    G4_CONSTRAINT_VIOLATION = "G4"
    G5_UNVERIFIED_COMPLETION = "G5"


@dataclass(frozen=True)
class GuardFiring:
    guard: Guard
    turn: int
    evidence: str


def _live(turn: Turn) -> bool:
    return turn.assistant_index >= N_DEMO_ASSISTANT_TURNS


def _empty_message(turn: Turn) -> bool:
    return not turn.tool_calls and not turn.content.strip() and not turn.reasoning.strip()


def _g1(turn: Turn, *, scan_reasoning_for_code: bool) -> GuardFiring | None:
    reasons: list[str] = []
    if turn.invalid_tool_calls:
        reasons.append("invalid_tool_calls")
    if len(turn.tool_calls) > 1:
        reasons.append(f"{len(turn.tool_calls)} tool calls")
    elif len(turn.tool_calls) == 1:
        call = turn.tool_calls[0]
        if call.name != "execute":
            reasons.append(f"tool {call.name!r}")
        cmd = call.command
        if not isinstance(cmd, str) or not cmd.strip():
            reasons.append("empty command")
    if _empty_message(turn):
        reasons.append("empty message")
    content = turn.content or ""
    if CODE_FENCE in content or APPWORLD_EXEC in content:
        reasons.append("code in visible content")
    reasoning = turn.reasoning or ""
    if any(m in reasoning for m in TOOL_MARKUP):
        reasons.append("tool markup in reasoning")
    if scan_reasoning_for_code and (CODE_FENCE in reasoning or APPWORLD_EXEC in reasoning):
        reasons.append("code in reasoning")
    if not reasons:
        return None
    return GuardFiring(Guard.G1_INVALID_ACTION, turn.turn, "; ".join(reasons))


def _same_action_outcome(a: Turn, b: Turn) -> bool:
    sig_a, sig_b = api_signature(a.command), api_signature(b.command)
    if sig_a and sig_a == sig_b:
        return (a.observation or "") == (b.observation or "")
    return (a.command, a.observation or "") == (b.command, b.observation or "")


def _g2(prefix: tuple[Turn, ...]) -> GuardFiring | None:
    live = [t for t in prefix if _live(t)]
    if len(live) >= 3:
        a, b, c = live[-3], live[-2], live[-1]
        if _same_action_outcome(a, b) and _same_action_outcome(b, c):
            return GuardFiring(Guard.G2_LOOP, c.turn, "identical action+outcome x3")
    if len(live) >= 6:
        six = live[-6:]
        sigs = [api_signature(t.command) or (t.command,) for t in six]
        first, second = sigs[0], sigs[1]
        if first != second and sigs == [first, second] * 3:
            return GuardFiring(Guard.G2_LOOP, six[-1].turn, "A/B alternation x6")
    return None


def _g3(prefix: tuple[Turn, ...], trajectory: Trajectory) -> GuardFiring | None:
    turn = prefix[-1]
    obs = turn.observation or ""
    if HTTP_ERROR.search(obs):
        return GuardFiring(Guard.G3_FAILED_PRECONDITION, turn.turn, "HTTP 401/403/404/422")
    prior = " ".join([trajectory.initial_context, *(t.observation or "" for t in prefix[:-1])])
    for match in ENTITY_ID.findall(turn.command or ""):
        token = match.strip("'\"")
        if token and token not in prior:
            return GuardFiring(Guard.G3_FAILED_PRECONDITION, turn.turn,
                               f"unseen entity {token!r}")
    return None


def _g4(prefix: tuple[Turn, ...], trajectory: Trajectory) -> GuardFiring | None:
    turn = prefix[-1]
    sig = api_signature(turn.command)
    apps = {app for app, _, _ in sig}
    allowed = trajectory.allowed_apps
    if allowed:
        extra = apps - allowed - {"supervisor", "api_docs"}
        if extra:
            return GuardFiring(Guard.G4_CONSTRAINT_VIOLATION, turn.turn,
                               f"apps outside task: {sorted(extra)}")
    if DESTRUCTIVE.search(turn.command or ""):
        return GuardFiring(Guard.G4_CONSTRAINT_VIOLATION, turn.turn, "destructive write/delete")
    return None


def _had_verification_read(prefix: tuple[Turn, ...]) -> bool:
    """True unless some write call has no read call after it (call order, across turns),
    up to the first `complete_task` call."""
    calls: list[str] = []
    for t in prefix:
        for _, endpoint, _ in api_signature(t.command):
            if endpoint == "complete_task":
                break
            calls.append(endpoint)
    writes = [i for i, e in enumerate(calls) if is_write(e)]
    if not writes:
        return True
    return any(is_read(e) for e in calls[writes[-1] + 1:])


def _g5(prefix: tuple[Turn, ...], trajectory: Trajectory) -> GuardFiring | None:
    turn = prefix[-1]
    cmd = turn.command or ""
    if "complete_task" in cmd and not _had_verification_read(prefix):
        return GuardFiring(Guard.G5_UNVERIFIED_COMPLETION, turn.turn,
                           "complete_task without verification read")
    if turn.tool_calls:
        return None
    if (
        trajectory.completion_status is True
        or any('"task_completed": true' in (t.observation or "").lower() for t in prefix)
    ):
        return None
    return GuardFiring(Guard.G5_UNVERIFIED_COMPLETION, turn.turn,
                       "no tool call and task not complete")


def check_turn(prefix: tuple[Turn, ...], trajectory: Trajectory,
               *, scan_reasoning_for_code: bool = False) -> list[GuardFiring]:
    """Guards firing on the last turn of `prefix`, using only `prefix`.

    G1 is evaluated first; G5 only if G1 did not fire. G1 scans visible `content` for
    code fences / `appworld exec`, and `reasoning` only for structural tool markup
    (`<tool_call>`, `<function=`) unless `scan_reasoning_for_code` (the sensitivity split).
    """
    if not prefix:
        return []
    turn = prefix[-1]
    if not _live(turn):
        return []
    out: list[GuardFiring] = []
    g1 = _g1(turn, scan_reasoning_for_code=scan_reasoning_for_code)
    if g1:
        out.append(g1)
    for extra in (_g2(prefix), _g3(prefix, trajectory), _g4(prefix, trajectory)):
        if extra:
            out.append(extra)
    if g1 is None:
        g5 = _g5(prefix, trajectory)
        if g5:
            out.append(g5)
    return out


def evaluate(trajectory: Trajectory, *, scan_reasoning_for_code: bool = False
             ) -> list[GuardFiring]:
    """All firings over the trajectory, ordered by turn."""
    firings: list[GuardFiring] = []
    for i in range(len(trajectory.turns)):
        firings.extend(check_turn(trajectory.turns[: i + 1], trajectory,
                                  scan_reasoning_for_code=scan_reasoning_for_code))
    return firings


def first_nonterminal(firings: list[GuardFiring], T: int,  # noqa: N803
                      exclude: frozenset[Guard] = frozenset()) -> GuardFiring | None:
    """First firing with turn g < T whose guard is not in `exclude` (e.g. held-out G3)."""
    for firing in firings:
        if firing.turn < T and firing.guard not in exclude:
            return firing
    return None
