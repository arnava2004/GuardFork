"""Mandatory Weekend 1 replay validations (0 LLM calls each)."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from guardfork.fork.harbor_replay import ReplayDiverged, build_replay_plan
from guardfork.telemetry.trials import load_trial


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return ""


def _assistant_text(m: dict[str, Any]) -> tuple[str, str]:
    extra = m.get("additional_kwargs") or {}
    explicit = (
        m.get("reasoning_content")
        or m.get("reasoning")
        or extra.get("reasoning_content")
        or extra.get("reasoning")
    )
    content = m.get("content") or ""
    if isinstance(content, list):
        r_parts = [
            str(b.get("reasoning") or b.get("thinking") or "")
            for b in content
            if isinstance(b, dict) and b.get("type") in {"reasoning", "thinking"}
        ]
        text = "\n".join(
            str(b["text"])
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        )
        reasoning = explicit or "\n".join(p for p in r_parts if p)
        return str(reasoning or ""), text
    return str(explicit or ""), str(content)


def _thinking_content(reasoning: str, text: str) -> str | list[dict[str, str]]:
    if not reasoning:
        return text
    parts = [{"type": "thinking", "thinking": reasoning}]
    if text:
        parts.append({"type": "text", "text": text})
    return parts


def _lc_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    if isinstance(tc.get("function"), dict):
        fn = tc["function"]
        args = fn.get("arguments") or ""
    else:
        fn = {"name": tc.get("name", "")}
        args = json.dumps(tc.get("args") or {}, ensure_ascii=False)
    with contextlib.suppress(json.JSONDecodeError):
        args = json.dumps(json.loads(args), ensure_ascii=False, sort_keys=True)
    return {
        "id": tc.get("id"),
        "type": "function",
        "function": {"name": fn["name"], "arguments": args},
    }


def _lc_to_openai(m: dict[str, Any]) -> dict[str, Any] | None:
    t = m.get("type") or m.get("role")
    if t in ("system", "user", "human"):
        content = m.get("content")
        return {
            "role": "system" if t == "system" else "user",
            "content": (
                content
                if t in ("user", "human") and isinstance(content, list)
                else _text_of(content)
            ),
        }
    if t in ("ai", "assistant"):
        reasoning, text = _assistant_text(m)
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": _thinking_content(reasoning, text),
        }
        tcs = m.get("tool_calls") or []
        if tcs:
            msg["tool_calls"] = [_lc_tool_call(tc) for tc in tcs]
        return msg
    if t == "tool":
        return {
            "role": "tool",
            "tool_call_id": m.get("tool_call_id"),
            "content": _text_of(m.get("content")),
        }
    return None


def canonical_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize LangChain/OpenAI messages like Harness-Zero `build_sft.canonical`."""
    out = []
    for raw in messages:
        conv = _lc_to_openai(raw)
        if conv is not None:
            out.append(conv)
    return out


@dataclass(frozen=True)
class ReconstructionReport:
    turns_checked: int
    canonical_equal: int
    raw_bytes_equal: int


def _event_prefix(event: dict[str, Any]) -> list[dict[str, Any]]:
    msgs = []
    if event.get("system_message"):
        msgs.append(event["system_message"])
    msgs.extend(event.get("context") or [])
    return msgs


def offline_request_reconstruction(trial_dir: Path) -> ReconstructionReport:
    """Rebuild the logged prefix at every turn and compare against
    `student_model_requests.jsonl` using canonical JSON."""
    teacher = trial_dir / "agent" / "teacher"
    events_path = teacher / "trial" / "trajectory.jsonl"
    req_path = trial_dir / "agent" / "student_model_requests.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    requests = [json.loads(line) for line in req_path.read_text().splitlines() if line.strip()]
    req_i = 0
    equal = raw_eq = 0
    for event in events:
        prefix = canonical_messages(_event_prefix(event))
        raw_prefix = json.dumps(_event_prefix(event), ensure_ascii=False)
        found = False
        while req_i < len(requests):
            payload = requests[req_i]
            req_i += 1
            logged = payload.get("messages") or payload.get("request", {}).get("messages") or []
            if canonical_messages(logged) == prefix:
                equal += 1
                if json.dumps(logged, ensure_ascii=False) == raw_prefix:
                    raw_eq += 1
                found = True
                break
        if not found:
            continue
    return ReconstructionReport(len(events), equal, raw_eq)


@dataclass(frozen=True)
class RewardReplayReport:
    n: int
    reproduced: int
    diverged: list[str]


ReplayOnce = Callable[[Path], float | None]


def full_replay_reward_parity(trial_dirs: list[Path],
                              replay: ReplayOnce | None = None) -> RewardReplayReport:
    """Replay each complete trajectory (t = T + 1, no model calls) and require the
    verifier to reproduce the original reward. Inject `replay` in tests."""
    diverged: list[str] = []
    reproduced = 0
    for path in trial_dirs:
        rec = load_trial(path / "agent" / "teacher", path.name)
        if replay is None:
            raise RuntimeError("full_replay_reward_parity needs a Harbor replay callback")
        got = replay(path)
        if rec.reward is not None and got == rec.reward:
            reproduced += 1
        else:
            diverged.append(str(path))
    return RewardReplayReport(len(trial_dirs), reproduced, diverged)


def determinism_check(trial_dirs: list[Path], turns_per_trial: int = 3,
                      replay_obs: Callable[[Path, int], str] | None = None) -> float:
    """Fraction of (trial, turn) replays whose observations match exactly under
    PYTHONHASHSEED=0, excluding timeouts and subagent turns. Feeds W1-a (>= 0.85)."""
    checked = matched = 0
    for path in trial_dirs:
        rec = load_trial(path / "agent" / "teacher", path.name)
        n = rec.n_turns
        if n <= 1:
            continue
        candidates = [tr for tr in rec.turns[:-1]
                      if not tr.subagent_call and tr.command
                      and "TimeoutError" not in (tr.parse_reasons or [])]
        sample = candidates[:turns_per_trial]
        for tr in sample:
            checked += 1
            plan = build_replay_plan(rec, tr.index + 1)
            expected = plan.expected_observations[-1] if plan.expected_observations else ""
            if replay_obs is None:
                raise RuntimeError("determinism_check needs a replay_obs callback")
            try:
                got = replay_obs(path, tr.index + 1)
            except ReplayDiverged:
                continue
            if got == expected:
                matched += 1
    return matched / checked if checked else 0.0
