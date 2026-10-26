from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from guardfork.fork.harbor_replay import ReplayDiverged, replay_prefix
from guardfork.fork.replay_checks import canonical_messages, offline_request_reconstruction
from guardfork.manifest import Manifest, harness_hash, write_manifest


def test_replay_prefix_raises_on_mismatch():
    async def execute(cmd: str) -> str:
        return "wrong"

    with pytest.raises(ReplayDiverged) as exc:
        asyncio.run(replay_prefix(("ls",), ("ok",), execute))
    assert exc.value.turn == 1


def test_replay_prefix_accepts_match():
    async def execute(cmd: str) -> str:
        return "ok"

    asyncio.run(replay_prefix(("ls", "cat"), ("ok", "ok"), execute))


def test_canonical_messages_sorts_arguments():
    msgs = [{"role": "assistant", "tool_calls": [
        {"function": {"name": "execute", "arguments": '{"b": 1, "a": 2}'}}]}]
    got = canonical_messages(msgs)
    assert '"a"' in got[0]["tool_calls"][0]["function"]["arguments"]


def test_offline_request_reconstruction(tmp_path):
    trial = tmp_path / "trial"
    teacher = trial / "agent" / "teacher" / "trial"
    teacher.mkdir(parents=True)
    event = {"system_message": None, "context": [{"role": "user", "content": "hi"}]}
    (teacher / "trajectory.jsonl").write_text(json.dumps(event) + "\n")
    req = {"messages": [{"role": "user", "content": "hi"}]}
    (trial / "agent" / "student_model_requests.jsonl").write_text(json.dumps(req) + "\n")
    report = offline_request_reconstruction(trial)
    assert report.turns_checked == 1 and report.canonical_equal == 1


def _record_with_contexts():
    from guardfork.telemetry.trials import TrialRecord, TurnRecord

    sys_msg = {"type": "system", "content": "sys"}
    base = [{"type": "human", "content": "task"}]
    events = []
    for i in range(3):
        ctx = list(base)
        for j in range(i):
            ctx += [{"type": "ai", "content": "", "tool_calls": [
                {"name": "execute", "args": {"command": f"c{j}"}, "id": f"id{j}",
                 "type": "tool_call"}]},
                    {"type": "tool", "content": f"o{j}", "tool_call_id": f"id{j}"}]
        cmd = f"c{i}" if i < 2 else None
        events.append({"system_message": sys_msg, "context": ctx,
                       "original": {"tool_call": {"name": "execute", "command": cmd}
                                    if cmd else None, "content": "", "reasoning": ""},
                       "raw_original": {"tool_calls": [], "invalid_tool_calls": []}})
    turns = [TurnRecord(i, f"c{i}" if i < 2 else None, False) for i in range(3)]
    return TrialRecord(Path("t"), "j", "aaaaaaa_1", 1.0, None, 1.0, turns, events)


def test_fork_plan_prefix_excludes_system_message():
    from guardfork.fork.harbor_replay import build_replay_plan

    plan = build_replay_plan(_record_with_contexts(), 2)
    assert plan.commands == ("c0",) and plan.expected_observations == ("o0",)
    assert [m["type"] for m in plan.message_prefix] == ["human", "ai", "tool"]


def test_full_replay_skips_no_command_turn_and_roundtrips(tmp_path):
    from guardfork.fork.harbor_replay import full_replay_plan, load_plan, save_plan

    plan = full_replay_plan(_record_with_contexts())
    assert plan.full_replay and plan.commands == ("c0", "c1")
    assert plan.expected_observations == ("o0", "o1")
    path = save_plan(plan, tmp_path / "p.json")
    assert load_plan(path) == plan


def test_manifest_roundtrip(tmp_path):
    compose = tmp_path / "compose.yaml"
    compose.write_text("services: {}\n")
    digest = harness_hash(tmp_path, compose)
    assert len(digest) == 64
    man = Manifest("abc", "cfg", digest, "Qwen/Qwen3.5-4B", "vllm", None, 0, 1.0, {"mem": "8g"})
    path = write_manifest(tmp_path / "trial", man)
    assert path.name == "guardfork_manifest.json"
    assert "Qwen" in path.read_text()
