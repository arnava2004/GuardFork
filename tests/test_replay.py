from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from guardfork.fork.harbor_replay import ReplayDiverged, ReplayPlan, replay_prefix
from guardfork.fork.launch import launch_continuations
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


def test_launch_uses_injected_runner(tmp_path, make_trial, event):
    job = make_trial("forkk", "k0", "aaaaaaa_1", 1.0, [event("aaaaaaa_1")])
    plan = ReplayPlan(tmp_path, "aaaaaaa_1", 2, "abc", ("ls",), ("ok",), ())

    def runner(_plan, k, dest: Path) -> Path:
        return job

    results = launch_continuations(plan, 1, job_name="f", output_dir=tmp_path / "out",
                                   concurrency=1, eval_seed=0, runner=runner)
    assert results[0].reward == 1.0


def test_launch_without_runner_errors(tmp_path):
    plan = ReplayPlan(tmp_path, "aaaaaaa_1", 2, "abc", (), (), ())
    with pytest.raises(RuntimeError):
        launch_continuations(plan, 1, job_name="f", output_dir=tmp_path, concurrency=1,
                             eval_seed=0)


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


def test_manifest_roundtrip(tmp_path):
    compose = tmp_path / "compose.yaml"
    compose.write_text("services: {}\n")
    digest = harness_hash(tmp_path, compose)
    assert len(digest) == 64
    man = Manifest("abc", "cfg", digest, "Qwen/Qwen3.5-4B", "vllm", None, 0, 1.0, {"mem": "8g"})
    path = write_manifest(tmp_path / "trial", man)
    assert path.name == "guardfork_manifest.json"
    assert "Qwen" in path.read_text()
