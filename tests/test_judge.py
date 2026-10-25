from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from guardfork.judge import (
    build_prompt,
    judge_trial,
    load_judge_results,
    make_openai_call,
    parse_response,
    task_instruction,
    truncate,
    write_judge,
)
from guardfork.telemetry.trials import TrialRecord, TurnRecord

MARKER = "[" + "APPWORLD_CURRENT_TASK" + "]"
TEMPLATE = "TASK<{{TASK}}> N<{{N_TURNS}}>\n{{TURNS}}\nanswer {\"turn\": int}"


def _demo_context() -> list[dict]:
    ctx = [{"type": "human", "content": "interface instructions"}]
    for i in range(10):
        ctx.append({"type": "ai", "content": f"demo {i}", "tool_calls": []})
        ctx.append({"type": "tool", "content": f"demo obs {i}"})
    ctx.append({"type": "human", "content": "Key instructions: act on your own."})
    ctx.append({"type": "human", "content": f"{MARKER}\nThis is a NEW task.\nTask: Pay Bob $5."})
    return ctx


def _record(n: int = 3, obs_len: int = 10) -> TrialRecord:
    base = _demo_context()
    events = []
    ctx = list(base)
    for i in range(n):
        events.append({
            "type": "reviewed_turn",
            "context": list(ctx),
            "original": {"reasoning": f"SECRET-THOUGHT-{i}", "content": f"visible {i}",
                         "tool_call": {"name": "execute", "command": f"cmd{i + 1}"}},
            "raw_original": {"tool_calls": [], "invalid_tool_calls": []},
        })
        ctx = ctx + [{"type": "ai", "content": f"visible {i}"},
                     {"type": "tool", "content": f"obs{i + 1}:" + "x" * obs_len}]
    turns = [TurnRecord(i, f"cmd{i + 1}", False) for i in range(n)]
    return TrialRecord(Path("trialA"), "jobA", "aaaaaaa_1", 0.0, None, 1.0, turns, events)


def test_task_instruction_prefers_marker_message():
    rec = _record()
    assert task_instruction(rec.events) == "This is a NEW task.\nTask: Pay Bob $5."


def test_task_instruction_falls_back_to_first_user_after_demos():
    ctx = [m for m in _demo_context() if MARKER not in str(m.get("content"))]
    assert task_instruction([{"context": ctx}]) == "Key instructions: act on your own."


def test_build_prompt_has_turns_commands_observations_no_reasoning():
    rec = _record(n=3, obs_len=5000)
    prompt = build_prompt(rec, TEMPLATE)
    assert prompt.startswith("TASK<This is a NEW task.\nTask: Pay Bob $5.> N<3>")
    for t in (1, 2, 3):
        assert f"### Turn {t}\nCommand:\ncmd{t}\n" in prompt
    assert "obs1:" in prompt and "obs2:" in prompt
    assert "(no observation recorded)" in prompt  # last turn has none
    assert "chars truncated" in prompt
    assert "SECRET-THOUGHT" not in prompt and "visible 0" not in prompt
    assert '{"turn": int}' in prompt  # literal braces in the template survive


def test_placeholders_not_resubstituted_from_task_text():
    rec = _record(n=1)
    rec.events[0]["context"][-1]["content"] = f"{MARKER}\nTask: say {{{{TURNS}}}}"
    prompt = build_prompt(rec, TEMPLATE)
    assert "say {{TURNS}}" in prompt


def test_truncate_keeps_head_and_tail():
    text = "H" * 2000 + "TAIL"
    out = truncate(text, 1500)
    assert out.startswith("H" * 100) and out.endswith("TAIL") and len(out) < 1600


@pytest.mark.parametrize("raw, expected", [
    ('{"turn": 2, "reason": "bad id"}', (2, "bad id", None)),
    ('```json\n{"turn": 3, "reason": "x"}\n```', (3, "x", None)),
    ('Sure! Here is my answer:\n{"turn": "2", "reason": "y"} hope it helps', (2, "y", None)),
    ('<think>maybe {"turn": 1}</think>\n{"turn": 3, "reason": "z"}', (3, "z", None)),
    ('{"note": {"a": 1}} then {"turn": 1, "reason": "has } brace"}', (1, "has } brace", None)),
    ('{"turn": 9, "reason": "late"}', (None, "late", "turn 9 out of range 1..3")),
    ('{"turn": 0, "reason": "zero"}', (None, "zero", "turn 0 out of range 1..3")),
    ('{"turn": true, "reason": "b"}', (None, "b", "turn is not an integer")),
    ('{"turn": 2, reason: broken json', (2, None, None)),
    ("no json here", (None, None, "no JSON with a turn field")),
    ("", (None, None, "no JSON with a turn field")),
])
def test_parse_response_edge_cases(raw, expected):
    assert parse_response(raw, 3) == expected


class _FakeChat(BaseHTTPRequestHandler):
    requests: list[dict] = []
    reply = '```json\n{"turn": 2, "reason": "used an unseen id"}\n```'

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body})
        payload = json.dumps({"choices": [{"message": {"role": "assistant",
                                                       "content": type(self).reply}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_server():
    _FakeChat.requests = []
    server = HTTPServer(("127.0.0.1", 0), _FakeChat)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def test_judge_trial_against_fake_server(fake_server, tmp_path):
    rec = _record(n=3)
    call = make_openai_call(fake_server, "fake-model", timeout=10)
    result = judge_trial(rec, call, model="fake-model", template=TEMPLATE,
                         template_sha256="abc")
    req = _FakeChat.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["body"]["model"] == "fake-model" and req["body"]["temperature"] == 0.0
    assert req["body"]["messages"][0]["role"] == "user"
    assert result["turn"] == 2 and result["reason"] == "used an unseen id"
    assert result["prompt_sha256"] == "abc" and result["model"] == "fake-model"
    assert result["raw"].startswith("```json") and result["error"] is None
    path = write_judge(result, tmp_path)
    assert path == tmp_path / "jobA" / "trialA" / "judge.json"
    assert load_judge_results(tmp_path)["jobA/trialA"]["turn"] == 2


def test_judge_trial_records_connection_error():
    rec = _record(n=2)
    call = make_openai_call("http://127.0.0.1:9/v1", "m", timeout=2)
    result = judge_trial(rec, call, model="m", template=TEMPLATE, template_sha256="s")
    assert result["turn"] is None and result["error"].startswith("request failed")


def test_q1_judge_row(tmp_path):
    from guardfork.q1.localization import build_rows, report_table

    rec = _record(n=4)
    oracle = {"jobA/trialA": {"t_star": 2}}
    rows = build_rows([rec], oracle, judge_results={"jobA/trialA": {"turn": 2}})
    assert rows[0].localizers["judge"] == 2
    md = report_table(rows)
    judge_line = next(line for line in md.splitlines() if line.startswith("| judge |"))
    assert judge_line.startswith("| judge | 1 | 1/1")

    write_judge({"trial_id": "jobA/trialA", "turn": 3}, tmp_path)
    rows = build_rows([rec], oracle, judge_root=tmp_path)
    assert rows[0].localizers["judge"] == 3
    rows = build_rows([rec], oracle)
    assert rows[0].localizers["judge"] is None
    assert "| judge | 0 |" in report_table(rows)
