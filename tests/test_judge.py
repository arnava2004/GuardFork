from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError

import pytest

from guardfork.judge import (
    JUDGE_VERSION,
    build_prompt,
    estimate_usd,
    judge_path,
    judge_trial,
    load_judge_results,
    load_judges,
    make_anthropic_call,
    make_openai_call,
    parse_distribution,
    parse_response,
    prepare_prompt,
    task_instruction,
    top_k,
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
        type(self).requests.append({"path": self.path, "body": body,
                                    "auth": self.headers.get("Authorization")})
        payload = json.dumps({"id": "chatcmpl-1", "model": "fake-model-001",
                              "system_fingerprint": "fp_1",
                              "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                              "choices": [{"message": {"role": "assistant",
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
    assert "judge" not in rows[0].localizers  # legacy slot only when data exists
    assert rows[0].localizers["judge_open"] is None
    md = report_table(rows)
    assert "| judge |" not in md and "| judge_open | 0 |" in md


# ------------------------------------------------------------------ distributions


def test_parse_distribution_top3_and_reasons():
    raw = ('<think>hmm</think>```json\n{"turns": [{"turn": 3, "p": 0.6, "reason": "a"}, '
           '{"turn": 2, "p": 0.3, "reason": "b"}, {"turn": 5, "p": 0.1, "reason": "c"}]}\n```')
    dist, top, reasons, err = parse_distribution(raw, 6)
    assert err is None and top == 3
    assert [round(p, 6) for p in dist] == [0, 0.3, 0.6, 0, 0.1, 0]
    assert reasons == {3: "a", 2: "b", 5: "c"}
    assert top_k(dist, 3) == [(3, 0.6), (2, 0.3), (5, 0.1)]


def test_parse_distribution_legacy_single_turn_is_one_hot():
    dist, top, reasons, err = parse_distribution('{"turn": 2, "reason": "x"}', 4)
    assert (dist, top, reasons, err) == ([0.0, 1.0, 0.0, 0.0], 2, {2: "x"}, None)
    dist, top, _, err = parse_distribution('{"turn": 9}', 4)
    assert top is None and err == "turn 9 out of range 1..4" and sum(dist) == pytest.approx(1)


def test_parse_distribution_drops_out_of_range_and_renormalises_full_mass():
    raw = '{"turns": [{"turn": 2, "p": 0.6}, {"turn": 9, "p": 0.4}]}'
    dist, top, _, err = parse_distribution(raw, 3)
    assert err is None and top == 2 and dist == [0.0, 1.0, 0.0]  # s >= 1: renormalise
    dist, top, _, err = parse_distribution('{"turns": [{"turn": 0, "p": 1}]}', 3)
    assert top is None and err.startswith("turn 0 out of range")


def test_parse_distribution_spreads_leftover_only_when_understated():
    dist, top, _, err = parse_distribution('{"turns": [{"turn": 2, "p": 0.6}]}', 4)
    assert err is None and top == 2
    assert dist == pytest.approx([0.4 / 3, 0.6, 0.4 / 3, 0.4 / 3])
    # over-stated mass (percentages) is scaled down, nothing spread
    dist, top, _, err = parse_distribution(
        '{"turns": [{"turn": 1, "p": 70}, {"turn": 4, "p": 30}]}', 4)
    assert err is None and top == 1 and [round(p, 3) for p in dist] == [0.7, 0, 0, 0.3]
    # duplicates merge; more than 3 entries are cut to the first 3
    raw = '{"turns": [{"turn": 1, "p": 0.3}, {"turn": 1, "p": 0.3}, {"turn": 2, "p": 0.4}, ' \
          '{"turn": 3, "p": 0.9}]}'
    dist, top, _, err = parse_distribution(raw, 3)
    assert err is None and top == 1 and [round(p, 3) for p in dist] == [0.6, 0.4, 0.0]


def test_parse_distribution_unweighted_entries_share_mass_and_errors():
    dist, top, _, err = parse_distribution('{"turns": [{"turn": 1}, {"turn": 3}]}', 3)
    assert err is None and top == 1 and dist == [0.5, 0.0, 0.5]
    dist, top, _, err = parse_distribution("nothing", 3)
    assert top is None and err and dist == [pytest.approx(1 / 3)] * 3
    _, top, _, err = parse_distribution('{"turns": [{"turn": 2, "p": 0}]}', 3)
    assert top is None and "zero" in err
    assert parse_response('{"turns": [{"turn": 2, "p": 1, "reason": "r"}]}', 3) == (2, "r", None)


# ------------------------------------------------------------------ truncation


def test_prepare_prompt_elides_middle_turns_deterministically():
    rec = _record(n=12, obs_len=400)
    full, trunc = prepare_prompt(rec, TEMPLATE, obs_chars=1500, max_prompt_chars=None)
    assert trunc["elided_turns"] == [] and trunc["prompt_chars"] == len(full)
    assert trunc["prompt_tokens_est"] == len(full) // 4 and trunc["fits"]
    budget = len(full) - 1000
    prompt, trunc = prepare_prompt(rec, TEMPLATE, obs_chars=1500, max_prompt_chars=budget)
    assert len(prompt) <= budget and trunc["fits"]
    elided = trunc["elided_turns"]
    assert elided and elided[0] == 4 and elided == list(range(4, elided[-1] + 1))
    assert elided[-1] <= 12 - 5
    assert f"### Turns 4–{elided[-1]} elided ({len(elided)} turns)" in prompt
    for t in (1, 2, 3, 8, 9, 10, 11, 12):
        assert f"### Turn {t}\n" in prompt
    for t in elided:
        assert f"### Turn {t}\n" not in prompt
    again, trunc2 = prepare_prompt(rec, TEMPLATE, obs_chars=1500, max_prompt_chars=budget)
    assert again == prompt and trunc2 == trunc
    assert trunc["obs_chars"] == 1500 and trunc["max_prompt_chars"] == budget
    # too short a budget: head + tail kept, flagged as not fitting
    tiny, trunc = prepare_prompt(rec, TEMPLATE, obs_chars=1500, max_prompt_chars=10)
    assert trunc["elided_turns"] == [4, 5, 6, 7] and not trunc["fits"]
    assert build_prompt(rec, TEMPLATE) == full


def test_prepare_prompt_short_runs_never_elide():
    rec = _record(n=6, obs_len=400)
    prompt, trunc = prepare_prompt(rec, TEMPLATE, obs_chars=1500, max_prompt_chars=10)
    assert trunc["elided_turns"] == [] and not trunc["fits"] and "### Turn 6" in prompt


# ------------------------------------------------------------------ providers


class _FakeAnthropic(BaseHTTPRequestHandler):
    requests: list[dict] = []
    statuses: list[int] = []
    reply_text = '{"turns": [{"turn": 2, "p": 0.8, "reason": "r"}, {"turn": 1, "p": 0.2}]}'

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"path": self.path, "body": body,
                                    "headers": {k.lower(): v for k, v in self.headers.items()}})
        status = type(self).statuses.pop(0) if type(self).statuses else 200
        if status != 200:
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b'{"error": "busy"}')
            return
        payload = json.dumps({
            "id": "msg_01", "model": "claude-sonnet-4-5-20250929", "stop_reason": "end_turn",
            "content": [{"type": "thinking", "thinking": "..."},
                        {"type": "text", "text": type(self).reply_text}],
            "usage": {"input_tokens": 1000, "output_tokens": 50},
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload.encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_anthropic():
    _FakeAnthropic.requests = []
    _FakeAnthropic.statuses = []
    server = HTTPServer(("127.0.0.1", 0), _FakeAnthropic)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_anthropic_call_headers_body_text_and_meta(fake_anthropic, monkeypatch):
    monkeypatch.setenv("FAKE_ANTHROPIC_KEY", "test-key-value")
    call = make_anthropic_call("claude-sonnet-4-5", api_key_env="FAKE_ANTHROPIC_KEY",
                               base_url=fake_anthropic, max_tokens=512, timeout=10)
    text, meta = call([{"role": "user", "content": "hello"}])
    req = _FakeAnthropic.requests[0]
    assert req["path"] == "/v1/messages"
    assert req["headers"]["x-api-key"] == "test-key-value"
    assert req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["headers"]["content-type"] == "application/json"
    assert req["body"] == {"model": "claude-sonnet-4-5", "max_tokens": 512, "temperature": 0.0,
                           "messages": [{"role": "user", "content": "hello"}]}
    assert text == _FakeAnthropic.reply_text
    assert meta == {"provider": "anthropic", "model_returned": "claude-sonnet-4-5-20250929",
                    "response_id": "msg_01", "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1000, "output_tokens": 50}}


def test_judge_trial_record_fields_with_anthropic(fake_anthropic, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_ANTHROPIC_KEY", "k")
    rec = _record(n=3)
    call = make_anthropic_call("claude-sonnet-4-5", api_key_env="FAKE_ANTHROPIC_KEY",
                               base_url=fake_anthropic, timeout=10)
    result = judge_trial(rec, call, judge_id="judge_claude", provider="anthropic",
                         model="claude-sonnet-4-5", template=TEMPLATE, template_sha256="abc",
                         sampling={"temperature": 0.0, "max_tokens": 4096}, eval_only=True)
    assert result["judge_id"] == "judge_claude" and result["provider"] == "anthropic"
    assert result["model_requested"] == "claude-sonnet-4-5"
    assert result["model_returned"] == "claude-sonnet-4-5-20250929"
    assert result["response_id"] == "msg_01" and result["stop_reason"] == "end_turn"
    assert result["timestamp_utc"].endswith("+00:00")
    assert result["sampling"] == {"temperature": 0.0, "max_tokens": 4096}
    assert result["usage"] == {"input_tokens": 1000, "output_tokens": 50}
    assert result["truncation"]["elided_turns"] == []
    assert result["distribution"] == pytest.approx([0.2, 0.8, 0.0]) and result["turn"] == 2
    assert result["reasons"] == {"2": "r"} and result["reason"] == "r"
    assert result["eval_only"] is True and result["error"] is None
    assert result["judge_version"] == JUDGE_VERSION
    path = write_judge(result, tmp_path)
    assert path == tmp_path / "jobA" / "trialA" / "judge_claude.json"
    assert load_judge_results(tmp_path, "judge_claude")["jobA/trialA"]["turn"] == 2
    assert load_judge_results(tmp_path) == {}  # legacy judge.json absent
    assert estimate_usd(result["usage"], 3.0, 15.0) == pytest.approx(0.003 + 0.00075)


def test_retry_on_429_then_success(fake_anthropic, monkeypatch):
    import guardfork.judge as judge_mod

    slept: list[float] = []
    monkeypatch.setattr(judge_mod, "_sleep", slept.append)
    monkeypatch.setenv("FAKE_ANTHROPIC_KEY", "k")
    _FakeAnthropic.statuses = [429]
    call = make_anthropic_call("m", api_key_env="FAKE_ANTHROPIC_KEY", base_url=fake_anthropic,
                               timeout=10)
    text, _ = call([{"role": "user", "content": "x"}])
    assert text == _FakeAnthropic.reply_text and len(_FakeAnthropic.requests) == 2
    assert slept == [2.0]
    # 401 is not retried
    _FakeAnthropic.requests = []
    _FakeAnthropic.statuses = [401]
    with pytest.raises(HTTPError):
        call([{"role": "user", "content": "x"}])
    assert len(_FakeAnthropic.requests) == 1
    # three retryable failures exhaust the attempts
    _FakeAnthropic.requests = []
    _FakeAnthropic.statuses = [529, 503, 500]
    with pytest.raises(HTTPError):
        call([{"role": "user", "content": "x"}])
    assert len(_FakeAnthropic.requests) == 3 and slept == [2.0, 2.0, 4.0]


def test_missing_env_var_raises_before_any_request(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_KEY_ENV", raising=False)
    with pytest.raises(RuntimeError, match="DEFINITELY_UNSET_KEY_ENV"):
        make_anthropic_call("m", api_key_env="DEFINITELY_UNSET_KEY_ENV")
    with pytest.raises(RuntimeError, match="DEFINITELY_UNSET_KEY_ENV"):
        make_openai_call("http://127.0.0.1:9/v1", "m", api_key_env="DEFINITELY_UNSET_KEY_ENV")
    with pytest.raises(RuntimeError):
        make_anthropic_call("m", api_key_env="")


def test_openai_call_returns_meta_and_uses_env_key(fake_server, monkeypatch):
    monkeypatch.setenv("FAKE_OPENROUTER_KEY", "or-key")
    call = make_openai_call(fake_server, "fake-model", api_key_env="FAKE_OPENROUTER_KEY",
                            timeout=10, extra_body={"chat_template_kwargs": {"x": 1}})
    text, meta = call([{"role": "user", "content": "q"}])
    assert text.startswith("```json")
    assert meta["provider"] == "openai_compat" and meta["model_returned"] == "fake-model-001"
    assert meta["system_fingerprint"] == "fp_1" and meta["response_id"] == "chatcmpl-1"
    assert meta["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert _FakeChat.requests[0]["body"]["chat_template_kwargs"] == {"x": 1}
    assert _FakeChat.requests[0]["auth"] == "Bearer or-key"
    assert estimate_usd(meta["usage"], 1.0, 2.0) == pytest.approx(20 / 1e6)
    assert estimate_usd(None, 1.0, 2.0) == 0.0


def test_judge_path_naming():
    assert judge_path(Path("r"), "j", "t") == Path("r/j/t/judge.json")
    assert judge_path(Path("r"), "j", "t", "judge") == Path("r/j/t/judge.json")
    assert judge_path(Path("r"), "j", "t", "judge_open") == Path("r/j/t/judge_open.json")
    assert judge_path(Path("r"), "j", "t", "x") == Path("r/j/t/judge_x.json")


def test_load_judges_config():
    judges = load_judges()
    assert set(judges) >= {"judge_open", "judge_local", "judge_claude"}
    assert judges["judge_claude"].provider == "anthropic" and judges["judge_claude"].eval_only
    assert judges["judge_claude"].api_key_env == "ANTHROPIC_API_KEY"
    assert judges["judge_open"].api_key_env == "OPENROUTER_API_KEY"
    assert not judges["judge_open"].eval_only and not judges["judge_local"].eval_only
    assert judges["judge_local"].extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    for spec in judges.values():
        assert spec.obs_chars == 1500 and spec.max_prompt_chars == 60000
        assert spec.temperature == 0.0 and spec.sampling["max_tokens"] == spec.max_tokens
