from __future__ import annotations

import json
from pathlib import Path

from guardfork.fork.harbor_replay import ReplayPlan, save_plan
from guardfork.telemetry.trials import iter_trials
from guardfork.train.export import (
    EXECUTE_TOOL,
    build_fork_pairs,
    mlx_split,
    pairs_from_forks_json,
    pairs_from_plan,
    render_qwen35,
    to_mlx_row,
    to_neutral_row,
)

SYSTEM = {"type": "system", "content": "You are an agent."}


def _exec(code: str) -> str:
    return f"appworld exec <<'APPWORLD_PY'\n{code}\nAPPWORLD_PY"


def _ai(i: int, cmd: str, reasoning: str = "") -> dict:
    return {"type": "ai", "content": f"step {i}",
            "additional_kwargs": {"reasoning_content": reasoning},
            "tool_calls": [{"name": "execute", "args": {"command": cmd}, "id": f"c{i}"}]}


def _tool(i: int, text: str) -> dict:
    return {"type": "tool", "content": text, "tool_call_id": f"c{i}"}


PREFIX = [
    {"type": "human", "content": "Use the execute tool."},
    _ai(0, "ls", "look around"),
    _tool(0, "files"),
    {"type": "human", "content": "Task: pay the bill."},
    _ai(1, _exec("print(apis.api_docs.show_api_descriptions(app_name='venmo'))"), "docs"),
    _tool(1, "[...]"),
]


def _event(context: list[dict], cmd: str, reasoning: str, content: str = "") -> dict:
    return {"type": "reviewed_turn", "task_id": "aaaaaaa_1", "system_message": SYSTEM,
            "context": context,
            "original": {"reasoning": reasoning, "content": content,
                         "tool_call": {"name": "execute", "command": cmd}},
            "raw_original": {"invalid_tool_calls": [],
                             "tool_calls": [{"name": "execute", "args": {"command": cmd},
                                             "id": "x"}]}}


BAD = _exec("print(apis.venmo.pay_bill(bill_id=7))")
GOOD = _exec("token = apis.venmo.login(username='u', password='p')")


def _source_events() -> list[dict]:
    first = _event(PREFIX[:4], PREFIX[4]["tool_calls"][0]["args"]["command"], "docs", "step 1")
    return [first, _event(PREFIX, BAD, "pay now", "paying")]


def _fork_job(make_trial, name: str, conts: list[tuple[str, float, str]],
              prefix: list[dict] | None = None) -> Path:
    job = None
    for k, (cmd, reward, reasoning) in enumerate(conts):
        job = make_trial(name, f"aaaaaaa_1__k{k}", "aaaaaaa_1", reward,
                         [_event(prefix or PREFIX, cmd, reasoning, "cont")])
    return job


def _records(make_trial, conts, prefix=None, source_reward=0.0):
    src_job = make_trial("base", "aaaaaaa_1__src", "aaaaaaa_1", source_reward, _source_events())
    source = next(iter_trials(src_job))
    fork_job = _fork_job(make_trial, "fork-t2", conts, prefix)
    return source, list(iter_trials(fork_job))


def test_pair_prompt_chosen_rejected(make_trial):
    source, conts = _records(make_trial, [(GOOD, 1.0, "log in first"), (BAD, 0.0, "x")])
    pairs, stats = build_fork_pairs(source, 2, conts)
    assert stats.passing == 1 and stats.pairs == 1 and len(pairs) == 1
    p = pairs[0]
    assert [m["role"] for m in p.prompt] == ["system", "user", "assistant", "tool", "user",
                                            "assistant", "tool"]
    assert p.prompt[2]["reasoning_content"] == "look around"
    assert json.loads(p.prompt[2]["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}
    assert p.tools == [EXECUTE_TOOL]
    assert p.chosen["role"] == "assistant" and p.chosen["reasoning_content"] == "log in first"
    assert json.loads(p.chosen["tool_calls"][0]["function"]["arguments"])["command"] == GOOD
    assert json.loads(p.rejected["tool_calls"][0]["function"]["arguments"])["command"] == BAD
    assert p.rejected["reasoning_content"] == "pay now" and p.rejected["content"] == "paying"
    assert p.chosen_reward == 1.0 and p.rejected_reward == 0.0
    assert not p.think_only_diff and not p.signature_collision
    row = to_neutral_row(p)
    assert row["chosen"] == [p.chosen] and row["rejected"] == [p.rejected]
    assert row["meta"]["fork_turn"] == 2 and row["meta"]["source_trial"] == "base/aaaaaaa_1__src"


def test_signature_collision_dropped_and_max_per_fork(make_trial):
    same_sig = _exec("print(apis.venmo.pay_bill(bill_id=8))")
    source, conts = _records(make_trial, [(same_sig, 1.0, "a"), (GOOD, 1.0, "b"),
                                          (GOOD, 1.0, "c")])
    pairs, stats = build_fork_pairs(source, 2, conts)
    assert stats.dropped_collision == 1 and stats.passing == 3 and len(pairs) == 1
    pairs, _ = build_fork_pairs(source, 2, conts, max_per_fork=5)
    assert len(pairs) == 2
    pairs, _ = build_fork_pairs(source, 2, conts, max_per_fork=5, allow_collision=True)
    assert len(pairs) == 3 and pairs[0].signature_collision


def test_think_only_flagged_and_identical_dropped(make_trial):
    src_job = make_trial("base", "aaaaaaa_1__src", "aaaaaaa_1", 0.0,
                         [_event(PREFIX, "ls -la", "first thought", "c")])
    source = next(iter_trials(src_job))
    job = make_trial("fork", "k0", "aaaaaaa_1", 1.0,
                     [_event(PREFIX, "ls -la", "other thought", "c")])
    make_trial("fork", "k1", "aaaaaaa_1", 1.0, [_event(PREFIX, "ls -la", "first thought", "c")])
    conts = list(iter_trials(job))
    pairs, stats = build_fork_pairs(source, 1, conts, max_per_fork=5)
    assert len(pairs) == 1 and pairs[0].think_only_diff and stats.think_only == 1
    assert stats.dropped_identical == 1
    pairs, _ = build_fork_pairs(source, 1, conts, max_per_fork=5, allow_identical=True)
    assert [p.identical for p in pairs] == [False, True]


def test_prefix_mismatch_and_passing_source(make_trial):
    other = PREFIX[:3] + [{"type": "human", "content": "Task: something else."}] + PREFIX[4:]
    source, conts = _records(make_trial, [(GOOD, 1.0, "b")], prefix=other)
    pairs, stats = build_fork_pairs(source, 2, conts)
    assert pairs == [] and stats.dropped_prefix_mismatch == 1
    assert len(build_fork_pairs(source, 2, conts, require_prefix_match=False)[0]) == 1


def test_passing_source_skipped(make_trial):
    source, conts = _records(make_trial, [(GOOD, 1.0, "b")], source_reward=1.0)
    pairs, stats = build_fork_pairs(source, 2, conts)
    assert pairs == [] and "did not fail" in stats.skipped_source[0]


def test_render_qwen35_small_conversation():
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": " hi "},
        {"role": "assistant", "content": "old", "reasoning_content": "hidden"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "\n\nok", "reasoning_content": "think",
         "tool_calls": [{"type": "function", "id": "1",
                         "function": {"name": "execute", "arguments": '{"command": "ls"}'}}]},
        {"role": "tool", "content": "out", "tool_call_id": "1"},
    ]
    tool = {"type": "function", "function": {"name": "execute"}}
    got = render_qwen35(msgs, [tool])
    assert got.startswith('<|im_start|>system\n# Tools\n\nYou have access to the following '
                          'functions:\n\n<tools>\n{"type": "function", "function": '
                          '{"name": "execute"}}\n</tools>\n\nIf you choose')
    assert "</IMPORTANT>\n\nSYS<|im_end|>\n" in got
    assert got.split("</IMPORTANT>\n\nSYS<|im_end|>\n", 1)[1] == (
        "<|im_start|>user\nhi<|im_end|>\n"
        "<|im_start|>assistant\nold<|im_end|>\n"
        "<|im_start|>user\ntask<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nthink\n</think>\n\nok\n\n<tool_call>\n"
        "<function=execute>\n<parameter=command>\nls\n</parameter>\n</function>\n"
        "</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nout\n</tool_response><|im_end|>\n"
    )
    assert render_qwen35(msgs[:4], add_generation_prompt=True).endswith(
        "<|im_start|>assistant\n<think>\n")


def test_mlx_row_rewraps_to_full_conversation(make_trial):
    source, conts = _records(make_trial, [(GOOD, 1.0, "log in first")])
    pair = build_fork_pairs(source, 2, conts)[0][0]
    row = to_mlx_row(pair)
    assert set(row) == {"system", "prompt", "chosen", "rejected"}
    assert row["system"].startswith("# Tools") and row["system"].endswith("You are an agent.")
    assert row["prompt"].startswith("Use the execute tool.")
    assert row["prompt"].endswith("</tool_response>")
    rewrapped = (f"<|im_start|>system\n{row['system']}<|im_end|>\n"
                 f"<|im_start|>user\n{row['prompt']}<|im_end|>\n")
    assert rewrapped == render_qwen35(pair.prompt, pair.tools)
    assert row["chosen"] == (
        "<think>\nlog in first\n</think>\n\ncont\n\n<tool_call>\n<function=execute>\n"
        f"<parameter=command>\n{GOOD}\n</parameter>\n</function>\n</tool_call>")
    full_chosen = render_qwen35(pair.prompt + [pair.chosen], pair.tools)
    assert full_chosen == rewrapped + f"<|im_start|>assistant\n{row['chosen']}<|im_end|>\n"
    assert mlx_split(pair.prompt, pair.tools) == (row["system"], row["prompt"])


def test_pairs_from_plan_and_forks_json(make_trial, tmp_path):
    source, conts = _records(make_trial, [(GOOD, 1.0, "b"), (BAD, 0.0, "c")])
    plan = ReplayPlan(source_trial=source.trial_dir, task_id="aaaaaaa_1", fork_turn=2,
                      initial_context_sha256="", commands=("x",), expected_observations=("",),
                      message_prefix=tuple(PREFIX))
    plan_path = save_plan(plan, tmp_path / "plans" / "p.json")
    pairs, stats = pairs_from_plan(plan_path, tmp_path / "fork-t2", root=tmp_path)
    assert len(pairs) == 1 and stats.continuations == 2

    forks = [{"turn": 2, "k": k, "reward": c.reward, "first_action": None,
              "trial_dir": str(c.trial_dir)} for k, c in enumerate(conts)]
    fj = tmp_path / "results" / "base" / "aaaaaaa_1__src" / "forks.json"
    fj.parent.mkdir(parents=True)
    fj.write_text(json.dumps(forks))
    pairs, stats = pairs_from_forks_json(fj, {"base/aaaaaaa_1__src": source}, root=tmp_path)
    assert len(pairs) == 1 and pairs[0].continuation_trial == "fork-t2/aaaaaaa_1__k0"
    pairs, stats = pairs_from_forks_json(fj, {}, root=tmp_path)
    assert pairs == [] and "not found" in stats.skipped_source[0]


def test_forks_json_eval_only_continuations_are_skipped(make_trial, tmp_path):
    source, conts = _records(make_trial, [(GOOD, 1.0, "b"), (BAD, 0.0, "c")])
    forks = [{"turn": 2, "k": k, "reward": c.reward, "first_action": None,
              "trial_dir": str(c.trial_dir), "reasons": ["judge_claude"], "eval_only": True}
             for k, c in enumerate(conts)]
    fj = tmp_path / "results" / "base" / "aaaaaaa_1__src" / "forks.json"
    fj.parent.mkdir(parents=True)
    fj.write_text(json.dumps(forks))
    pairs, stats = pairs_from_forks_json(fj, {"base/aaaaaaa_1__src": source}, root=tmp_path)
    assert pairs == [] and stats.dropped_eval_only == 2 and stats.continuations == 0
    forks[0]["eval_only"] = False
    forks[0]["reasons"] = ["judge_claude", "bisection"]
    fj.write_text(json.dumps(forks))
    pairs, stats = pairs_from_forks_json(fj, {"base/aaaaaaa_1__src": source}, root=tmp_path)
    assert len(pairs) == 1 and stats.dropped_eval_only == 1
