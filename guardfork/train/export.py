"""Export same-state DPO pairs from fork continuations (Q2 plumbing).

For a failed source trial forked at turn t:
  prompt   = student-visible messages before turn t (system prompt + logged prefix), OpenAI
             chat format via `canonical_messages`, plus the `execute` tool schema;
  chosen   = first action (reasoning + execute call) of a passing continuation;
  rejected = the source run's action at turn t.
Pairs whose AST API signatures collide are dropped; think-only differences are flagged.

Two output formats:
  neutral       {"prompt": [msgs], "tools": [...], "chosen": [assistant], "rejected":
                [assistant], "meta": {...}}  (OpenAI wire messages: string tool-call
                arguments, reasoning in `reasoning_content`; TRL conversational layout)
  mlx-lm-lora   {"system": str, "prompt": str, "chosen": str, "rejected": str}
                Its DPODataset (mlx_lm_lora/trainer/datasets.py) renders
                [system, user(prompt)] + assistant(chosen|rejected) with the tokenizer's chat
                template. We pre-render the multi-turn conversation with the Qwen3.5 template
                and split it so that wrapping reproduces the real token stream exactly
                (scripts/check_dpo_render.py checks this with the real tokenizer).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from guardfork.fork.harbor_replay import ReplayPlan, load_plan
from guardfork.fork.replay_checks import canonical_messages
from guardfork.guards.ast_sig import api_signature
from guardfork.telemetry.trials import TrialRecord, iter_trials, load_trial

EXECUTE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": "Execute one bash command in the task sandbox. Each call uses a fresh "
                       "shell. Use the `agent \"<task>\"` command for optional subagent "
                       "delegation.",
        "parameters": {
            "properties": {"command": {"minLength": 1, "type": "string"}},
            "required": ["command"],
            "type": "object",
        },
    },
}


def logged_tools(trial_dir: Path) -> list[dict[str, Any]]:
    """Tool schema the student was actually sent (first logged request), else EXECUTE_TOOL."""
    path = trial_dir / "agent" / "student_model_requests.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    break
                tools = payload.get("tools") or (payload.get("request") or {}).get("tools")
                if tools:
                    return tools
                break
    return [EXECUTE_TOOL]


def _parts_text(content: Any, kinds: tuple[str, ...]) -> str:
    if isinstance(content, str):
        return content if "text" in kinds else ""
    out = []
    for part in content or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in kinds:
            out.append(str(part.get(kind) if kind != "text" else part.get("text") or ""))
    return "\n".join(p for p in out if p)


def to_openai(message: dict[str, Any]) -> dict[str, Any]:
    """Canonical message (thinking parts) -> OpenAI wire message (`reasoning_content`)."""
    msg = dict(message)
    if msg["role"] in ("assistant", "system", "tool") or isinstance(msg.get("content"), list):
        content = msg.get("content")
        if msg["role"] == "assistant" and isinstance(content, list):
            reasoning = _parts_text(content, ("thinking", "reasoning"))
            if reasoning:
                msg["reasoning_content"] = reasoning
        msg["content"] = _parts_text(content, ("text",))
    return msg


def openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [to_openai(m) for m in canonical_messages(messages)]


def _event_command(event: dict[str, Any]) -> str | None:
    tc = (event.get("original") or {}).get("tool_call") or {}
    return tc.get("command") if isinstance(tc, dict) else None


def action_message(event: dict[str, Any]) -> dict[str, Any]:
    """The assistant message produced at a reviewed_turn event (reasoning + tool call)."""
    original = event.get("original") or {}
    raw = event.get("raw_original") or {}
    tool_calls = list(raw.get("tool_calls") or [])
    command = _event_command(event)
    if not tool_calls and command:
        tool_calls = [{"name": (original.get("tool_call") or {}).get("name") or "execute",
                       "args": {"command": command}, "id": None}]
    reasoning = (original.get("reasoning")
                 or (raw.get("additional_kwargs") or {}).get("reasoning_content") or "")
    lc = {"type": "ai", "content": original.get("content") or raw.get("content") or "",
          "reasoning_content": reasoning, "tool_calls": tool_calls}
    return openai_messages([lc])[0]


def message_command(message: dict[str, Any]) -> str | None:
    for tc in message.get("tool_calls") or []:
        try:
            args = json.loads((tc.get("function") or {}).get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict) and isinstance(args.get("command"), str):
            return args["command"]
    return None


def prompt_messages(event: dict[str, Any],
                    prefix: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None
                    ) -> list[dict[str, Any]]:
    """System prompt + student-visible messages before this event's action."""
    raw = ([event["system_message"]] if event.get("system_message") else [])
    raw += list(prefix if prefix is not None else (event.get("context") or []))
    return openai_messages(raw)


@dataclass(frozen=True)
class DPOPair:
    task_id: str
    source_trial: str
    fork_turn: int
    continuation_trial: str
    prompt: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    chosen: dict[str, Any]
    rejected: dict[str, Any]
    chosen_reward: float | None
    rejected_reward: float | None
    think_only_diff: bool
    arm: str = "GF-DPO"
    signature_collision: bool = False
    identical: bool = False


@dataclass
class ExportStats:
    forks: int = 0
    continuations: int = 0
    passing: int = 0
    pairs: int = 0
    dropped_collision: int = 0
    dropped_identical: int = 0
    dropped_prefix_mismatch: int = 0
    dropped_no_action: int = 0
    dropped_eval_only: int = 0
    skipped_source: list[str] = field(default_factory=list)
    think_only: int = 0

    def add(self, other: ExportStats) -> None:
        for name, value in vars(other).items():
            if isinstance(value, list):
                getattr(self, name).extend(value)
            else:
                setattr(self, name, getattr(self, name) + value)


def _same_prefix(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> bool:
    def strip(msgs):
        return [{k: v for k, v in m.items() if k != "tool_call_id"} for m in msgs]
    return json.dumps(strip(a), sort_keys=True) == json.dumps(strip(b), sort_keys=True)


def build_fork_pairs(source: TrialRecord, fork_turn: int, continuations: list[TrialRecord], *,
                     message_prefix: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None
                     = None, tools: list[dict[str, Any]] | None = None, max_per_fork: int = 1,
                     require_prefix_match: bool = True, arm: str = "GF-DPO",
                     allow_collision: bool = False, allow_identical: bool = False,
                     ) -> tuple[list[DPOPair], ExportStats]:
    """Pairs for one fork state. `allow_collision` / `allow_identical` keep same-signature /
    byte-identical pairs (flagged); plumbing smoke tests only."""
    stats = ExportStats(forks=1, continuations=len(continuations))
    source_id = f"{source.job}/{source.trial_dir.name}"
    if source.reward is None or source.reward >= 1.0:
        stats.skipped_source.append(f"{source_id}: source did not fail")
        return [], stats
    if not 1 <= fork_turn <= len(source.events):
        stats.skipped_source.append(f"{source_id}: no event for turn {fork_turn}")
        return [], stats
    event = source.events[fork_turn - 1]
    prompt = prompt_messages(event, message_prefix)
    rejected = action_message(event)
    tools = tools or logged_tools(source.trial_dir)
    sig_r = api_signature(message_command(rejected))
    pairs: list[DPOPair] = []
    for cont in continuations:
        if cont.reward is None or cont.reward < 1.0:
            continue
        stats.passing += 1
        if len(pairs) >= max_per_fork:
            continue
        if not cont.events:
            stats.dropped_no_action += 1
            continue
        first = cont.events[0]
        if require_prefix_match and not _same_prefix(prompt_messages(first), prompt):
            stats.dropped_prefix_mismatch += 1
            continue
        chosen = action_message(first)
        cmd_c, cmd_r = message_command(chosen), message_command(rejected)
        sig_c = api_signature(cmd_c)
        collision = bool(sig_c) and sig_c == sig_r
        if collision and not allow_collision:
            stats.dropped_collision += 1
            continue
        identical = chosen.get("content") == rejected.get("content") and cmd_c == cmd_r and (
            chosen.get("reasoning_content") == rejected.get("reasoning_content"))
        if identical and not allow_identical:
            stats.dropped_identical += 1
            continue
        think_only = cmd_c == cmd_r and not identical
        stats.think_only += int(think_only)
        pairs.append(DPOPair(
            task_id=source.task_id or "", source_trial=source_id, fork_turn=fork_turn,
            continuation_trial=f"{cont.job}/{cont.trial_dir.name}", prompt=prompt, tools=tools,
            chosen=chosen, rejected=rejected, chosen_reward=cont.reward,
            rejected_reward=source.reward, think_only_diff=think_only, arm=arm,
            signature_collision=collision, identical=identical,
        ))
    stats.pairs = len(pairs)
    return pairs, stats


# --- Qwen3.5 chat-template rendering (mirrors chat_template.jinja of Qwen3.5-*-4bit) ----

QWEN35_TOOL_INSTRUCTIONS = (
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
    "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second"
    " parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner "
    "<function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your "
    "function call in natural language BEFORE the function call, but NOT after\n- If there is "
    "no function call available, answer the question like normal with your current knowledge "
    "and do not tell the user about function calls\n</IMPORTANT>"
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return ""


def _tool_call_xml(tool_calls: list[dict[str, Any]], content: str) -> str:
    out = []
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function") or tc
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        lead = ("\n\n" if content.strip() else "") if i == 0 else "\n"
        out.append(f"{lead}<tool_call>\n<function={fn['name']}>\n")
        for name, value in (args or {}).items():
            text = (json.dumps(value, ensure_ascii=False)
                    if isinstance(value, dict | list) else str(value))
            out.append(f"<parameter={name}>\n{text}\n</parameter>\n")
        out.append("</function>\n</tool_call>")
    return "".join(out)


def _split_reasoning(message: dict[str, Any]) -> tuple[str, str]:
    content = _content_text(message.get("content")).strip()
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = ""
        if "</think>" in content:
            reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            content = content.split("</think>")[-1].lstrip("\n")
    return reasoning.strip(), content


def render_assistant_body(message: dict[str, Any], with_think: bool) -> str:
    """Assistant turn text after '<|im_start|>assistant\\n', up to (not incl.) <|im_end|>."""
    reasoning, content = _split_reasoning(message)
    head = f"<think>\n{reasoning}\n</think>\n\n{content}" if with_think else content
    return head + _tool_call_xml(message.get("tool_calls") or [], content)


def render_qwen35(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
                  add_generation_prompt: bool = False) -> str:
    """Python mirror of the Qwen3.5 chat template for system/user/assistant/tool messages.
    `tools` render like transformers' `tojson` (json.dumps, ensure_ascii=False)."""
    out = []
    first_system = messages and messages[0]["role"] == "system"
    if tools:
        out.append("<|im_start|>system\n# Tools\n\nYou have access to the following functions:"
                   "\n\n<tools>")
        out += ["\n" + json.dumps(t, ensure_ascii=False) for t in tools]
        out.append("\n</tools>" + QWEN35_TOOL_INSTRUCTIONS)
        if first_system:
            sys_text = _content_text(messages[0]["content"]).strip()
            if sys_text:
                out.append("\n\n" + sys_text)
        out.append("<|im_end|>\n")
    elif first_system:
        out.append("<|im_start|>system\n" + _content_text(messages[0]["content"]).strip()
                   + "<|im_end|>\n")
    last_query = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m["role"] == "user":
            text = _content_text(m.get("content")).strip()
            if not (text.startswith("<tool_response>") and text.endswith("</tool_response>")):
                last_query = i
                break
    if last_query is None:
        raise ValueError("No user query found in messages.")
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "system":
            if i != 0:
                raise ValueError("System message must be at the beginning.")
        elif role == "user":
            out.append(f"<|im_start|>user\n{_content_text(m.get('content')).strip()}<|im_end|>\n")
        elif role == "assistant":
            out.append("<|im_start|>assistant\n" + render_assistant_body(m, i > last_query)
                       + "<|im_end|>\n")
        elif role == "tool":
            if i > 0 and messages[i - 1]["role"] != "tool":
                out.append("<|im_start|>user")
            out.append(f"\n<tool_response>\n{_content_text(m.get('content')).strip()}"
                       "\n</tool_response>")
            last = i == len(messages) - 1
            if last or messages[i + 1]["role"] != "tool":
                out.append("<|im_end|>\n")
        else:
            raise ValueError(f"Unexpected message role {role!r}")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n<think>\n")
    return "".join(out)


def assistant_completion(message: dict[str, Any]) -> str:
    """Assistant `content` string that the template re-renders as this exact turn:
    '<think>\\n{reasoning}\\n</think>\\n\\n{content}' + tool-call XML."""
    return render_assistant_body(message, with_think=True)


def mlx_split(prompt: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[str, str]:
    """(system, prompt) such that the template's rendering of
    [system, user(prompt)] equals the rendering of the full conversation `prompt`."""
    full = render_qwen35(prompt, tools)
    sys_open, end, user_open = "<|im_start|>system\n", "<|im_end|>\n", "<|im_start|>user\n"
    if not full.startswith(sys_open):
        raise ValueError("expected a system block (tools or system prompt)")
    sys_end = full.index(end)
    system = full[len(sys_open):sys_end]
    body = full[sys_end + len(end):]
    if not body.startswith(user_open) or not body.endswith(end):
        raise ValueError("conversation must start with a user message and end a turn")
    user = body[len(user_open):-len(end)]
    if user != user.strip() or system != system.strip():
        raise ValueError("leading/trailing whitespace would be trimmed by the template")
    return system, user


def to_mlx_row(pair: DPOPair) -> dict[str, str]:
    system, prompt = mlx_split(pair.prompt, pair.tools)
    return {"system": system, "prompt": prompt,
            "chosen": assistant_completion(pair.chosen),
            "rejected": assistant_completion(pair.rejected)}


def to_neutral_row(pair: DPOPair) -> dict[str, Any]:
    return {
        "prompt": pair.prompt, "tools": pair.tools,
        "chosen": [pair.chosen], "rejected": [pair.rejected],
        "meta": {"task_id": pair.task_id, "arm": pair.arm, "source_trial": pair.source_trial,
                 "fork_turn": pair.fork_turn, "continuation_trial": pair.continuation_trial,
                 "chosen_reward": pair.chosen_reward, "rejected_reward": pair.rejected_reward,
                 "think_only_diff": pair.think_only_diff,
                 "signature_collision": pair.signature_collision,
                 "identical": pair.identical},
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# --- Sources: a ReplayPlan + its fork job, or referee forks.json --------------------------

def _resolve(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def pairs_from_plan(plan: ReplayPlan | Path, fork_job: Path, *, root: Path,
                    **kwargs: Any) -> tuple[list[DPOPair], ExportStats]:
    plan = load_plan(plan) if isinstance(plan, Path) else plan
    source_dir = _resolve(plan.source_trial, root)
    source = load_trial(source_dir / "agent" / "teacher", source_dir.parent.name)
    continuations = list(iter_trials(fork_job, fork_job.name))
    return build_fork_pairs(source, plan.fork_turn, continuations,
                            message_prefix=plan.message_prefix or None, **kwargs)


def pairs_from_forks_json(forks_json: Path, sources: dict[str, TrialRecord], *, root: Path,
                          **kwargs: Any) -> tuple[list[DPOPair], ExportStats]:
    """results/<job>/<trial>/forks.json: [{"turn", "k", "reward", "trial_dir", ...}].
    `sources` maps "<job>/<trial>" to the source TrialRecord. Continuations tagged
    `eval_only: true` by the referee (probed only because of the eval-only `judge_claude`
    pick, AGENTS.md rule 3) are never exported; they are counted in
    `stats.dropped_eval_only`."""
    trial_id = f"{forks_json.parent.parent.name}/{forks_json.parent.name}"
    stats = ExportStats()
    source = sources.get(trial_id)
    if source is None:
        stats.skipped_source.append(f"{trial_id}: source trial not found in --base-jobs")
        return [], stats
    by_turn: dict[int, list[TrialRecord]] = {}
    for entry in json.loads(forks_json.read_text(encoding="utf-8")):
        if entry.get("eval_only"):
            stats.dropped_eval_only += 1
            continue
        trial_dir = _resolve(Path(entry["trial_dir"]), root)
        rec = load_trial(trial_dir / "agent" / "teacher", trial_dir.parent.name)
        by_turn.setdefault(int(entry["turn"]), []).append(rec)
    pairs: list[DPOPair] = []
    for turn, conts in sorted(by_turn.items()):
        got, st = build_fork_pairs(source, turn, conts, **kwargs)
        pairs += got
        stats.add(st)
    return pairs, stats
