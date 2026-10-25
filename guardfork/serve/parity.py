"""Serving vs. training chat-template parity.

Compares vLLM `prompt_token_ids` logged by the proxy against what the training
renderer (`Qwen3_5Renderer(strip_thinking_from_history=False)`) builds for the same
messages. Any mismatch means the served `--chat-template` must change before any
training data is built.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ParityResult:
    record_id: str
    equal: bool
    served_len: int
    rendered_len: int
    first_mismatch: int | None


def first_mismatch(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(record_id: str, served: list[int], rendered: list[int]) -> ParityResult:
    idx = first_mismatch(served, rendered)
    return ParityResult(record_id, idx is None, len(served), len(rendered), idx)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def openai_to_tinker_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI chat messages -> tinker-cookbook `Message` dicts.

    Assistant `reasoning_content` (or `reasoning`) becomes a leading ThinkingPart, tool
    calls become `ToolCall` objects. Imports tinker_cookbook lazily.
    """
    from tinker_cookbook.renderers.base import ToolCall  # type: ignore[import-not-found]

    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        text = _content_text(m.get("content"))
        if role == "assistant":
            reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
            parts: list[dict[str, Any]] = []
            if reasoning:
                parts.append({"type": "thinking", "thinking": reasoning})
            if text:
                parts.append({"type": "text", "text": text})
            msg: dict[str, Any] = {"role": "assistant", "content": parts or ""}
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                calls.append(ToolCall(
                    id=tc.get("id"),
                    function=ToolCall.FunctionBody(
                        name=fn.get("name", ""),
                        arguments=args if isinstance(args, str) else json.dumps(args),
                    ),
                ))
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif role == "tool":
            msg = {"role": "tool", "content": text}
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            if m.get("name"):
                msg["name"] = m["name"]
            out.append(msg)
        else:
            out.append({"role": role, "content": text})
    return out


def build_tinker_renderer(model: str):
    from tinker_cookbook import tokenizer_utils  # type: ignore[import-not-found]
    from tinker_cookbook.renderers.qwen3_5 import Qwen3_5Renderer  # type: ignore[import-not-found]

    tokenizer = tokenizer_utils.get_tokenizer(model)
    try:
        return Qwen3_5Renderer(tokenizer, strip_thinking_from_history=False)
    except TypeError:
        renderer = Qwen3_5Renderer(tokenizer)
        renderer.strip_thinking_from_history = False
        return renderer


def render_with_tinker(renderer, messages: list[dict[str, Any]],
                       tools: list[dict[str, Any]]) -> list[int]:
    msgs = openai_to_tinker_messages(messages)
    if tools:
        system = ""
        if msgs and msgs[0]["role"] == "system":
            system = _content_text(msgs.pop(0)["content"])
        specs = [t.get("function", t) for t in tools]
        msgs = renderer.create_conversation_prefix_with_tools(specs, system) + msgs
    return list(renderer.build_generation_prompt(msgs).to_ints())


def render_with_hf(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
                   ) -> list[int]:
    """Reference rendering with the model's HF chat template (what vLLM uses by default)."""
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tok = AutoTokenizer.from_pretrained(model)
    return list(tok.apply_chat_template(messages, tools=tools or None, tokenize=True,
                                        add_generation_prompt=True))
