"""Read `guardfork.serve.proxy` JSONL logs."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

N_DEMO_ASSISTANT_TURNS = 10


@dataclass
class ProxyRecord:
    raw: dict[str, Any]

    @property
    def messages(self) -> list[dict[str, Any]]:
        return (self.raw.get("request") or {}).get("messages") or []

    @property
    def tools(self) -> list[dict[str, Any]]:
        return (self.raw.get("request") or {}).get("tools") or []

    @property
    def prompt_token_ids(self) -> list[int] | None:
        return self.raw.get("prompt_token_ids")

    @property
    def prompt_tokens(self) -> int | None:
        if self.prompt_token_ids is not None:
            return len(self.prompt_token_ids)
        usage = self.raw.get("usage") or {}
        value = usage.get("prompt_tokens")
        return int(value) if value is not None else None

    @property
    def completion_tokens(self) -> int | None:
        value = (self.raw.get("usage") or {}).get("completion_tokens")
        return int(value) if value is not None else None

    @property
    def live_turn_index(self) -> int:
        """0-based index of the turn being generated, excluding the demo turns."""
        n_assistant = sum(1 for m in self.messages if m.get("role") == "assistant")
        return n_assistant - N_DEMO_ASSISTANT_TURNS

    @property
    def ended_with_tool_call(self) -> bool:
        choices = (self.raw.get("response") or {}).get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            return False
        message = choices[0].get("message") or {}
        return bool(message.get("tool_calls")) or choices[0].get("finish_reason") == "tool_calls"

    @property
    def ok(self) -> bool:
        return self.raw.get("status") == 200


def iter_proxy_log(path: Path) -> Iterator[ProxyRecord]:
    files = sorted(path.glob("**/*.jsonl")) if path.is_dir() else [path]
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    yield ProxyRecord(json.loads(line))
                except json.JSONDecodeError:
                    continue
