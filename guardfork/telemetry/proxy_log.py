"""Read `guardfork.serve.proxy` JSONL logs and join them to Harbor trials.

The proxy cannot see Harbor's trial name (the agent calls it as a plain OpenAI endpoint), so
records are joined offline: every chat completion carries a unique response id
(`chatcmpl-...`), which the proxy logs and Harness-Zero stores per turn in the trial's
`trajectory.jsonl` (`raw_original.response_metadata.id`). `trial_response_ids` reads those
ids; `records_for_trial` / `index_by_trial` return the matching records keyed by the
referee's trial id `<job>/<trial_dir>`."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
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

    @property
    def response_id(self) -> str | None:
        """Completion id; logged at top level since proxy v2, read from the response before."""
        return self.raw.get("response_id") or (self.raw.get("response") or {}).get("id")


def iter_proxy_log(path: Path) -> Iterator[ProxyRecord]:
    files = sorted(path.glob("**/*.jsonl")) if path.is_dir() else [path]
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    yield ProxyRecord(json.loads(line))
                except json.JSONDecodeError:
                    continue


def trial_response_ids(events: Iterable[dict[str, Any]]) -> list[str]:
    """Completion ids of a trial's live turns, in order, from its `reviewed_turn` events."""
    ids = []
    for e in events:
        if e.get("type") != "reviewed_turn":
            continue
        rid = ((e.get("raw_original") or {}).get("response_metadata") or {}).get("id")
        if rid:
            ids.append(rid)
    return ids


def records_for_trial(records: Iterable[ProxyRecord], events: Iterable[dict[str, Any]],
                      ) -> list[ProxyRecord]:
    """Proxy records whose response id belongs to the trial, in the trial's turn order."""
    by_id = {r.response_id: r for r in records if r.response_id}
    return [by_id[rid] for rid in trial_response_ids(events) if rid in by_id]


def index_by_trial(records: Iterable[ProxyRecord], trials: Iterable[Any],
                   ) -> dict[str, list[ProxyRecord]]:
    """`<job>/<trial_dir>` -> that trial's proxy records, for `TrialRecord`-like objects
    (`job`, `trial_dir`, `events`). Trials with no matching record are omitted."""
    records = list(records)
    out = {}
    for t in trials:
        matched = records_for_trial(records, t.events)
        if matched:
            out[f"{t.job}/{t.trial_dir.name}"] = matched
    return out
