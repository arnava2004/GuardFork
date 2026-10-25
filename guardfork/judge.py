"""Zero-rollout LLM-judge localizer: ask an open-weight model which live turn of a failed
run first went irrecoverably wrong.

The prompt template (`config/judge_prompt.md`) is frozen like the guards: its sha256 is
pinned per version in `config/judge_freeze.toml`. Debug the template on dev-set failures
only, freeze it, then run it on fresh failures. The judge sees the task instruction and,
per live turn, the executed command and a truncated observation (no model reasoning).
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from guardfork import REPO_ROOT
from guardfork.telemetry.to_trajectory import message_text, to_trajectory
from guardfork.telemetry.trials import TrialRecord
from guardfork.types import N_DEMO_ASSISTANT_TURNS

JUDGE_VERSION = "v1"
JUDGE_PROMPT_PATH = REPO_ROOT / "config" / "judge_prompt.md"
JUDGE_FREEZE_PATH = REPO_ROOT / "config" / "judge_freeze.toml"
DEFAULT_BASE_URL = "http://127.0.0.1:8200/v1"
DEFAULT_MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
OBS_CHARS = 1500
_PLACEHOLDER = re.compile(r"\{\{(TASK|N_TURNS|TURNS)\}\}")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TURN_RE = re.compile(r'"turn"\s*:\s*"?(-?\d+)')

Call = Callable[[list[dict[str, str]]], str]


def _task_marker() -> str:
    return "[" + "APPWORLD_CURRENT_TASK" + "]"


def prompt_sha256(path: Path = JUDGE_PROMPT_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_template(path: Path = JUDGE_PROMPT_PATH) -> str:
    return path.read_text(encoding="utf-8")


def judge_versions(path: Path = JUDGE_FREEZE_PATH) -> dict[str, dict[str, Any]]:
    return dict(tomllib.loads(path.read_text(encoding="utf-8")).get("versions", {}))


def frozen_version_for(sha: str, path: Path = JUDGE_FREEZE_PATH) -> str | None:
    """Name of the frozen (not draft) version whose sha256 is `sha`, if any."""
    for name, entry in judge_versions(path).items():
        if entry.get("sha256") == sha and entry.get("status") == "frozen":
            return name
    return None


def _is_human(message: dict[str, Any]) -> bool:
    return message.get("type") in ("human", "user") or message.get("role") == "user"


def _is_ai(message: dict[str, Any]) -> bool:
    return message.get("type") in ("ai", "assistant") or message.get("role") == "assistant"


def task_instruction(events: list[dict[str, Any]]) -> str:
    """The task prompt from the first event's context: the user message after the demo
    turns that carries the current-task marker (marker line removed), else the first user
    message after the demo turns."""
    if not events:
        return ""
    context = events[0].get("context") or []
    n_ai, after_demo = 0, []
    for message in context:
        if _is_ai(message):
            n_ai += 1
        elif _is_human(message) and n_ai >= N_DEMO_ASSISTANT_TURNS:
            after_demo.append(message_text(message))
    if not after_demo:
        humans = [message_text(m) for m in context if _is_human(m)]
        after_demo = humans[-1:]
    marker = _task_marker()
    for text in after_demo:
        if marker in text:
            return text.split(marker, 1)[1].strip()
    return after_demo[0].strip() if after_demo else ""


def truncate(text: str, limit: int = OBS_CHARS) -> str:
    """Keep the head and tail (errors and the AppWorld result line sit at the end)."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} chars truncated]...\n{text[-tail:]}"


def turn_blocks(record: TrialRecord, obs_chars: int = OBS_CHARS) -> list[str]:
    blocks = []
    for turn in to_trajectory(record).turns:
        commands = [c.command for c in turn.tool_calls if c.command]
        command = "\n".join(commands) if commands else "(no tool call)"
        obs = turn.observation
        observation = truncate(obs, obs_chars) if obs is not None else "(no observation recorded)"
        blocks.append(f"### Turn {turn.turn}\nCommand:\n{command}\nObservation:\n{observation}")
    return blocks


def build_prompt(record: TrialRecord, template: str, obs_chars: int = OBS_CHARS) -> str:
    values = {
        "TASK": task_instruction(record.events) or "(task instruction not found)",
        "N_TURNS": str(record.n_turns),
        "TURNS": "\n\n".join(turn_blocks(record, obs_chars)),
    }
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def _json_candidates(text: str) -> list[str]:
    """Balanced `{...}` spans, in order of appearance."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"' and depth > 0:
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def _as_turn(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
        return int(value)
    return None


def parse_response(raw: str, n_turns: int) -> tuple[int | None, str | None, str | None]:
    """(turn, reason, error). Strips <think> blocks and code fences, takes the first JSON
    object with a "turn" key, falls back to a `"turn": N` regex. Out-of-range -> None."""
    text = _THINK.sub("", raw or "")
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0]
    text = text.replace("</think>", "")
    spans = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    turn: int | None = None
    reason: str | None = None
    found = False
    for span in spans:
        for cand in _json_candidates(span):
            try:
                obj = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "turn" in obj:
                turn = _as_turn(obj.get("turn"))
                r = obj.get("reason")
                reason = r if isinstance(r, str) else (None if r is None else str(r))
                found = True
                break
        if found:
            break
    if not found:
        m = _TURN_RE.search(text)
        if not m:
            return None, None, "no JSON with a turn field"
        turn = int(m.group(1))
    if turn is None:
        return None, reason, "turn is not an integer"
    if not 1 <= turn <= n_turns:
        return None, reason, f"turn {turn} out of range 1..{n_turns}"
    return turn, reason, None


def make_openai_call(base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, *,
                     temperature: float = 0.0, max_tokens: int = 8192,
                     timeout: float = 900.0, api_key: str = "local",
                     extra_body: dict[str, Any] | None = None) -> Call:
    """POST /chat/completions on an OpenAI-compatible server (stdlib only). Returns the
    message content, falling back to reasoning fields when content is empty."""
    url = base_url.rstrip("/") + "/chat/completions"

    def call(messages: list[dict[str, str]]) -> str:
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "stream": False, **(extra_body or {})}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        message = (data.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content") or ""
        if not content.strip():
            content = message.get("reasoning_content") or message.get("reasoning") or ""
        return content

    return call


def judge_trial(record: TrialRecord, call: Call, *, model: str, template: str,
                template_sha256: str, obs_chars: int = OBS_CHARS) -> dict[str, Any]:
    prompt = build_prompt(record, template, obs_chars)
    try:
        raw = call([{"role": "user", "content": prompt}])
        turn, reason, error = parse_response(raw, record.n_turns)
    except OSError as err:  # urllib errors, timeouts
        raw, turn, reason, error = "", None, None, f"request failed: {err}"
    return {
        "trial_id": f"{record.job}/{record.trial_dir.name}",
        "task_id": record.task_id,
        "T": record.n_turns,
        "turn": turn,
        "reason": reason,
        "error": error,
        "model": model,
        "judge_version": JUDGE_VERSION,
        "prompt_sha256": template_sha256,
        "filled_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "raw": raw,
    }


def judge_path(results_root: Path, job: str, trial: str) -> Path:
    return results_root / job / trial / "judge.json"


def write_judge(result: dict[str, Any], results_root: Path) -> Path:
    job, trial = result["trial_id"].split("/", 1)
    path = judge_path(results_root, job, trial)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_judge_results(results_root: Path) -> dict[str, dict[str, Any]]:
    """trial_id ("<job>/<trial>") -> judge.json contents, for every judge.json under root."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(results_root.glob("*/*/judge.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out[data.get("trial_id") or f"{path.parent.parent.name}/{path.parent.name}"] = data
    return out
