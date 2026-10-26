"""Zero-rollout LLM-judge localizers: ask a model which live turn of a failed run first went
irrecoverably wrong, as a probability distribution over turns.

Judge slots live in `config/judges.toml` (`judge_open` hosted open-weight primary,
`judge_local` on the Mac, `judge_claude` eval-only frontier reference; AGENTS.md rule 3).
Every slot gets the identical prompt template (`config/judge_prompt.md`, sha256 pinned per
version in `config/judge_freeze.toml`) with the identical, logged transcript truncation
(`prepare_prompt`). The judge sees the task instruction and, per live turn, the executed
command and a truncated observation (no model reasoning). API keys come from environment
variables only (`api_key_env`); nothing here ever reads a key from a file or an argument.

Output schema (judge-v1): `{"turns": [{"turn": k, "p": float, "reason": str}, ...]}` with up
to three entries; `parse_distribution` turns it into a vector over 1..T (rules documented
there). Legacy `{"turn": k, "reason": ...}` answers still parse as a one-hot distribution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from guardfork import REPO_ROOT
from guardfork.telemetry.to_trajectory import message_text, to_trajectory
from guardfork.telemetry.trials import TrialRecord
from guardfork.types import N_DEMO_ASSISTANT_TURNS

JUDGE_VERSION = "v1"
JUDGE_PROMPT_PATH = REPO_ROOT / "config" / "judge_prompt.md"
JUDGE_FREEZE_PATH = REPO_ROOT / "config" / "judge_freeze.toml"
JUDGES_PATH = REPO_ROOT / "config" / "judges.toml"
DEFAULT_BASE_URL = "http://127.0.0.1:8200/v1"
DEFAULT_MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
OBS_CHARS = 1500
MAX_PROMPT_CHARS = 60_000
KEEP_HEAD_TURNS = 3  # live turns always kept at the start when eliding
KEEP_TAIL_TURNS = 5  # live turns always kept at the end when eliding
MAX_DIST_ENTRIES = 3
JUDGE_SLOTS = ("judge_open", "judge_local", "judge_claude")
LEGACY_JUDGE_ID = "judge"  # results file `judge.json` (pre-slot layout)
PROVIDERS = ("openai_compat", "anthropic")
RETRY_STATUS = frozenset({429, 500, 502, 503, 529})
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_S = (2.0, 4.0, 8.0)  # sleep after attempt i (only the first ATTEMPTS-1 used)
_PLACEHOLDER = re.compile(r"\{\{(TASK|N_TURNS|TURNS)\}\}")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TURN_RE = re.compile(r'"turn"\s*:\s*"?(-?\d+)')

# A call takes chat messages and returns the text, or `(text, meta)` for provider calls.
Call = Callable[[list[dict[str, str]]], str | tuple[str, dict[str, Any]]]

_sleep = time.sleep  # patched in tests


def _task_marker() -> str:
    return "[" + "APPWORLD_CURRENT_TASK" + "]"


# ---------------------------------------------------------------------------- template / freeze


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


# ---------------------------------------------------------------------------- judge slots


@dataclass(frozen=True)
class JudgeSpec:
    """One `[judge_<id>]` table of `config/judges.toml`, merged with `[common]`."""

    judge_id: str
    provider: str
    model: str
    base_url: str = ""
    api_key_env: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 900.0
    usd_per_m_input: float = 0.0
    usd_per_m_output: float = 0.0
    eval_only: bool = False
    extra_body: dict[str, Any] | None = None
    obs_chars: int = OBS_CHARS
    max_prompt_chars: int = MAX_PROMPT_CHARS

    @property
    def sampling(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "max_tokens": self.max_tokens}


def load_judges(path: Path = JUDGES_PATH) -> dict[str, JudgeSpec]:
    """judge_id -> JudgeSpec for every non-`common` table in `config/judges.toml`."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    common = data.get("common") or {}
    out: dict[str, JudgeSpec] = {}
    for name, table in data.items():
        if name == "common" or not isinstance(table, dict):
            continue
        provider = table.get("provider")
        if provider not in PROVIDERS:
            raise ValueError(f"[{name}] provider must be one of {PROVIDERS}, got {provider!r}")
        if not table.get("model"):
            raise ValueError(f"[{name}] needs a model")
        extra = table.get("extra_body")
        out[name] = JudgeSpec(
            judge_id=name, provider=provider, model=str(table["model"]),
            base_url=str(table.get("base_url", "")),
            api_key_env=str(table.get("api_key_env", "")),
            temperature=float(table.get("temperature", 0.0)),
            max_tokens=int(table.get("max_tokens", 4096)),
            timeout=float(table.get("timeout", common.get("timeout", 900.0))),
            usd_per_m_input=float(table.get("usd_per_m_input", 0.0)),
            usd_per_m_output=float(table.get("usd_per_m_output", 0.0)),
            eval_only=bool(table.get("eval_only", False)),
            extra_body=dict(extra) if isinstance(extra, dict) and extra else None,
            obs_chars=int(table.get("obs_chars", common.get("obs_chars", OBS_CHARS))),
            max_prompt_chars=int(table.get("max_prompt_chars",
                                           common.get("max_prompt_chars", MAX_PROMPT_CHARS))),
        )
    return out


def make_call(spec: JudgeSpec, *, api_key_env: str | None = None,
              timeout: float | None = None) -> Call:
    """Provider call for a judge slot. `api_key_env` overrides the slot's env-var name."""
    env = spec.api_key_env if api_key_env is None else api_key_env
    timeout = spec.timeout if timeout is None else timeout
    if spec.provider == "anthropic":
        return make_anthropic_call(spec.model, api_key_env=env or "ANTHROPIC_API_KEY",
                                   base_url=spec.base_url or ANTHROPIC_BASE_URL,
                                   temperature=spec.temperature, max_tokens=spec.max_tokens,
                                   timeout=timeout)
    return make_openai_call(spec.base_url or DEFAULT_BASE_URL, spec.model,
                            api_key_env=env or None, temperature=spec.temperature,
                            max_tokens=spec.max_tokens, timeout=timeout,
                            extra_body=spec.extra_body)


# ---------------------------------------------------------------------------- prompt


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


def numbered_turn_blocks(record: TrialRecord, obs_chars: int = OBS_CHARS,
                         ) -> list[tuple[int, str]]:
    blocks = []
    for turn in to_trajectory(record).turns:
        commands = [c.command for c in turn.tool_calls if c.command]
        command = "\n".join(commands) if commands else "(no tool call)"
        obs = turn.observation
        observation = truncate(obs, obs_chars) if obs is not None else "(no observation recorded)"
        blocks.append((turn.turn,
                       f"### Turn {turn.turn}\nCommand:\n{command}\nObservation:\n{observation}"))
    return blocks


def turn_blocks(record: TrialRecord, obs_chars: int = OBS_CHARS) -> list[str]:
    return [block for _, block in numbered_turn_blocks(record, obs_chars)]


def _fill(template: str, task: str, n_turns: int, blocks: list[str]) -> str:
    values = {"TASK": task, "N_TURNS": str(n_turns), "TURNS": "\n\n".join(blocks)}
    return _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def elision_line(first: int, last: int) -> str:
    return f"### Turns {first}–{last} elided ({last - first + 1} turns)"


def prepare_prompt(record: TrialRecord, template: str, *, obs_chars: int = OBS_CHARS,
                   max_prompt_chars: int | None = MAX_PROMPT_CHARS,
                   ) -> tuple[str, dict[str, Any]]:
    """Fill the template and return `(prompt, truncation)`.

    Per-observation truncation is `truncate(obs, obs_chars)`. When the filled prompt still
    exceeds `max_prompt_chars`, whole middle turns are elided: the first `KEEP_HEAD_TURNS`
    and last `KEEP_TAIL_TURNS` live turns are always kept, and the oldest middle turns are
    replaced, one more at a time, by a single `### Turns a–b elided (n turns)` line until
    the prompt fits (or only the head/tail remain). Deterministic in the record and the
    parameters. `truncation` logs `obs_chars`, `max_prompt_chars`, `elided_turns`,
    `prompt_chars`, `prompt_tokens_est` (chars / 4) and `fits`.
    """
    task = task_instruction(record.events) or "(task instruction not found)"
    blocks = numbered_turn_blocks(record, obs_chars)
    prompt = _fill(template, task, record.n_turns, [b for _, b in blocks])
    elided: list[int] = []
    n = len(blocks)
    if (max_prompt_chars is not None and len(prompt) > max_prompt_chars
            and n > KEEP_HEAD_TURNS + KEEP_TAIL_TURNS):
        start = KEEP_HEAD_TURNS
        for end in range(start, n - KEEP_TAIL_TURNS):
            elided = [t for t, _ in blocks[start:end + 1]]
            kept = ([b for _, b in blocks[:start]]
                    + [elision_line(elided[0], elided[-1])]
                    + [b for _, b in blocks[end + 1:]])
            prompt = _fill(template, task, record.n_turns, kept)
            if len(prompt) <= max_prompt_chars:
                break
    truncation = {
        "obs_chars": obs_chars,
        "max_prompt_chars": max_prompt_chars,
        "elided_turns": elided,
        "prompt_chars": len(prompt),
        "prompt_tokens_est": len(prompt) // 4,
        "fits": max_prompt_chars is None or len(prompt) <= max_prompt_chars,
    }
    return prompt, truncation


def build_prompt(record: TrialRecord, template: str, obs_chars: int = OBS_CHARS, *,
                 max_prompt_chars: int | None = None) -> str:
    """Prompt text only (see `prepare_prompt`; no elision unless `max_prompt_chars`)."""
    return prepare_prompt(record, template, obs_chars=obs_chars,
                          max_prompt_chars=max_prompt_chars)[0]


# ---------------------------------------------------------------------------- parsing


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


def _as_prob(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        p = float(value)
    elif isinstance(value, str):
        try:
            p = float(value.strip().rstrip("%"))
        except ValueError:
            return None
        if value.strip().endswith("%"):
            p /= 100.0
    else:
        return None
    return p if p >= 0.0 and not math.isnan(p) else None


def _clean(raw: str) -> str:
    text = _THINK.sub("", raw or "")
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0]
    return text.replace("</think>", "")


def _first_object(text: str, keys: tuple[str, ...]) -> dict[str, Any] | None:
    """First JSON object (inside a code fence first, then anywhere) carrying one of `keys`."""
    spans = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for span in spans:
        for cand in _json_candidates(span):
            try:
                obj = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and any(k in obj for k in keys):
                return obj
    return None


def parse_response(raw: str, n_turns: int) -> tuple[int | None, str | None, str | None]:
    """Legacy point parser: `(turn, reason, error)`. Strips <think> blocks and code fences,
    takes the first JSON object with a "turn" key, falls back to a `"turn": N` regex, then
    to the argmax of a `"turns"` distribution. Out-of-range -> None."""
    text = _clean(raw)
    obj = _first_object(text, ("turn",))
    turn: int | None = None
    reason: str | None = None
    if obj is not None:
        turn = _as_turn(obj.get("turn"))
        r = obj.get("reason")
        reason = r if isinstance(r, str) else (None if r is None else str(r))
    elif _first_object(text, ("turns",)) is not None:
        _, top, reasons, _ = parse_distribution(raw, n_turns)
        if top is None:
            return None, None, "no JSON with a turn field"
        return top, reasons.get(top), None
    else:
        m = _TURN_RE.search(text)
        if not m:
            return None, None, "no JSON with a turn field"
        turn = int(m.group(1))
    if turn is None:
        return None, reason, "turn is not an integer"
    if not 1 <= turn <= n_turns:
        return None, reason, f"turn {turn} out of range 1..{n_turns}"
    return turn, reason, None


def parse_distribution(raw: str, n_turns: int,
                       ) -> tuple[list[float], int | None, dict[int, str], str | None]:
    """Parse a judge answer into `(dist, top_turn, reasons, error)`.

    `dist` has length `n_turns` and sums to 1 (uniform when `error` is set); `top_turn` is
    the argmax (ties -> earliest turn) or None on error; `reasons` maps in-range turns to
    the stated reason. Accepts `{"turns": [{"turn", "p", "reason"}, ...]}` (first
    `MAX_DIST_ENTRIES` entries) or the legacy `{"turn": k, "reason": ...}` (one-hot).

    Mass rules. Entries whose turn is outside 1..`n_turns` are dropped; duplicate turns
    are merged by summing `p`; when no entry states a `p`, the listed turns share the mass
    equally. Let `s` be the total `p` stated over all parsed entries (including dropped
    ones). If `s >= 1` the judge gave a full distribution: the in-range entries are
    renormalised to sum to 1 (no leftover is spread). If `0 < s < 1` the judge explicitly
    left `1 - s` unassigned: the in-range entries keep their stated `p` and the leftover
    (`1 - sum(in-range p)`, so dropped mass is also leftover) is spread uniformly over the
    turns that were not listed; if every turn was listed, the entries are renormalised.
    `s == 0` is an error.
    """
    uniform = [1.0 / n_turns] * n_turns if n_turns > 0 else []
    if n_turns <= 0:
        return uniform, None, {}, "n_turns must be positive"
    text = _clean(raw)
    obj = _first_object(text, ("turns", "turn"))
    if obj is None:
        m = _TURN_RE.search(text)
        if not m:
            return uniform, None, {}, "no JSON with a turn/turns field"
        obj = {"turn": int(m.group(1))}
    if isinstance(obj.get("turns"), list):
        entries: list[Any] = list(obj["turns"])[:MAX_DIST_ENTRIES]
    else:
        entries = [{"turn": obj.get("turn"), "p": 1.0, "reason": obj.get("reason")}]
    mass: dict[int, float] = {}
    unweighted: list[int] = []
    reasons: dict[int, str] = {}
    stated = 0.0
    dropped: list[int] = []
    bad_turn = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        t = _as_turn(entry.get("turn"))
        if t is None:
            bad_turn = True
            continue
        p = _as_prob(entry.get("p"))
        if p is not None:
            stated += p
        if not 1 <= t <= n_turns:
            dropped.append(t)
            continue
        if p is None:
            unweighted.append(t)
            mass.setdefault(t, 0.0)
        else:
            mass[t] = mass.get(t, 0.0) + p
        r = entry.get("reason")
        if isinstance(r, str) and r and t not in reasons:
            reasons[t] = r
    if not mass:
        if dropped:
            return uniform, None, {}, f"turn {dropped[0]} out of range 1..{n_turns}"
        if bad_turn:
            return uniform, None, {}, "turn is not an integer"
        return uniform, None, {}, "no turn entries"
    if stated <= 0.0:
        if unweighted and len(unweighted) == len(mass):
            for t in mass:
                mass[t] = 1.0 / len(mass)
            stated = 1.0
        else:
            return uniform, None, reasons, "stated probabilities sum to zero"
    s_in = sum(mass.values())
    dist = [0.0] * n_turns
    unlisted = [t for t in range(1, n_turns + 1) if t not in mass]
    if stated >= 1.0 - 1e-9 or not unlisted or s_in <= 0.0:
        if s_in <= 0.0:
            return uniform, None, reasons, "in-range probabilities sum to zero"
        for t, p in mass.items():
            dist[t - 1] = p / s_in
    else:
        leftover = max(0.0, 1.0 - s_in)
        for t, p in mass.items():
            dist[t - 1] = p
        for t in unlisted:
            dist[t - 1] = leftover / len(unlisted)
    z = sum(dist)
    dist = [p / z for p in dist]
    top = min(range(n_turns), key=lambda i: (-dist[i], i)) + 1
    return dist, top, reasons, None


def top_k(dist: list[float] | None, k: int = 3) -> list[tuple[int, float]]:
    """`[(turn, p), ...]` for the `k` most likely turns (ties -> earliest), p > 0 only."""
    if not dist:
        return []
    order = sorted(range(len(dist)), key=lambda i: (-dist[i], i))
    return [(i + 1, dist[i]) for i in order[:k] if dist[i] > 0.0]


# ---------------------------------------------------------------------------- providers


def _env_key(api_key_env: str | None) -> str | None:
    """Read the API key from the named environment variable; None when no name is given."""
    if not api_key_env:
        return None
    key = os.environ.get(api_key_env, "")
    if not key.strip():
        raise RuntimeError(
            f"environment variable {api_key_env} is not set; API keys come from the "
            f"environment only (--api-key-env / config/judges.toml api_key_env)")
    return key.strip()


def _post_json(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float,
               ) -> dict[str, Any]:
    """POST JSON with retry: `RETRY_ATTEMPTS` tries, exponential backoff `RETRY_BACKOFF_S`
    on HTTP 429/500/502/503/529, URLError and timeouts; 4xx such as 400/401/403 raise at
    once. Re-raises the last error after the final attempt."""
    data = json.dumps(body).encode()
    last: Exception | None = None
    for attempt in range(RETRY_ATTEMPTS):
        req = urllib.request.Request(url, data=data, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as err:
            if err.code not in RETRY_STATUS:
                raise
            last = err
        except (urllib.error.URLError, TimeoutError) as err:
            last = err
        if attempt + 1 < RETRY_ATTEMPTS:
            _sleep(RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)])
    assert last is not None
    raise last


def make_openai_call(base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL, *,
                     api_key_env: str | None = None, temperature: float = 0.0,
                     max_tokens: int = 8192, timeout: float = 900.0,
                     extra_body: dict[str, Any] | None = None) -> Call:
    """POST /chat/completions on an OpenAI-compatible server (OpenRouter, vLLM, mlx_lm;
    stdlib only). Returns `(text, meta)`; text falls back to reasoning fields when the
    content is empty. With no `api_key_env` the bearer token is the placeholder `local`.
    A named but unset env var raises `RuntimeError` here, before any request."""
    key = _env_key(api_key_env) or "local"
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}

    def call(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "stream": False, **(extra_body or {})}
        data = _post_json(url, body, headers, timeout)
        message = (data.get("choices") or [{}])[0].get("message") or {}
        content = message.get("content") or ""
        if not content.strip():
            content = message.get("reasoning_content") or message.get("reasoning") or ""
        meta = {"provider": "openai_compat", "model_returned": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"),
                "response_id": data.get("id"), "usage": data.get("usage")}
        return content, meta

    return call


def make_anthropic_call(model: str, *, api_key_env: str = "ANTHROPIC_API_KEY",
                        base_url: str = ANTHROPIC_BASE_URL, temperature: float = 0.0,
                        max_tokens: int = 4096, timeout: float = 900.0) -> Call:
    """POST /v1/messages (Anthropic Messages API, stdlib only). Returns `(text, meta)`
    where text joins the `text` content blocks. Eval-only judge rows (AGENTS.md rule 3).
    An unset env var raises `RuntimeError` here, before any request."""
    key = _env_key(api_key_env)
    if key is None:
        raise RuntimeError("make_anthropic_call needs api_key_env (e.g. ANTHROPIC_API_KEY)")
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
               "content-type": "application/json"}

    def call(messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")}
                             for m in messages]}
        data = _post_json(url, body, headers, timeout)
        blocks = data.get("content") or []
        text = "\n".join(b.get("text", "") for b in blocks
                         if isinstance(b, dict) and b.get("type") == "text")
        meta = {"provider": "anthropic", "model_returned": data.get("model"),
                "response_id": data.get("id"), "stop_reason": data.get("stop_reason"),
                "usage": data.get("usage")}
        return text, meta

    return call


def estimate_usd(usage: dict[str, Any] | None, usd_per_m_input: float,
                 usd_per_m_output: float) -> float:
    """Cost estimate from a usage block (OpenAI `prompt_tokens`/`completion_tokens` or
    Anthropic `input_tokens`/`output_tokens`); 0.0 without usage."""
    if not isinstance(usage, dict):
        return 0.0

    def _tok(*names: str) -> int:
        for name in names:
            v = usage.get(name)
            if isinstance(v, int | float) and not isinstance(v, bool):
                return int(v)
        return 0

    inp = _tok("prompt_tokens", "input_tokens")
    out = _tok("completion_tokens", "output_tokens")
    return (inp * usd_per_m_input + out * usd_per_m_output) / 1_000_000


# ---------------------------------------------------------------------------- judging


def judge_trial(record: TrialRecord, call: Call, *, model: str, template: str,
                template_sha256: str, judge_id: str = LEGACY_JUDGE_ID,
                provider: str = "openai_compat", obs_chars: int = OBS_CHARS,
                max_prompt_chars: int | None = MAX_PROMPT_CHARS,
                sampling: dict[str, Any] | None = None, eval_only: bool = False,
                ) -> dict[str, Any]:
    """One judge call on one failed trial -> the judge record (see module docstring).
    Request failures (`OSError`, incl. HTTP errors after retries) are recorded in `error`
    rather than raised; `distribution` is None in that case."""
    prompt, truncation = prepare_prompt(record, template, obs_chars=obs_chars,
                                        max_prompt_chars=max_prompt_chars)
    raw, meta = "", {}
    dist: list[float] | None
    try:
        out = call([{"role": "user", "content": prompt}])
        raw, meta = out if isinstance(out, tuple) else (out, {})
        dist, top, reasons, error = parse_distribution(raw, record.n_turns)
    except OSError as err:  # urllib errors, timeouts
        dist, top, reasons, error = None, None, {}, f"request failed: {err}"
    return {
        "trial_id": f"{record.job}/{record.trial_dir.name}",
        "task_id": record.task_id,
        "T": record.n_turns,
        "judge_id": judge_id,
        "provider": meta.get("provider") or provider,
        "model": model,
        "model_requested": model,
        "model_returned": meta.get("model_returned"),
        "response_id": meta.get("response_id"),
        "system_fingerprint": meta.get("system_fingerprint"),
        "stop_reason": meta.get("stop_reason"),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "sampling": dict(sampling or {}),
        "usage": meta.get("usage"),
        "truncation": truncation,
        "judge_version": JUDGE_VERSION,
        "prompt_sha256": template_sha256,
        "filled_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "distribution": dist,
        "turn": top,
        "reason": reasons.get(top) if top is not None else None,
        "reasons": {str(t): r for t, r in sorted(reasons.items())},
        "eval_only": eval_only,
        "error": error,
        "raw": raw,
    }


def judge_filename(judge_id: str = LEGACY_JUDGE_ID) -> str:
    """`judge_<id>.json` (slot ids already start with `judge_`, so `judge_open` ->
    `judge_open.json`); the legacy id `judge` keeps `judge.json`."""
    if judge_id == LEGACY_JUDGE_ID:
        return "judge.json"
    return f"{judge_id}.json" if judge_id.startswith("judge_") else f"judge_{judge_id}.json"


def judge_path(results_root: Path, job: str, trial: str,
               judge_id: str = LEGACY_JUDGE_ID) -> Path:
    """`results/<job>/<trial>/<judge_filename(judge_id)>`."""
    return results_root / job / trial / judge_filename(judge_id)


def write_judge(result: dict[str, Any], results_root: Path) -> Path:
    job, trial = result["trial_id"].split("/", 1)
    path = judge_path(results_root, job, trial, result.get("judge_id") or LEGACY_JUDGE_ID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_judge_results(results_root: Path, judge_id: str = LEGACY_JUDGE_ID,
                       ) -> dict[str, dict[str, Any]]:
    """trial_id ("<job>/<trial>") -> record, for every `judge_filename(judge_id)` under
    root."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(results_root.glob(f"*/*/{judge_filename(judge_id)}")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out[data.get("trial_id") or f"{path.parent.parent.name}/{path.parent.name}"] = data
    return out
