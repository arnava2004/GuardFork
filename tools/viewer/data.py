"""Read-only data layer for the trajectory viewer.

Sources (all optional except jobs):
- jobs/<job>/<trial>/...                  Harbor trials (see guardfork.telemetry.trials)
- results/<job>/<trial>/oracle.json      {"K": 8, "values": {"<turn>": [passes, graded]},
                                           "t_star": int | null, "t_star_drop": int | null}
- results/<job>/<trial>/forks.json       [{"turn": int, "k": int, "reward": float | null,
                                            "first_action": str | null}]
- runs/proxy/*.jsonl                     proxy log, joined to turns by exact message text
"""

from __future__ import annotations

import hashlib
import json
from functools import cached_property
from pathlib import Path
from typing import Any

from guardfork.fork.replay_checks import canonical_messages
from guardfork.telemetry.proxy_log import ProxyRecord, iter_proxy_log
from guardfork.telemetry.to_trajectory import message_text, to_trajectory
from guardfork.telemetry.trials import TrialRecord, iter_trials

WINDOWS = (0, 1, 2, 3)


def _conversation_key(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(canonical_messages(messages), ensure_ascii=False, sort_keys=True)
    payload = payload.replace("[REDACTED]", "dummy")
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _event_key(event: dict[str, Any]) -> str:
    msgs = ([event["system_message"]] if event.get("system_message") else []) + list(
        event.get("context") or [])
    return _conversation_key(msgs)


def _request_key(rec: ProxyRecord) -> str:
    return _conversation_key(rec.messages)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _guards(record: TrialRecord) -> tuple[bool, list[dict[str, Any]], int | None]:
    """(available, firings, first non-terminal turn). Unavailable until rules.py exists."""
    try:
        from guardfork.guards import rules

        traj = to_trajectory(record)
        firings = rules.evaluate(traj)
        first = rules.first_nonterminal(firings, traj.T)
    except NotImplementedError:
        return False, [], None
    return True, [{"guard": str(f.guard), "turn": f.turn, "evidence": f.evidence}
                  for f in firings], (first.turn if first else None)


class RunIndex:
    def __init__(self, jobs_root: Path, results_root: Path | None = None,
                 proxy_log: Path | None = None) -> None:
        self.jobs_root = jobs_root
        self.results_root = results_root
        self.proxy_log = proxy_log

    @cached_property
    def records(self) -> dict[str, TrialRecord]:
        out: dict[str, TrialRecord] = {}
        if not self.jobs_root.is_dir():
            return out
        for job_dir in sorted(p for p in self.jobs_root.iterdir() if p.is_dir()):
            for rec in iter_trials(job_dir, job_dir.name):
                out[f"{rec.job}/{rec.trial_dir.name}"] = rec
        return out

    @cached_property
    def proxy(self) -> dict[str, ProxyRecord]:
        if not self.proxy_log or not self.proxy_log.exists():
            return {}
        return {_request_key(r): r for r in iter_proxy_log(self.proxy_log) if r.ok}

    def _results_dir(self, run_id: str) -> Path | None:
        return self.results_root / run_id if self.results_root else None

    def runs(self) -> list[dict[str, Any]]:
        return [{
            "id": rid, "job": r.job, "task_id": r.task_id, "reward": r.reward,
            "n_turns": r.n_turns, "has_oracle": self._oracle(rid) is not None,
        } for rid, r in self.records.items()]

    def _oracle(self, run_id: str) -> dict[str, Any] | None:
        d = self._results_dir(run_id)
        return _read_json(d / "oracle.json") if d else None

    def _forks(self, run_id: str) -> list[dict[str, Any]]:
        d = self._results_dir(run_id)
        return (_read_json(d / "forks.json") or []) if d else []

    def detail(self, run_id: str) -> dict[str, Any] | None:
        rec = self.records.get(run_id)
        if rec is None:
            return None
        traj = to_trajectory(rec)
        available, firings, first_nt = _guards(rec)
        by_turn: dict[int, list[dict[str, Any]]] = {}
        for f in firings:
            by_turn.setdefault(f["turn"], []).append(f)
        forks = self._forks(run_id)
        turns = []
        for i, t in enumerate(traj.turns):
            proxy = self.proxy.get(_event_key(rec.events[i])) if i < len(rec.events) else None
            tr = rec.turns[i]
            turns.append({
                "turn": t.turn,
                "content": t.content,
                "reasoning": t.reasoning,
                "command": t.command,
                "n_tool_calls": len(t.tool_calls),
                "observation": t.observation,
                "parse_reasons": tr.parse_reasons,
                "subagent_call": tr.subagent_call,
                "session_reset": tr.session_reset,
                "prompt_tokens": proxy.prompt_tokens if proxy else None,
                "completion_tokens": proxy.completion_tokens if proxy else None,
                "latency_s": proxy.raw.get("latency_s") if proxy else None,
                "guards": by_turn.get(t.turn, []),
                "forks": [f for f in forks if f.get("turn") == t.turn],
            })
        first_ctx = (rec.events[0].get("context") or []) if rec.events else []
        task_msgs = [m for m in first_ctx if m.get("type") == "human" or m.get("role") == "user"]
        return {
            "id": run_id,
            "task_id": rec.task_id,
            "task": message_text(task_msgs[-1]) if task_msgs else None,
            "reward": rec.reward,
            "status": str(traj.status),
            "wall_seconds": rec.wall_seconds,
            "exception": rec.exception,
            "T": traj.T,
            "guards_available": available,
            "first_nonterminal_guard": first_nt,
            "oracle": self._oracle(run_id),
            "turns": turns,
        }

    def summary(self) -> dict[str, Any]:
        """Descriptive guards-vs-oracle counts (inference lives in guardfork.stats)."""
        rows = []
        for rid, rec in self.records.items():
            oracle = self._oracle(rid)
            if oracle is None:
                continue
            available, _, g = _guards(rec)
            rows.append({"id": rid, "T": rec.n_turns, "g": g, "t_star": oracle.get("t_star"),
                         "guards_available": available})
        defined = [r for r in rows if r["t_star"] is not None and r["g"] is not None]
        hits = {w: sum(abs(r["g"] - r["t_star"]) <= w for r in defined) for w in WINDOWS}
        return {
            "runs_with_oracle": len(rows),
            "t_star_undefined": sum(r["t_star"] is None for r in rows),
            "no_nonterminal_guard": sum(r["g"] is None for r in rows),
            "comparable": len(defined),
            "hits": hits,
            "rows": rows,
        }
