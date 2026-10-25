"""Read-only data layer for the trajectory viewer.

Sources (all optional except jobs):
- jobs/<job>/<trial>/...                  Harbor trials (see guardfork.telemetry.trials)
- forks/<group>/<job>/<trial>/...        fork / replay Harbor jobs, same trial layout
- results/<job>/<trial>/oracle.json      {"K": 8, "values": {"<turn>": [passes, graded]},
                                           "t_star": int | null, "t_star_drop": int | null}
- results/<job>/<trial>/forks.json       [{"turn": int, "k": int, "reward": float | null,
                                            "first_action": str | null}]
- runs/proxy/*.jsonl                     proxy logs, joined to turns by exact message text

The index is rescanned on every query: directory listings and a few stats per trial,
re-reading only trials whose artifacts changed and only the appended tail of proxy logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

from guardfork.fork.replay_checks import canonical_messages
from guardfork.telemetry.proxy_log import ProxyRecord
from guardfork.telemetry.to_trajectory import message_text, to_trajectory
from guardfork.telemetry.trials import TrialRecord, load_trial

WINDOWS = (0, 1, 2, 3)
MAX_JOB_DEPTH = 3
TEACHER = Path("agent") / "teacher"
TRIAL_FILES = ("result.json", "agent/teacher/trial/trajectory.jsonl",
               "agent/teacher/trial/result.json")

Sig = tuple[tuple[int, int] | None, ...]


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
    except (json.JSONDecodeError, OSError):
        return None


def _stat(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _subdirs(path: Path) -> list[Path]:
    try:
        return sorted(Path(e.path) for e in os.scandir(path) if e.is_dir())
    except OSError:
        return []


def _is_trial_dir(path: Path) -> bool:
    return (path / "config.json").is_file() and not (path / "job.log").exists()


def _is_job_dir(path: Path) -> bool:
    return (path / "job.log").exists() or any(_is_trial_dir(c) for c in _subdirs(path))


def _find_jobs(root: Path, depth: int = 1) -> list[Path]:
    """Job dirs below (never at) `root`: `jobs/<job>` and `forks/<group>/<job>`."""
    out: list[Path] = []
    for child in _subdirs(root):
        if _is_job_dir(child):
            out.append(child)
        elif depth < MAX_JOB_DEPTH:
            out.extend(_find_jobs(child, depth + 1))
    return out


def _exception_type(exc: Any) -> str | None:
    if not exc:
        return None
    if isinstance(exc, dict):
        return str(exc.get("exception_type") or exc.get("type") or "exception")
    return str(exc)


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


class _ProxyFile:
    """Incrementally tailed JSONL proxy log: only appended bytes are parsed."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.inode: int | None = None
        self.offset = 0
        self.by_key: dict[str, dict[str, Any]] = {}

    def update(self) -> None:
        try:
            st = self.path.stat()
        except OSError:
            self.inode, self.offset, self.by_key = None, 0, {}
            return
        if st.st_ino != self.inode or st.st_size < self.offset:
            self.inode, self.offset, self.by_key = st.st_ino, 0, {}
        if st.st_size == self.offset:
            return
        with self.path.open("rb") as f:
            f.seek(self.offset)
            chunk = f.read(st.st_size - self.offset)
        end = chunk.rfind(b"\n") + 1
        self.offset += end
        for line in chunk[:end].decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                rec = ProxyRecord(json.loads(line))
            except json.JSONDecodeError:
                continue
            if rec.ok:
                self.by_key[_request_key(rec)] = {
                    "prompt_tokens": rec.prompt_tokens,
                    "completion_tokens": rec.completion_tokens,
                    "latency_s": rec.raw.get("latency_s"),
                }


class RunIndex:
    def __init__(self, jobs_root: Path, results_root: Path | None = None,
                 proxy_log: Path | None = None, forks_root: Path | None = None) -> None:
        self.jobs_root = jobs_root
        self.results_root = results_root
        self.proxy_log = proxy_log
        self.forks_root = forks_root
        self._lock = threading.RLock()
        self._trials: dict[str, tuple[Sig, TrialRecord]] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._proxy_files: dict[Path, _ProxyFile] = {}

    def _sources(self) -> list[tuple[str, str, Path]]:
        out = [("jobs", "", self.jobs_root)]
        if self.forks_root is not None:
            out.append(("forks", "forks/", self.forks_root))
        return out

    def reload(self) -> None:
        with self._lock:
            self._trials.clear()
            self._proxy_files.clear()
        self.refresh()

    def refresh(self) -> None:
        """Rescan job/trial dirs; re-load only trials whose artifact stats changed."""
        with self._lock:
            trials: dict[str, tuple[Sig, TrialRecord]] = {}
            meta: dict[str, dict[str, Any]] = {}
            jobs: dict[str, dict[str, Any]] = {}
            for source, prefix, root in self._sources():
                if root is None or not root.is_dir():
                    continue
                for job_dir in _find_jobs(root):
                    job = prefix + job_dir.relative_to(root).as_posix()
                    job_result = _read_json(job_dir / "result.json") or {}
                    job_finished = bool(job_result.get("finished_at"))
                    jobs[job] = {"job": job, "source": source,
                                 "n_total_trials": job_result.get("n_total_trials"),
                                 "finished": job_finished}
                    for trial_dir in _subdirs(job_dir):
                        if not _is_trial_dir(trial_dir):
                            continue
                        rid = f"{job}/{trial_dir.name}"
                        sig = tuple(_stat(trial_dir / f) for f in TRIAL_FILES)
                        cached = self._trials.get(rid)
                        if cached is None or cached[0] != sig:
                            cached = (sig, load_trial(trial_dir / TEACHER, job))
                        trials[rid] = cached
                        meta[rid] = {"source": source, "job": job, "trial": trial_dir.name,
                                     "has_result": sig[0] is not None,
                                     "job_finished": job_finished}
            self._trials, self._meta, self._jobs = trials, meta, jobs

    @property
    def records(self) -> dict[str, TrialRecord]:
        self.refresh()
        with self._lock:
            return {rid: rec for rid, (_, rec) in self._trials.items()}

    def _proxy_lookup(self, key: str) -> dict[str, Any] | None:
        for pf in reversed(list(self._proxy_files.values())):
            if key in pf.by_key:
                return pf.by_key[key]
        return None

    def _refresh_proxy(self) -> None:
        if not self.proxy_log or not self.proxy_log.exists():
            self._proxy_files = {}
            return
        files = (sorted(self.proxy_log.glob("**/*.jsonl")) if self.proxy_log.is_dir()
                 else [self.proxy_log])
        self._proxy_files = {f: self._proxy_files.get(f) or _ProxyFile(f) for f in files}
        for pf in self._proxy_files.values():
            pf.update()

    def _status(self, rid: str, rec: TrialRecord) -> str:
        if rec.reward is not None:
            return "pass" if rec.passed else "fail"
        m = self._meta[rid]
        if m["has_result"] or m["job_finished"]:
            return "error"
        return "running"

    def _results_dir(self, run_id: str) -> Path | None:
        return self.results_root / run_id if self.results_root else None

    def _run_row(self, rid: str, rec: TrialRecord) -> dict[str, Any]:
        m = self._meta[rid]
        status = self._status(rid, rec)
        return {
            "id": rid, "source": m["source"], "job": m["job"], "trial": m["trial"],
            "task_id": rec.task_id, "reward": rec.reward, "status": status,
            "n_turns": rec.n_turns, "wall_seconds": rec.wall_seconds,
            "exception_type": _exception_type(rec.exception) if status == "error" else None,
            "has_oracle": self._oracle(rid) is not None,
        }

    def runs(self) -> list[dict[str, Any]]:
        self.refresh()
        with self._lock:
            return [self._run_row(rid, rec) for rid, (_, rec) in self._trials.items()]

    def jobs(self) -> list[dict[str, Any]]:
        rows = self.runs()
        with self._lock:
            out = {j: dict(v, n_trials=0, graded=0, passed=0, failed=0, running=0, error=0)
                   for j, v in self._jobs.items()}
        for r in rows:
            j = out[r["job"]]
            j["n_trials"] += 1
            j[{"pass": "passed", "fail": "failed"}.get(r["status"], r["status"])] += 1
        for j in out.values():
            j["graded"] = j["passed"] + j["failed"]
            j["pass_rate"] = j["passed"] / j["graded"] if j["graded"] else None
        return list(out.values())

    def _oracle(self, run_id: str) -> dict[str, Any] | None:
        d = self._results_dir(run_id)
        return _read_json(d / "oracle.json") if d else None

    def _forks(self, run_id: str) -> list[dict[str, Any]]:
        d = self._results_dir(run_id)
        return (_read_json(d / "forks.json") or []) if d else []

    def detail(self, run_id: str) -> dict[str, Any] | None:
        self.refresh()
        with self._lock:
            cached = self._trials.get(run_id)
            if cached is None:
                return None
            rec = cached[1]
            row = self._run_row(run_id, rec)
            self._refresh_proxy()
            traj = to_trajectory(rec)
            available, firings, first_nt = _guards(rec)
            by_turn: dict[int, list[dict[str, Any]]] = {}
            for f in firings:
                by_turn.setdefault(f["turn"], []).append(f)
            forks = self._forks(run_id)
            turns = []
            for i, t in enumerate(traj.turns):
                proxy = (self._proxy_lookup(_event_key(rec.events[i]))
                         if i < len(rec.events) else None) or {}
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
                    "prompt_tokens": proxy.get("prompt_tokens"),
                    "completion_tokens": proxy.get("completion_tokens"),
                    "latency_s": proxy.get("latency_s"),
                    "guards": by_turn.get(t.turn, []),
                    "forks": [f for f in forks if f.get("turn") == t.turn],
                })
        first_ctx = (rec.events[0].get("context") or []) if rec.events else []
        task_msgs = [m for m in first_ctx if m.get("type") == "human" or m.get("role") == "user"]
        return {
            **row,
            "task": message_text(task_msgs[-1]) if task_msgs else None,
            "trajectory_status": str(traj.status),
            "exception": rec.exception if row["status"] != "running" else None,
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
