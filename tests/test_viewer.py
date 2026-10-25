from __future__ import annotations

import json
import sys
import threading
import urllib.request
from pathlib import Path

VIEWER = Path(__file__).resolve().parent.parent / "tools" / "viewer"
sys.path.insert(0, str(VIEWER))

from data import RunIndex  # noqa: E402
from server import make_handler  # noqa: E402


def _ctx(*msgs):
    return [{"type": t, "content": c} for t, c in msgs]


def _setup(tmp_path: Path, make_trial, event):
    e1 = event("aaaaaaa_1", command="ls")
    e1["context"] = _ctx(("human", "do the task"))
    e2 = event("aaaaaaa_1", command="cat x")
    e2["context"] = _ctx(("human", "do the task"), ("ai", ""), ("tool", "file-list"))
    make_trial("job0", "trial0", "aaaaaaa_1", 0.0, [e1, e2])
    res = tmp_path / "results" / "job0" / "trial0"
    res.mkdir(parents=True)
    (res / "oracle.json").write_text(json.dumps({"K": 8, "values": {"1": [4, 8]}, "t_star": 1}))
    (res / "forks.json").write_text(json.dumps([{"turn": 1, "k": 0, "reward": 1.0}]))
    proxy = tmp_path / "proxy.jsonl"
    proxy.write_text(json.dumps({
        "status": 200, "latency_s": 1.5, "usage": {"completion_tokens": 7},
        "prompt_token_ids": [1, 2, 3],
        "request": {"messages": [{"role": "user", "content": "do the task"}]},
    }) + "\n")
    return RunIndex(tmp_path, tmp_path / "results", proxy)


def test_detail_joins_everything(tmp_path, make_trial, event):
    index = _setup(tmp_path, make_trial, event)
    (run,) = [r for r in index.runs() if r["id"] == "job0/trial0"]
    assert run["has_oracle"]
    d = index.detail("job0/trial0")
    assert d["task"] == "do the task" and d["guards_available"] is True
    t1, t2 = d["turns"]
    assert t1["observation"] == "file-list" and t2["observation"] is None
    assert t1["prompt_tokens"] == 3 and t1["latency_s"] == 1.5 and t2["prompt_tokens"] is None
    assert t1["forks"][0]["reward"] == 1.0
    assert index.detail("../etc/passwd") is None
    s = index.summary()
    assert s["runs_with_oracle"] == 1 and s["no_nonterminal_guard"] == 1


def _running_trial(job_dir: Path, trial: str, events: list[dict] | None = None) -> Path:
    """Harbor trial dir mid-run: config.json (+ maybe a partial trajectory), no result.json."""
    trial_dir = job_dir / trial
    (trial_dir / "agent").mkdir(parents=True)
    (trial_dir / "config.json").write_text("{}")
    if events is not None:
        tdir = trial_dir / "agent" / "teacher" / "trial"
        tdir.mkdir(parents=True)
        (tdir / "trajectory.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return trial_dir


def _by_id(index: RunIndex) -> dict[str, dict]:
    return {r["id"]: r for r in index.runs()}


def test_rescan_picks_up_new_and_finished_trials(tmp_path, make_trial, event):
    make_trial("jobs/jobA", "t1", "aaaaaaa_1", 1.0, [event("aaaaaaa_1")])
    index = RunIndex(tmp_path / "jobs")
    assert set(_by_id(index)) == {"jobA/t1"}

    make_trial("jobs/jobA", "t2", "bbbbbbb_1", 0.0, [event("bbbbbbb_1")])
    running = _running_trial(tmp_path / "jobs" / "jobA", "t3", [event("ccccccc_1")])
    runs = _by_id(index)
    assert set(runs) == {"jobA/t1", "jobA/t2", "jobA/t3"}
    assert [runs[k]["status"] for k in sorted(runs)] == ["pass", "fail", "running"]
    assert runs["jobA/t3"]["trial"] == "t3" and runs["jobA/t3"]["job"] == "jobA"
    assert runs["jobA/t3"]["task_id"] == "ccccccc_1" and runs["jobA/t3"]["n_turns"] == 1

    (running / "result.json").write_text(json.dumps(
        {"task_name": "ccccccc_1", "exception_info": None,
         "verifier_result": {"rewards": {"reward": 1.0}}}))
    assert _by_id(index)["jobA/t3"]["status"] == "pass"
    (tmp_path / "jobs" / "jobA" / "t2").rename(tmp_path / "jobs" / "jobA" / "gone")
    assert "jobA/t2" not in _by_id(index)


def test_running_vs_error_status(tmp_path, make_trial, event):
    job = tmp_path / "jobs" / "jobB"
    make_trial("jobs/jobB", "crashed", "aaaaaaa_1", None, [event("aaaaaaa_1")])
    _running_trial(job, "starting")
    (job / "config.json").write_text("{}")
    (job / "job.log").write_text("")
    index = RunIndex(tmp_path / "jobs")
    runs = _by_id(index)
    assert runs["jobB/crashed"]["status"] == "error"
    assert runs["jobB/crashed"]["exception_type"] == "Boom"
    assert runs["jobB/starting"]["status"] == "running"
    assert runs["jobB/starting"]["exception_type"] is None
    assert index.detail("jobB/starting")["exception"] is None
    assert index.jobs()[0]["running"] == 1 and index.jobs()[0]["error"] == 1

    (job / "result.json").write_text(json.dumps({"finished_at": "2026-10-24T01:00:00+00:00",
                                                 "n_total_trials": 2}))
    assert _by_id(index)["jobB/starting"]["status"] == "error"


def test_forks_indexed_with_job_counts(tmp_path, make_trial, event):
    make_trial("jobs/screen", "p", "aaaaaaa_1", 1.0, [event("aaaaaaa_1")])
    make_trial("jobs/screen", "f", "aaaaaaa_1", 0.0, [event("aaaaaaa_1")])
    _running_trial(tmp_path / "jobs" / "screen", "r")
    fork_job = make_trial("forks/fork_smoke/fork-t6", "x__abc", "aaaaaaa_1", 1.0,
                          [event("aaaaaaa_1")])
    for name in ("config.json", "job.log"):
        (fork_job / name).write_text("{}")
    (fork_job / "result.json").write_text(json.dumps(
        {"finished_at": "2026-10-24T01:00:00+00:00", "n_total_trials": 1}))
    (tmp_path / "forks" / "fork_smoke" / "plans").mkdir()

    index = RunIndex(tmp_path / "jobs", forks_root=tmp_path / "forks")
    runs = _by_id(index)
    assert set(runs) == {"screen/p", "screen/f", "screen/r", "forks/fork_smoke/fork-t6/x__abc"}
    fork = runs["forks/fork_smoke/fork-t6/x__abc"]
    assert fork["source"] == "forks" and fork["status"] == "pass"
    assert index.detail(fork["id"])["T"] == 1

    jobs = {j["job"]: j for j in index.jobs()}
    assert set(jobs) == {"screen", "forks/fork_smoke/fork-t6"}
    s = jobs["screen"]
    assert (s["n_trials"], s["graded"], s["passed"], s["running"], s["pass_rate"]) == (
        3, 2, 1, 1, 0.5)
    f = jobs["forks/fork_smoke/fork-t6"]
    assert (f["graded"], f["passed"], f["n_total_trials"], f["finished"]) == (1, 1, 1, True)


def test_proxy_log_dir_is_tailed(tmp_path, make_trial, event):
    e1 = event("aaaaaaa_1")
    e1["context"] = _ctx(("human", "first"))
    make_trial("jobs/j", "t", "aaaaaaa_1", 1.0, [e1])
    proxy_dir = tmp_path / "proxy"
    proxy_dir.mkdir()
    (proxy_dir / "other.jsonl").write_text("")
    log = proxy_dir / "live.jsonl"
    log.write_text("")
    index = RunIndex(tmp_path / "jobs", proxy_log=proxy_dir)
    assert index.detail("j/t")["turns"][0]["prompt_tokens"] is None
    rec = {"status": 200, "latency_s": 2.0, "prompt_token_ids": [1, 2],
           "request": {"messages": [{"role": "user", "content": "first"}]}}
    with log.open("a") as f:
        f.write(json.dumps(rec) + "\n" + json.dumps(rec)[:10])
    assert index.detail("j/t")["turns"][0]["prompt_tokens"] == 2


def test_server_routes(tmp_path, make_trial, event):
    from http.server import ThreadingHTTPServer

    index = _setup(tmp_path, make_trial, event)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(index))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            assert b"GuardFork viewer" in r.read()
        with urllib.request.urlopen(base + "/api/run?id=job0%2Ftrial0", timeout=5) as r:
            d = json.loads(r.read())
            assert d["T"] == 2 and d["status"] == "fail" and d["trial"] == "trial0"
        with urllib.request.urlopen(base + "/api/jobs", timeout=5) as r:
            (job,) = json.loads(r.read())
            assert job["job"] == "job0" and job["graded"] == 1 and job["pass_rate"] == 0.0
        with urllib.request.urlopen(base + "/api/reload", timeout=5) as r:
            assert json.loads(r.read())["runs"] == 1
    finally:
        srv.shutdown()
        srv.server_close()
