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
            assert json.loads(r.read())["T"] == 2
    finally:
        srv.shutdown()
        srv.server_close()
