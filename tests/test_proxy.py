from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from guardfork.serve.proxy import extract_response_fields, inject_request_fields, serve


def test_inject_never_overrides():
    out = inject_request_fields({"messages": [], "logprobs": False}, "vllm")
    assert out["logprobs"] is False and out["return_token_ids"] is True
    assert "return_token_ids" not in inject_request_fields({}, "mlx")


def test_sampling_config_injected_with_logged_seed_but_never_overrides():
    import random

    from guardfork.sampling import load_sampling

    cfg = load_sampling()
    assert cfg.request["temperature"] == 0.6 and cfg.seed_per_request and len(cfg.sha256) == 64
    out = inject_request_fields({"messages": []}, "mlx", cfg, random.Random(0))
    assert out["temperature"] == 0.6 and out["top_k"] == 20 and isinstance(out["seed"], int)
    kept = inject_request_fields({"temperature": 0.0, "seed": 7}, "vllm", cfg)
    assert kept["temperature"] == 0.0 and kept["seed"] == 7
    assert "temperature" not in inject_request_fields({}, "mlx", load_sampling("none"))


def test_extract_vllm_and_mlx_shapes():
    vllm = {"prompt_token_ids": [1, 2], "usage": {"prompt_tokens": 2},
            "choices": [{"token_ids": [3], "finish_reason": "stop",
                         "logprobs": {"content": [{"token": "x", "logprob": -0.5}]}}]}
    got = extract_response_fields(vllm)
    assert got["prompt_token_ids"] == [1, 2] and got["completion_logprobs"] == [[-0.5]]
    mlx = {"choices": [{"finish_reason": "stop", "logprobs": {"token_logprobs": [-1.0]}}]}
    assert extract_response_fields(mlx)["completion_logprobs"] == [[-1.0]]


class _Upstream(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Upstream.seen.append(body)
        payload = json.dumps({"prompt_token_ids": [7, 8, 9], "usage": {"prompt_tokens": 3},
                              "choices": [{"message": {"content": "ok"}, "token_ids": [1],
                                           "finish_reason": "stop"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        payload = b'{"data": []}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def servers(tmp_path: Path):
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    log = tmp_path / "proxy.jsonl"
    proxy = serve(f"http://127.0.0.1:{up.server_port}", 0, log)
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (up, proxy)]
    for t in threads:
        t.start()
    yield proxy.server_port, log
    for s in (up, proxy):
        s.shutdown()
        s.server_close()


def test_roundtrip_logs_and_passes_through(servers):
    port, log = servers
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode(),
        headers={"Content-Type": "application/json", "x-guardfork-tag": "unit"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = json.loads(resp.read())
    assert body["choices"][0]["message"]["content"] == "ok"
    assert _Upstream.seen[-1]["return_token_ids"] is True
    deadline = time.monotonic() + 5
    while not (log.exists() and log.read_text().strip()) and time.monotonic() < deadline:
        time.sleep(0.01)
    (rec,) = [json.loads(line) for line in log.read_text().splitlines()]
    assert rec["prompt_token_ids"] == [7, 8, 9] and rec["tag"] == "unit"
    assert rec["status"] == 200 and rec["latency_s"] >= 0

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
        assert json.loads(resp.read()) == {"data": []}
    assert len(log.read_text().splitlines()) == 1
