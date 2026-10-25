"""OpenAI-compatible logging proxy between the Harness-Zero student and vLLM / mlx-lm.

Injects token-id and logprob requests into chat completions and appends one JSONL
record per call (request, response, prompt/completion token ids, logprobs, latency).
The response is returned to the client unchanged.

    python -m guardfork.serve.proxy --upstream http://127.0.0.1:8000 --port 8100 \
        --backend vllm --log runs/proxy/w0_k0.jsonl

Logs contain AppWorld task text: keep them under runs/ (gitignored, hook-blocked).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "host"}


def inject_request_fields(body: dict[str, Any], backend: str) -> dict[str, Any]:
    """Return a copy of a chat-completions body with token-id/logprob requests added.

    Never overrides a field the client already set.
    """
    out = dict(body)
    out.setdefault("logprobs", True)
    if backend == "vllm":
        out.setdefault("return_token_ids", True)
    return out


def extract_response_fields(resp: dict[str, Any]) -> dict[str, Any]:
    """Pull token ids, logprobs, usage, finish reasons out of a completion response.

    Tolerates vLLM (top-level `prompt_token_ids`, per-choice `token_ids`) and mlx-lm
    (no token ids; logprobs shape varies by version).
    """
    choices = resp.get("choices") or []
    completion_ids: list[list[int] | None] = []
    logprobs: list[list[float] | None] = []
    finish: list[str | None] = []
    for ch in choices:
        if not isinstance(ch, dict):
            continue
        completion_ids.append(ch.get("token_ids"))
        finish.append(ch.get("finish_reason"))
        lp = ch.get("logprobs")
        if isinstance(lp, dict) and isinstance(lp.get("content"), list):
            logprobs.append([t.get("logprob") for t in lp["content"] if isinstance(t, dict)])
        elif isinstance(lp, dict) and isinstance(lp.get("token_logprobs"), list):
            logprobs.append(lp["token_logprobs"])
        else:
            logprobs.append(None)
    prompt_ids = resp.get("prompt_token_ids")
    if prompt_ids is None and choices and isinstance(choices[0], dict):
        prompt_ids = choices[0].get("prompt_token_ids")
    return {
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion_ids,
        "completion_logprobs": logprobs,
        "finish_reasons": finish,
        "usage": resp.get("usage"),
    }


class JsonlLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def make_handler(upstream: str, backend: str, log: JsonlLog, tag: str | None, timeout: float):
    upstream = upstream.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _forward(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body_bytes = self.rfile.read(length) if length else b""
            is_chat = method == "POST" and self.path.rstrip("/").endswith("/chat/completions")
            req_json: dict[str, Any] | None = None
            if is_chat:
                try:
                    req_json = json.loads(body_bytes)
                except json.JSONDecodeError:
                    req_json = None
            if req_json is not None:
                body_bytes = json.dumps(inject_request_fields(req_json, backend)).encode()

            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}
            req = urllib.request.Request(
                upstream + self.path, data=body_bytes or None, method=method, headers=headers
            )
            start = time.monotonic()
            record: dict[str, Any] = {
                "id": str(uuid.uuid4()),
                "ts": time.time(),
                "tag": self.headers.get("x-guardfork-tag") or tag,
                "backend": backend,
                "path": self.path,
                "request": req_json,
            }
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status, resp_headers = resp.status, resp.headers
                    streaming = "text/event-stream" in (resp_headers.get("Content-Type") or "")
                    self.send_response(status)
                    for k, v in resp_headers.items():
                        if k.lower() not in HOP_HEADERS:
                            self.send_header(k, v)
                    if streaming:
                        self.send_header("Transfer-Encoding", "chunked")
                        self.end_headers()
                        chunks = []
                        for chunk in iter(lambda: resp.read1(65536), b""):
                            chunks.append(chunk)
                            self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                        payload = b"".join(chunks)
                        record["stream_raw"] = payload.decode("utf-8", errors="replace")
                    else:
                        payload = resp.read()
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
            except urllib.error.HTTPError as err:
                status, payload = err.code, err.read()
                self.send_response(status)
                ctype = err.headers.get("Content-Type", "application/json")
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                streaming = False
            except (urllib.error.URLError, TimeoutError) as err:
                payload = json.dumps({"error": f"upstream unreachable: {err}"}).encode()
                status, streaming = 502, False
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            if not is_chat:
                return
            record["status"] = status
            record["latency_s"] = time.monotonic() - start
            if not streaming:
                try:
                    resp_json = json.loads(payload)
                except json.JSONDecodeError:
                    resp_json = None
                record["response"] = resp_json
                if isinstance(resp_json, dict):
                    record.update(extract_response_fields(resp_json))
            log.write(record)

        def do_GET(self) -> None:
            self._forward("GET")

        def do_POST(self) -> None:
            self._forward("POST")

    return Handler


def serve(
    upstream: str,
    port: int,
    log_path: Path,
    backend: str = "vllm",
    host: str = "127.0.0.1",
    tag: str | None = None,
    timeout: float = 600.0,
) -> ThreadingHTTPServer:
    handler = make_handler(upstream, backend, JsonlLog(log_path), tag, timeout)
    return ThreadingHTTPServer((host, port), handler)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream", required=True, help="e.g. http://127.0.0.1:8000")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 only if Harbor containers must reach the proxy")
    ap.add_argument("--backend", choices=["vllm", "mlx"], default="vllm")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--tag", default=None, help="label written into every record")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    server = serve(args.upstream, args.port, args.log, args.backend, args.host, args.tag,
                   args.timeout)
    print(f"guardfork proxy {args.host}:{args.port} -> {args.upstream} ({args.backend}), "
          f"log={args.log}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
