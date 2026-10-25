"""Local, read-only trajectory viewer.

    uv run python tools/viewer/server.py   # defaults: jobs/ forks/ results/ runs/proxy/ :8765

Then open http://127.0.0.1:8765. Binds to localhost only: pages show AppWorld task text.
Every API request rescans the run dirs, so new and in-progress trials appear without restart.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import RunIndex  # noqa: E402

INDEX_HTML = Path(__file__).resolve().parent / "index.html"


def make_handler(index: RunIndex):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            return

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self) -> None:
            try:
                self._route()
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _route(self) -> None:
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/runs":
                self._json(index.runs())
            elif url.path == "/api/jobs":
                self._json(index.jobs())
            elif url.path == "/api/run":
                run_id = (parse_qs(url.query).get("id") or [""])[0]
                detail = index.detail(run_id)
                self._json(detail if detail else {"error": "unknown run"},
                           200 if detail else 404)
            elif url.path == "/api/summary":
                self._json(index.summary())
            elif url.path == "/api/reload":
                index.reload()
                self._json({"runs": len(index.runs())})
            else:
                self._json({"error": "not found"}, 404)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, default=ROOT / "jobs")
    ap.add_argument("--forks", type=Path, default=ROOT / "forks")
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--proxy-log", type=Path, default=ROOT / "runs" / "proxy",
                    help="proxy JSONL file or a directory of them")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    index = RunIndex(args.jobs, args.results, args.proxy_log, forks_root=args.forks)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(index))
    print(f"viewer: http://127.0.0.1:{args.port}  ({len(index.runs())} runs from "
          f"{args.jobs}, {args.forks})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
