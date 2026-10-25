"""Weekend 0 template parity: vLLM prompt_token_ids vs Qwen3_5Renderer(strip=False).

    uv run --with tinker-cookbook python scripts/check_template_parity.py \
        --log runs/proxy/w0_4b.jsonl --model Qwen/Qwen3.5-4B --n 20

Picks the N longest multi-turn requests (most assistant turns) so thinking-in-history
is exercised. Exit code 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardfork.serve import parity  # noqa: E402
from guardfork.telemetry.proxy_log import iter_proxy_log  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--renderer", choices=["tinker", "hf"], default="tinker")
    ap.add_argument("--show", type=int, default=80, help="tokens of context around a mismatch")
    args = ap.parse_args()

    records = [r for r in iter_proxy_log(args.log) if r.ok and r.prompt_token_ids]
    if not records:
        print("no records with prompt_token_ids; was the proxy run with --backend vllm?")
        return 2
    records.sort(key=lambda r: r.live_turn_index, reverse=True)
    picked = records[: args.n]

    renderer = parity.build_tinker_renderer(args.model) if args.renderer == "tinker" else None
    tokenizer = getattr(renderer, "tokenizer", None)
    failures = 0
    for r in picked:
        if renderer is not None:
            rendered = parity.render_with_tinker(renderer, r.messages, r.tools)
        else:
            rendered = parity.render_with_hf(args.model, r.messages, r.tools)
        res = parity.compare(r.raw.get("id", "?"), r.prompt_token_ids, rendered)
        mark = "OK " if res.equal else "DIFF"
        print(f"{mark} {res.record_id} turn={r.live_turn_index} served={res.served_len} "
              f"rendered={res.rendered_len} first_mismatch={res.first_mismatch}")
        if not res.equal:
            failures += 1
            if tokenizer is not None and res.first_mismatch is not None:
                i, w = res.first_mismatch, args.show
                lo = max(0, i - w)
                print("  served  :", repr(tokenizer.decode(r.prompt_token_ids[lo:i + w])))
                print("  rendered:", repr(tokenizer.decode(rendered[lo:i + w])))
    print(f"\n{len(picked) - failures}/{len(picked)} identical")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
