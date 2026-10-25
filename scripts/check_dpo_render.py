"""Check exported DPO rows against the real tokenizer chat template (no model weights).

For every pair: (1) `render_qwen35` == tokenizer.apply_chat_template(prompt + chosen,
tools); (2) the token ids mlx-lm-lora's DPODataset builds from the mlx row
([system, user(prompt)] + assistant(chosen|rejected)) equal the ids of the real
conversation, and so does the generation-prompt length.

    HF_HUB_OFFLINE=1 uv run --no-project --with transformers --with jinja2 \
        python scripts/check_dpo_render.py --data runs/dpo/smoke \
        --model mlx-community/Qwen3.5-9B-4bit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from guardfork.train.export import render_qwen35  # noqa: E402


def hf_messages(messages: list[dict]) -> list[dict]:
    """OpenAI wire messages -> HF template input (tool-call arguments as dicts)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc["function"])
                if isinstance(fn.get("arguments"), str):
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                calls.append({**tc, "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    return out


def ids(encoded) -> list[int]:
    """Newer transformers return a BatchEncoding from apply_chat_template(tokenize=True)."""
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    return [int(t) for t in encoded]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="mlx-community/Qwen3.5-9B-4bit")
    args = ap.parse_args()
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tok = AutoTokenizer.from_pretrained(args.model)
    neutral = [json.loads(x) for x in (args.data / "neutral.jsonl").read_text().splitlines()
               if x.strip()]
    mlx = [json.loads(x) for x in (args.data / "mlx" / "train.jsonl").read_text().splitlines()
           if x.strip()]
    ok = True
    for i, (n, row) in enumerate(zip(neutral, mlx, strict=True)):
        for side in ("chosen", "rejected"):
            real_msgs = hf_messages(n["prompt"] + n[side])
            real_text = tok.apply_chat_template(real_msgs, tools=n["tools"], tokenize=False)
            ours = render_qwen35(n["prompt"] + n[side], n["tools"])
            real_ids = ids(tok.apply_chat_template(real_msgs, tools=n["tools"], tokenize=True))
            dpo_msgs = [{"role": "system", "content": row["system"]},
                        {"role": "user", "content": row["prompt"]}]
            dpo_ids = ids(tok.apply_chat_template(
                dpo_msgs + [{"role": "assistant", "content": row[side]}], tokenize=True))
            real_prompt = ids(tok.apply_chat_template(
                hf_messages(n["prompt"]), tools=n["tools"], tokenize=True,
                add_generation_prompt=True))
            dpo_prompt = ids(tok.apply_chat_template(dpo_msgs, tokenize=True,
                                                     add_generation_prompt=True))
            checks = {
                "render_text": ours == real_text,
                "dpo_ids": dpo_ids == real_ids,
                "prompt_ids": dpo_prompt == real_prompt,
                "prompt_is_prefix": real_ids[:len(real_prompt)] == real_prompt,
            }
            ok &= all(checks.values())
            print(f"pair {i} {side}: {checks} tokens={len(real_ids)}")
            if not checks["render_text"]:
                j = next((k for k, (a, b) in enumerate(zip(ours, real_text, strict=False))
                          if a != b), min(len(ours), len(real_text)))
                print(f"  first text diff at {j}: ours={ours[j - 40:j + 40]!r}\n"
                      f"  real={real_text[j - 40:j + 40]!r}")
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
