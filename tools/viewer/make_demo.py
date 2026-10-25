"""Write a synthetic (invented, not AppWorld) run into runs/viewer_demo/ for trying the viewer.

    uv run python tools/viewer/make_demo.py
    uv run python tools/viewer/server.py --jobs runs/viewer_demo/jobs \
        --results runs/viewer_demo/results --proxy-log runs/viewer_demo/proxy.jsonl
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "runs" / "viewer_demo"
TASK = "Pay my roommate back for every dinner they paid for last month."
PAY = "print(apis.venmo.create_transaction(receiver='roommate', amount=42))"
STEPS = [
    ("Find the roommate in contacts.", "print(apis.phone.search_contacts(relationship='roommate'))",
     '[{"name": "Sam", "email": "sam@example.com"}]'),
    ("List last month's transactions.", "print(apis.venmo.list_transactions(month='last'))",
     '[... 20 transactions, page 1 of 3 ...]'),
    ("Filter for dinners.", "dinners = [t for t in page1 if 'dinner' in t['note']]",
     "2 dinners found"),
    ("Pay back.", PAY, '{"status": "ok"}'),
    ("Pay back.", PAY, '{"status": "ok"}'),
    ("Pay back.", PAY, '{"status": "ok"}'),
    ("Done!", None, None),
]


def main() -> None:
    shutil.rmtree(ROOT, ignore_errors=True)
    trial = ROOT / "jobs" / "demo_job" / "demo_trial_0"
    (trial / "agent" / "teacher" / "trial").mkdir(parents=True)
    (trial / "config.json").write_text("{}")
    (trial / "result.json").write_text(json.dumps({
        "task_name": "demo000_1", "started_at": "2026-10-24T00:00:00+00:00",
        "finished_at": "2026-10-24T00:03:12+00:00",
        "verifier_result": {"rewards": {"reward": 0.0}}}))
    context = [{"type": "human", "content": TASK}]
    events, proxy = [], []
    for i, (thought, cmd, obs) in enumerate(STEPS):
        events.append({
            "type": "reviewed_turn", "task_id": "demo000_1", "trial": 0, "candidate_id": str(i),
            "system_message": None, "context": list(context),
            "original": {"reasoning": thought, "content": "" if cmd else thought,
                         "tool_call": {"name": "execute", "command": cmd} if cmd else None},
            "raw_original": {"tool_calls": [{"name": "execute", "args": {"command": cmd}}]
                             if cmd else [], "invalid_tool_calls": []},
            "candidate_issue": None,
        })
        proxy.append({"status": 200, "latency_s": 2.0 + i * 0.4,
                      "usage": {"completion_tokens": 120 + 10 * i},
                      "prompt_token_ids": list(range(3000 + 400 * i)),
                      "request": {"messages": [{"role": "user" if m["type"] == "human" else
                                                m["type"], "content": m["content"]}
                                               for m in context]}})
        context += [{"type": "ai", "content": ""}]
        if obs is not None:
            context += [{"type": "tool", "content": obs}]
    (trial / "agent" / "teacher" / "trial" / "trajectory.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events))
    (ROOT / "proxy.jsonl").write_text("".join(json.dumps(p) + "\n" for p in proxy))
    res = ROOT / "results" / "demo_job" / "demo_trial_0"
    res.mkdir(parents=True)
    (res / "oracle.json").write_text(json.dumps({
        "K": 8, "values": {"1": [5, 8], "2": [5, 8], "3": [4, 8], "4": [0, 8], "6": [0, 8]},
        "t_star": 3}))
    (res / "forks.json").write_text(json.dumps(
        [{"turn": 6, "k": k, "reward": 0.0, "first_action": "pay again"} for k in range(8)]))
    print(f"wrote {ROOT}")


if __name__ == "__main__":
    main()
