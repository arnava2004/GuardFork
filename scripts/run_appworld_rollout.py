"""Launch AppWorld rollouts under Harbor with the dedicated AppWorld agent + environment.

Unlike `harness-zero run-rollout` (which hardcodes the generic agent/docker env), this
passes `envs.appworld.agent:AppWorldMinisweAgent` and
`envs.appworld.harbor_environment:AppWorldDockerEnvironment`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

FORK_AGENT = "guardfork.fork.fork_agent:ForkReplayAppWorldAgent"
GUARDFORK_ROOT = Path(__file__).resolve().parents[1]
if str(GUARDFORK_ROOT) not in sys.path:
    sys.path.insert(0, str(GUARDFORK_ROOT))


def harbor_env(hz_root: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(hz_root), str(hz_root / "src"), str(GUARDFORK_ROOT)]),
        "PYTHONHASHSEED": "0",
        "BUILDX_BUILDER": "default",
        "COMPOSE_BAKE": "false",
    }


def build_harbor_command(
    *,
    hz_root: Path,
    tasks: list[str],
    student_model: str,
    student_base_url: str,
    job_name: str,
    output_dir: Path,
    student_max_turns: int = 40,
    attempts: int = 1,
    concurrency: int = 1,
    max_retries: int = 0,
    student_reasoning_effort: str = "high",
    student_reasoning_prefill: str = "none",
    student_harness_dir: str | None = None,
    agent_import_path: str = "envs.appworld.agent:AppWorldMinisweAgent",
    replay_plan_path: Path | None = None,
    student_timeout: float | None = 600.0,
) -> list[str]:
    if replay_plan_path is not None:
        agent_import_path = FORK_AGENT
    harbor = hz_root / ".venv" / "bin" / "harbor"
    dataset = hz_root / "data" / "appworld-harbor" / "tasks"
    components = hz_root / "harness_bank" / "empty"
    student_kwargs: dict = {
        "base_url": student_base_url,
        "api_key": "guardfork-local-gateway-token",
        "use_responses_api": False,
        "reasoning_effort": student_reasoning_effort,
    }
    if student_timeout is not None:
        student_kwargs["timeout"] = student_timeout
    model_kwargs = json.dumps(student_kwargs, separators=(",", ":"))
    cmd = [
        str(harbor),
        "run",
        "--path",
        str(dataset),
        "--env",
        "envs.appworld.harbor_environment:AppWorldDockerEnvironment",
        "--agent",
        agent_import_path,
        "--model",
        student_model,
        "--agent-kwarg",
        f"model_kwargs={model_kwargs}",
        "--agent-kwarg",
        'teacher_provider="passthrough"',
        "--agent-kwarg",
        'teacher_model="none"',
        "--agent-kwarg",
        'teacher_reasoning_effort="none"',
        "--agent-kwarg",
        f"student_reasoning_prefill={json.dumps(student_reasoning_prefill)}",
        "--agent-kwarg",
        "max_replacements_per_trial=0",
        "--agent-kwarg",
        f"student_max_turns={student_max_turns}",
        "--agent-kwarg",
        f"components_dir={json.dumps(str(components))}",
        "--agent-kwarg",
        "teacher_middleware_factory=null",
        "--agent-env",
        f"HARNESS_ZERO_GATEWAY_URL={student_base_url}",
        "--agent-env",
        "HARNESS_ZERO_GATEWAY_TOKEN=guardfork-local-gateway-token",
        "--n-attempts",
        str(attempts),
        "--n-concurrent",
        str(concurrency),
        "--max-retries",
        str(max_retries),
        "--job-name",
        job_name,
        "--jobs-dir",
        str(output_dir),
        "--yes",
    ]
    extra: list[str] = []
    if student_harness_dir is not None:
        extra += ["--agent-kwarg", f"student_harness_dir={json.dumps(student_harness_dir)}"]
    if replay_plan_path is not None:
        extra += ["--agent-kwarg", f"replay_plan_path={json.dumps(str(replay_plan_path))}"]
    if extra:
        idx = cmd.index("--n-attempts")
        cmd[idx:idx] = extra
    for task in tasks:
        cmd.extend(["--include-task-name", task])
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hz-root", type=Path,
                    default=Path(__file__).resolve().parents[2] / "harness-zero")
    ap.add_argument("--task", action="append", dest="tasks", default=[],
                    help="task id (repeatable)")
    ap.add_argument("--tasks-file", type=Path, default=None)
    ap.add_argument("--student-model", default="openai:mlx-community/Qwen3.5-4B-4bit")
    ap.add_argument("--student-base-url", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--student-max-turns", type=int, default=40)
    ap.add_argument("--attempts", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--student-harness-dir", default=None)
    ap.add_argument("--replay-plan", type=Path, default=None,
                    help="ReplayPlan JSON; runs ForkReplayAppWorldAgent on the plan's task")
    ap.add_argument("--student-timeout", type=float, default=600.0,
                    help="per-request client timeout in seconds (Harness-Zero's own default "
                         "of 120 s kills long 9B reasoning turns)")
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "jobs")
    args = ap.parse_args()

    tasks = list(args.tasks)
    if args.tasks_file:
        tasks.extend(
            line.strip()
            for line in args.tasks_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        )
    if args.replay_plan is not None and not tasks:
        tasks = [json.loads(args.replay_plan.read_text())["task_id"]]
    if not tasks:
        raise SystemExit("pass at least one --task or --tasks-file")

    hz_root = args.hz_root.resolve()
    env = harbor_env(hz_root)
    cmd = build_harbor_command(
        hz_root=hz_root,
        tasks=tasks,
        student_model=args.student_model,
        student_base_url=args.student_base_url,
        job_name=args.job_name,
        output_dir=args.output_dir.resolve(),
        student_max_turns=args.student_max_turns,
        attempts=args.attempts,
        concurrency=args.concurrency,
        student_harness_dir=args.student_harness_dir,
        replay_plan_path=args.replay_plan.resolve() if args.replay_plan else None,
        student_timeout=args.student_timeout,
    )
    code = subprocess.run(cmd, cwd=hz_root, env=env, check=False).returncode
    write_run_record(args.output_dir.resolve() / args.job_name, hz_root, {
        "student_model": args.student_model, "student_base_url": args.student_base_url,
        "student_max_turns": args.student_max_turns, "student_timeout": args.student_timeout,
        "attempts": args.attempts, "concurrency": args.concurrency,
        "student_harness_dir": args.student_harness_dir,
        "replay_plan": str(args.replay_plan) if args.replay_plan else None,
        "n_tasks": len(tasks), "harbor_exit_code": code,
    })
    return code


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_run_record(job_dir: Path, hz_root: Path, launch: dict) -> None:
    """`guardfork_run.json` beside Harbor's job files: launch config, sampling, commits.
    Sampling is applied by the proxy from config/sampling.toml; keep the two in sync."""
    from guardfork.sampling import load_sampling

    if not job_dir.is_dir():
        return
    record = {
        **launch,
        "sampling_expected": load_sampling().summary(),
        "guardfork_commit": _git_sha(GUARDFORK_ROOT),
        "harness_zero_commit": _git_sha(hz_root),
        "pythonhashseed": "0",
    }
    (job_dir / "guardfork_run.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
