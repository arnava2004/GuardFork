"""Weekend 0 descriptive measurements and K0–K4 evaluation.

Point estimates only. Confidence intervals and tests belong to `guardfork/stats/`.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from guardfork.config import KillConfig
from guardfork.splits import scenario_of
from guardfork.telemetry.proxy_log import ProxyRecord
from guardfork.telemetry.trials import TrialRecord

MAX_TURNS = 40


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def tgc(trials: Sequence[TrialRecord]) -> float | None:
    """Task goal completion (%) over all graded episodes."""
    graded = [t for t in trials if t.reward is not None]
    if not graded:
        return None
    return 100.0 * sum(t.passed for t in graded) / len(graded)


def sgc(trials: Sequence[TrialRecord]) -> float | None:
    """Scenario goal completion (%), computed within each job (one seed pass), then averaged.

    A scenario counts only if all of its tasks in that job were graded; it passes iff
    every one of them passed.
    """
    per_job: dict[str, dict[str, list[TrialRecord]]] = defaultdict(lambda: defaultdict(list))
    for t in trials:
        if t.task_id:
            per_job[t.job][scenario_of(t.task_id)].append(t)
    job_scores = []
    for scenarios in per_job.values():
        complete = [ts for ts in scenarios.values()
                    if all(t.reward is not None for t in ts) and len(ts) >= 3]
        if complete:
            job_scores.append(100.0 * sum(all(t.passed for t in ts) for ts in complete)
                              / len(complete))
    return statistics.fmean(job_scores) if job_scores else None


def per_task_pass(trials: Sequence[TrialRecord]) -> dict[str, tuple[int, int]]:
    """task_id -> (passes, graded episodes)."""
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for t in trials:
        if t.task_id and t.reward is not None:
            out[t.task_id][0] += int(t.passed)
            out[t.task_id][1] += 1
    return {k: (v[0], v[1]) for k, v in sorted(out.items())}


def pass_band_fraction(
    counts: dict[str, tuple[int, int]], band: tuple[float, float], min_samples: int
) -> tuple[float | None, int, int]:
    """(fraction of tasks with >= min_samples and p_hat in band, n_in_band, n_tasks)."""
    if not counts:
        return None, 0, 0
    lo, hi = band
    in_band = sum(1 for p, n in counts.values() if n >= min_samples and lo <= p / n <= hi)
    return in_band / len(counts), in_band, len(counts)


def second_stage_tasks(counts: dict[str, tuple[int, int]], first_stage: int = 4) -> list[str]:
    """Tasks that passed 1..first_stage-1 of their first-stage episodes (get +4 seeds)."""
    return [k for k, (p, n) in counts.items() if n == first_stage and 0 < p < first_stage]


@dataclass
class W0Summary:
    n_trials: int
    n_graded: int
    tgc: float | None
    sgc: float | None
    pass_band: tuple[float | None, int, int]
    wall_p50_s: float | None
    wall_p95_s: float | None
    mean_wall_s: float | None
    turns_p50: float | None
    status_counts: dict[str, int]
    n_turns_total: int
    parse_failure_rate: float | None
    parse_reason_counts: dict[str, int]
    subagent_calls: int
    trials_with_subagent: int
    session_resets: int
    prompt_tokens_p50: float | None
    prompt_tokens_p95: float | None
    prompt_tokens_by_turn_p50: dict[int, float] = field(default_factory=dict)
    completion_tokens_total: int = 0
    proxy_requests: int = 0
    proxy_ok_rate: float | None = None
    second_stage: list[str] = field(default_factory=list)


def status_of(t: TrialRecord) -> str:
    if t.reward is None:
        return "error" if t.exception else "ungraded"
    if t.passed:
        return "passed"
    if t.n_turns >= MAX_TURNS:
        return "failed_max_turns"
    return "failed"


def summarize(
    trials: Sequence[TrialRecord],
    proxy: Sequence[ProxyRecord],
    kill: KillConfig,
) -> W0Summary:
    w0 = kill.weekend_0
    counts = per_task_pass(trials)
    walls = [t.wall_seconds for t in trials if t.wall_seconds is not None]
    turns = [tr for t in trials for tr in t.turns]
    reasons = Counter(r for tr in turns for r in tr.parse_reasons)
    prompt = [p.prompt_tokens for p in proxy if p.ok and p.prompt_tokens is not None]
    by_turn: dict[int, list[int]] = defaultdict(list)
    for p in proxy:
        if p.ok and p.prompt_tokens is not None and p.live_turn_index >= 0:
            by_turn[p.live_turn_index].append(p.prompt_tokens)
    return W0Summary(
        n_trials=len(trials),
        n_graded=sum(t.reward is not None for t in trials),
        tgc=tgc(trials),
        sgc=sgc(trials),
        pass_band=pass_band_fraction(counts, w0.k1_pass_band, w0.k1_min_samples_per_task),
        wall_p50_s=_quantile(walls, 0.5),
        wall_p95_s=_quantile(walls, 0.95),
        mean_wall_s=statistics.fmean(walls) if walls else None,
        turns_p50=_quantile([t.n_turns for t in trials], 0.5),
        status_counts=dict(Counter(status_of(t) for t in trials)),
        n_turns_total=len(turns),
        parse_failure_rate=(sum(tr.parse_failure for tr in turns) / len(turns)) if turns else None,
        parse_reason_counts=dict(reasons),
        subagent_calls=sum(tr.subagent_call for tr in turns),
        trials_with_subagent=sum(any(tr.subagent_call for tr in t.turns) for t in trials),
        session_resets=sum(tr.session_reset for tr in turns),
        prompt_tokens_p50=_quantile(prompt, 0.5),
        prompt_tokens_p95=_quantile(prompt, 0.95),
        prompt_tokens_by_turn_p50={k: _quantile(v, 0.5) for k, v in sorted(by_turn.items())},
        completion_tokens_total=sum(p.completion_tokens or 0 for p in proxy),
        proxy_requests=len(proxy),
        proxy_ok_rate=(sum(p.ok for p in proxy) / len(proxy)) if proxy else None,
        second_stage=second_stage_tasks(counts),
    )


@dataclass
class KillResult:
    name: str
    value: Any
    threshold: Any
    passed: bool | None
    note: str = ""


def projected_cost_usd(mean_wall_s: float | None, kill: KillConfig) -> float | None:
    if mean_wall_s is None:
        return None
    c = kill.cost
    episodes = c.planned_q1_episodes + c.planned_q2_episodes
    gpu_hours = episodes * mean_wall_s / 3600.0 / max(c.episodes_per_gpu_concurrently, 1)
    return gpu_hours * c.gpu_usd_per_hour


def evaluate_kill(
    s: W0Summary,
    kill: KillConfig,
    *,
    is_k0_run: bool,
    k4_prompt_tokens: float | None = None,
    k4_note: str = "",
) -> list[KillResult]:
    w0 = kill.weekend_0
    out: list[KillResult] = []
    if is_k0_run:
        lo, hi = w0.k0_sgc_range
        out.append(KillResult("K0 9B B0 test_normal SGC", s.sgc, [lo, hi],
                              None if s.sgc is None else lo <= s.sgc <= hi))
    else:
        frac = s.pass_band[0]
        out.append(KillResult(
            "K1 pass-band fraction", frac, f">= {w0.k1_min_pass_band_fraction}",
            None if frac is None else frac >= w0.k1_min_pass_band_fraction,
            f"{s.pass_band[1]}/{s.pass_band[2]} tasks in {list(w0.k1_pass_band)} "
            f"with >= {w0.k1_min_samples_per_task} samples",
        ))
    r = s.parse_failure_rate
    out.append(KillResult("K2 parse-failure rate / turn", r, f"<= {w0.k2_max_parse_failure_rate}",
                          None if r is None else r <= w0.k2_max_parse_failure_rate))
    cost = projected_cost_usd(s.mean_wall_s, kill)
    out.append(KillResult("K3 projected Q1+Q2 cost (USD)", cost, f"<= {w0.k3_max_cost_usd}",
                          None if cost is None else cost <= w0.k3_max_cost_usd,
                          "wall-clock x [cost] inputs in config/kill.toml"))
    out.append(KillResult("K4 median prompt tokens @ first non-terminal guard", k4_prompt_tokens,
                          f"<= {w0.k4_max_prompt_tokens}",
                          None if k4_prompt_tokens is None
                          else k4_prompt_tokens <= w0.k4_max_prompt_tokens,
                          k4_note))
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)


def render_markdown(title: str, s: W0Summary, results: list[KillResult], meta: dict[str, Any]
                    ) -> str:
    lines = [f"### {title}", ""]
    lines += [f"- {k}: `{v}`" for k, v in meta.items()]
    lines += [
        f"- trials: {s.n_trials} (graded {s.n_graded}); status: {s.status_counts}",
        f"- TGC: {_fmt(s.tgc)}  SGC: {_fmt(s.sgc)}",
        f"- wall s p50/p95: {_fmt(s.wall_p50_s)} / {_fmt(s.wall_p95_s)}; turns p50: "
        f"{_fmt(s.turns_p50)}",
        f"- parse failures: {_fmt(s.parse_failure_rate)} over {s.n_turns_total} turns "
        f"{s.parse_reason_counts}",
        f"- subagent calls: {s.subagent_calls} (in {s.trials_with_subagent} trials); "
        f"session_reset: {s.session_resets}",
        f"- proxy: {s.proxy_requests} requests, ok {_fmt(s.proxy_ok_rate)}; prompt tokens "
        f"p50/p95: {_fmt(s.prompt_tokens_p50)} / {_fmt(s.prompt_tokens_p95)}; completion "
        f"tokens total: {s.completion_tokens_total}",
        f"- second-stage (+4 seed) tasks: {len(s.second_stage)}",
        "",
        "| criterion | value | threshold | pass |",
        "|---|---|---|---|",
    ]
    for r in results:
        verdict = {True: "PASS", False: "**FAIL**", None: "not measured"}[r.passed]
        note = f" ({r.note})" if r.note else ""
        lines.append(f"| {r.name}{note} | {_fmt(r.value)} | {r.threshold} | {verdict} |")
    return "\n".join(lines) + "\n"
