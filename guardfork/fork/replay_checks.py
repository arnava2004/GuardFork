"""Mandatory Weekend 1 replay validations (0 LLM calls each)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReconstructionReport:
    turns_checked: int
    canonical_equal: int
    raw_bytes_equal: int


def offline_request_reconstruction(trial_dir: Path) -> ReconstructionReport:
    """Rebuild LangChain messages at every turn, call
    `ReasoningChatOpenAI._get_request_payload`, and compare (canonical JSON, as in
    `build_sft.canonical`) against `student_model_requests.jsonl`. Must be 100% canonical-equal.
    """
    raise NotImplementedError("W1: replay_checks.offline_request_reconstruction")


@dataclass(frozen=True)
class RewardReplayReport:
    n: int
    reproduced: int
    diverged: list[str]


def full_replay_reward_parity(trial_dirs: list[Path]) -> RewardReplayReport:
    """Replay each complete trajectory (t = T + 1, no model calls) through
    ForkReplayAppWorldAgent and require the verifier to reproduce the original reward 20/20."""
    raise NotImplementedError("W1: replay_checks.full_replay_reward_parity")


def determinism_check(trial_dirs: list[Path], turns_per_trial: int = 3) -> float:
    """Fraction of (trial, turn) replays whose observations match exactly under
    PYTHONHASHSEED=0, excluding timeouts and subagent turns. Feeds W1-a (>= 0.85)."""
    raise NotImplementedError("W1: replay_checks.determinism_check")
