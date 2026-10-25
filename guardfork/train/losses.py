"""Turn-level preference losses. HAND-WRITTEN BY ARNAV (AGENTS.md rule 1).

Framework-free signatures for now; the training backend is chosen in Weekend 5.
"""

from __future__ import annotations


def dpo_loss(policy_chosen_logp: float, policy_rejected_logp: float,
             ref_chosen_logp: float, ref_rejected_logp: float, *, beta: float = 0.1,
             chosen_len: int | None = None, rejected_len: int | None = None,
             length_normalize: bool = False) -> float:
    """-log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))). Plain DPO is primary;
    `length_normalize` divides each log-ratio by its token count (ablation)."""
    raise NotImplementedError("Arnav: train/losses.py")
