"""Turn-level preference losses. Plain DPO is primary; length-normalized is an ablation."""

from __future__ import annotations

import math


def dpo_loss(policy_chosen_logp: float, policy_rejected_logp: float,
             ref_chosen_logp: float, ref_rejected_logp: float, *, beta: float = 0.1,
             chosen_len: int | None = None, rejected_len: int | None = None,
             length_normalize: bool = False) -> float:
    """-log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))). Plain DPO is primary;
    `length_normalize` divides each log-ratio by its token count (ablation)."""
    chosen = policy_chosen_logp - ref_chosen_logp
    rejected = policy_rejected_logp - ref_rejected_logp
    if length_normalize:
        if not chosen_len or not rejected_len:
            raise ValueError("length_normalize requires positive chosen_len and rejected_len")
        chosen /= chosen_len
        rejected /= rejected_len
    return -math.log(1.0 / (1.0 + math.exp(-beta * (chosen - rejected))))
