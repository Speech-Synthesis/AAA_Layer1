# =============================================================
# training/stability_head.py
# AAA Engine – Stability Head q(x)   [CUSTOM]
# =============================================================
#
# Implements the auxiliary stability head q(x) described in
# AAA-SV.md §4, §6, §7, §12.
#
# Architecture (spec §4):
#   Input  : pooled stats vector, shape (B, pool_dim)  — DETACHED
#   Layers : Linear(pool_dim, 256) → ReLU → Linear(256, 1) → Sigmoid
#   Output : (B,) ∈ [0, 1]  — per-utterance stability score
#
# Target (spec §7):
#   y_q = clamp( cosine_similarity( e(x_deg),
#                                   stop_grad(e(x_ref)) ), 0, 1 )
#   where x_ref is a clean, longer view (up to q_ref_max_duration s)
#   and   x_deg is the degraded / shorter training view.
#
# Loss (spec §12):
#   L_q = MSE( q(stop_grad(pooled_deg)), y_q )
#   L   = L_AAM + lambda_q * L_q        (joint loss)
#
# Design constraints:
#   • q head is a SEPARATE module — B1 checkpoints load unchanged.
#   • No gradients flow through the backbone when computing L_q:
#     both the pooled-stats input and the reference embedding are
#     detached before entering this module.
# =============================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# 1.  Configuration dataclass
# ═══════════════════════════════════════════════════════════════

@dataclass
class StabilityHeadConfig:
    """
    Hyper-parameters for the Stability Head.

    Attributes
    ----------
    pool_dim           : Dimensionality of the pooled-stats vector fed
                         into the head.  For ECAPA-TDNN + ASP this is
                         6144 (3072 mean + 3072 std, spec §6).
    hidden_dim         : Width of the single hidden layer (spec §4: 256).
    lambda_q           : Weight of L_q in the total loss (spec §12: 0.1).
    q_ref_max_duration : Maximum duration (seconds) for the clean
                         reference view x_ref (spec §12: 6.0).
    """
    pool_dim: int           = 6144
    hidden_dim: int         = 256
    lambda_q: float         = 0.1
    q_ref_max_duration: float = 6.0


# ═══════════════════════════════════════════════════════════════
# 2.  Stability Head module
# ═══════════════════════════════════════════════════════════════

class StabilityHead(nn.Module):
    """
    [CUSTOM] Auxiliary stability head q(x).

    Takes a **detached** pooled-stats vector and predicts a scalar
    stability score in [0, 1].  The module deliberately has no
    connection to backbone parameters — callers MUST pass
    ``pooled_stats.detach()`` as required by spec §4 and §12.

    Parameters
    ----------
    config : StabilityHeadConfig (or None to use defaults)
    """

    def __init__(self, config: Optional[StabilityHeadConfig] = None) -> None:
        super().__init__()
        self.config = config or StabilityHeadConfig()
        pool_dim   = self.config.pool_dim
        hidden_dim = self.config.hidden_dim

        self.net = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    # ------------------------------------------------------------------
    def forward(self, pooled_stats_detached: torch.Tensor) -> torch.Tensor:
        """
        Compute per-utterance stability scores.

        Parameters
        ----------
        pooled_stats_detached : Tensor of shape (B, pool_dim).
                                **Must be detached from the backbone
                                computation graph.**

        Returns
        -------
        q : Tensor of shape (B,) with values in [0, 1].
        """
        # Safety guard: refuse to propagate gradients to backbone.
        # This is belt-and-suspenders — callers should detach first.
        if pooled_stats_detached.requires_grad:
            pooled_stats_detached = pooled_stats_detached.detach()

        q = self.net(pooled_stats_detached)   # (B, 1)
        return q.squeeze(-1)                   # (B,)

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        c = self.config
        return (
            f"pool_dim={c.pool_dim}, hidden_dim={c.hidden_dim}, "
            f"lambda_q={c.lambda_q}"
        )


# ═══════════════════════════════════════════════════════════════
# 3.  Target and loss helpers
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def stability_target(
    emb_deg: torch.Tensor,
    emb_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Compute y_q = clamp(cosine_similarity(e_deg, stop_grad(e_ref)), 0, 1).

    Both inputs are L2-normalised internally, so raw embedding tensors
    (before or after normalisation) are acceptable.

    ``emb_ref`` is treated as stop-gradient here (the decorator
    ``@torch.no_grad()`` prevents any gradient flow; callers may also
    pass ``emb_ref.detach()`` for clarity).

    Parameters
    ----------
    emb_deg : (B, D) — degraded-view embeddings
    emb_ref : (B, D) — clean-reference embeddings (stop-grad)

    Returns
    -------
    y_q : (B,) in [0, 1]
    """
    e_deg = F.normalize(emb_deg.detach(), dim=1)
    e_ref = F.normalize(emb_ref.detach(), dim=1)
    cos_sim = (e_deg * e_ref).sum(dim=1)          # (B,)
    return cos_sim.clamp(0.0, 1.0)


def stability_loss(
    q: torch.Tensor,
    y_q: torch.Tensor,
) -> torch.Tensor:
    """
    L_q = MSE(q, y_q)   (spec §12)

    Parameters
    ----------
    q   : (B,) — predicted stability scores from StabilityHead.forward()
    y_q : (B,) — target scores from stability_target()

    Returns
    -------
    scalar loss tensor (gradient-enabled through q only)
    """
    return F.mse_loss(q, y_q.detach())


def joint_loss(
    l_aam: torch.Tensor,
    l_q: torch.Tensor,
    lambda_q: float = 0.1,
) -> torch.Tensor:
    """
    L = L_AAM + lambda_q * L_q   (spec §12)

    Parameters
    ----------
    l_aam    : AAM-Softmax loss tensor
    l_q      : stability MSE loss tensor
    lambda_q : weighting coefficient (default 0.1 per spec)

    Returns
    -------
    scalar joint loss tensor
    """
    return l_aam + lambda_q * l_q
