# =============================================================
# tests/test_stability_head.py
# AAA Engine – Unit tests for the Stability Head q(x)
# =============================================================
#
# Covers spec requirements (AAA-SV.md §4, §6, §7, §12):
#   1. test_q_output_range          — q ∈ [0, 1]
#   2. test_no_backbone_gradient    — backbone grad is None after L_q.backward()
#   3. test_no_gradient_through_target — y_q is stop-grad (no grad to emb_ref)
#   4. test_save_load_resume        — full save/load preserves weights + optim
#   5. test_variable_length_input   — works with different batch sizes / pool_dim
#   6. test_q_mse_loss_zero         — loss == 0 when q exactly equals y_q
# =============================================================

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from aaa_engine.training.stability_head import (
    StabilityHead,
    StabilityHeadConfig,
    joint_loss,
    stability_loss,
    stability_target,
)


# ───────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> StabilityHeadConfig:
    return StabilityHeadConfig(pool_dim=6144, hidden_dim=256, lambda_q=0.1)


@pytest.fixture
def small_config() -> StabilityHeadConfig:
    """Smaller config for fast tests (avoids allocating 6144-dim tensors)."""
    return StabilityHeadConfig(pool_dim=64, hidden_dim=32, lambda_q=0.1)


@pytest.fixture
def q_head_small(small_config) -> StabilityHead:
    return StabilityHead(config=small_config)


# ───────────────────────────────────────────────────────────────
# 1. Output range ∈ [0, 1]
# ───────────────────────────────────────────────────────────────

def test_q_output_range(small_config):
    """q(x) must be in [0, 1] for any input (Sigmoid guarantee)."""
    head = StabilityHead(config=small_config)
    B = 16
    # Random large-magnitude inputs to stress the sigmoid boundary
    x = torch.randn(B, small_config.pool_dim) * 100.0
    with torch.no_grad():
        q = head(x)
    assert q.shape == (B,), f"Expected shape ({B},), got {q.shape}"
    assert q.min().item() >= 0.0, f"q.min()={q.min().item()} < 0"
    assert q.max().item() <= 1.0, f"q.max()={q.max().item()} > 1"


# ───────────────────────────────────────────────────────────────
# 2. No gradient flows to backbone through L_q
# ───────────────────────────────────────────────────────────────

def test_no_backbone_gradient(small_config):
    """
    Spec §4, §12: backbone gradients must be None when L_q backward is called.

    Simulates the training loop:
      backbone → pooled_stats (detached) → q_head → L_q → backward
    """
    # Minimal backbone proxy (single linear layer)
    backbone = nn.Linear(small_config.pool_dim, small_config.pool_dim)
    for p in backbone.parameters():
        p.requires_grad_(True)

    head = StabilityHead(config=small_config)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.01)

    B = 4
    x = torch.randn(B, small_config.pool_dim)

    # Simulate backbone forward
    pooled = backbone(x)   # has grad_fn
    # Detach — this is what train_aaa_sv.py does
    q = head(pooled.detach())

    y_q = torch.rand(B)    # dummy target
    loss = stability_loss(q, y_q)
    optimizer.zero_grad()
    loss.backward()

    # Backbone parameters must have no gradient
    for name, param in backbone.named_parameters():
        assert param.grad is None, (
            f"Backbone param '{name}' has gradient after L_q.backward() — "
            "detach is not working correctly."
        )


# ───────────────────────────────────────────────────────────────
# 3. stability_target is stop-grad (no gradient to emb_ref)
# ───────────────────────────────────────────────────────────────

def test_no_gradient_through_target():
    """
    Spec §7: y_q = stop_grad(cosine_sim(e_deg, e_ref))
    Neither emb_deg nor emb_ref should receive gradients through y_q.
    """
    B, D = 8, 192
    emb_deg = torch.randn(B, D, requires_grad=True)
    emb_ref = torch.randn(B, D, requires_grad=True)

    # stability_target is decorated @torch.no_grad() — y_q has no grad_fn
    y_q = stability_target(emb_deg, emb_ref)

    assert not y_q.requires_grad, "y_q must not require grad (stop-grad target)"
    assert y_q.grad_fn is None, "y_q must have no grad_fn"

    # Range check
    assert y_q.min().item() >= 0.0
    assert y_q.max().item() <= 1.0


# ───────────────────────────────────────────────────────────────
# 4. Save / load / resume — weights + optimizer state preserved
# ───────────────────────────────────────────────────────────────

def test_save_load_resume(tmp_path, small_config):
    """
    Full save/load cycle must reproduce identical forward output.
    The q_head optimizer state must also survive the round-trip.
    """
    head_orig = StabilityHead(config=small_config)
    optimizer = torch.optim.AdamW(head_orig.parameters(), lr=1e-3)

    # Simulate one optimiser step to populate optimizer state
    B = 4
    x = torch.randn(B, small_config.pool_dim)
    q = head_orig(x.detach())
    loss = stability_loss(q, torch.rand(B))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save
    ckpt_path = tmp_path / "checkpoint_test.pt"
    state = {
        "q_head":    head_orig.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, ckpt_path)

    # Load into a fresh head + optimizer
    head_loaded = StabilityHead(config=small_config)
    optimizer2  = torch.optim.AdamW(head_loaded.parameters(), lr=1e-3)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    head_loaded.load_state_dict(ckpt["q_head"])
    optimizer2.load_state_dict(ckpt["optimizer"])

    # Forward outputs must match exactly
    with torch.no_grad():
        q_orig   = head_orig(x.detach())
        q_loaded = head_loaded(x.detach())
    assert torch.allclose(q_orig, q_loaded, atol=1e-6), (
        f"q_head output mismatch after save/load. "
        f"max_diff={( q_orig - q_loaded).abs().max().item():.2e}"
    )

    # Optimizer state round-trips without error (checked implicitly above)


# ───────────────────────────────────────────────────────────────
# 5. Works with different input shapes (variable batch / pool_dim)
# ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("batch_size,pool_dim", [
    (1,   64),
    (4,   64),
    (32,  128),
    (1,   6144),
])
def test_variable_length_input(batch_size, pool_dim):
    """StabilityHead must handle various batch sizes and pool dims."""
    config = StabilityHeadConfig(pool_dim=pool_dim, hidden_dim=32)
    head   = StabilityHead(config=config)
    x = torch.randn(batch_size, pool_dim)
    with torch.no_grad():
        q = head(x)
    assert q.shape == (batch_size,)
    assert (q >= 0).all() and (q <= 1).all()


# ───────────────────────────────────────────────────────────────
# 6. stability_loss == 0 when q exactly equals y_q
# ───────────────────────────────────────────────────────────────

def test_q_mse_loss_zero(small_config):
    """
    Spec §12: L_q = MSE(q, y_q).
    When predicted == target, loss must be 0 (or machine-epsilon close).
    """
    head = StabilityHead(config=small_config)
    B = 8
    # Build a fixed target in [0, 1]
    y_q = torch.rand(B)

    # Manually set q_head weights to output y_q exactly is not feasible,
    # so instead we pass y_q as the predicted value directly to test the
    # loss formula itself.
    loss = stability_loss(y_q, y_q)
    assert loss.item() < 1e-12, (
        f"stability_loss(y_q, y_q) should be ~0, got {loss.item():.2e}"
    )


# ───────────────────────────────────────────────────────────────
# 7. joint_loss combines L_AAM and L_q correctly
# ───────────────────────────────────────────────────────────────

def test_joint_loss_formula():
    """L = L_AAM + lambda_q * L_q  (spec §12)."""
    l_aam = torch.tensor(2.0)
    l_q   = torch.tensor(1.0)
    lambda_q = 0.1
    expected = 2.0 + 0.1 * 1.0  # 2.1
    result   = joint_loss(l_aam, l_q, lambda_q=lambda_q)
    assert abs(result.item() - expected) < 1e-6, (
        f"joint_loss expected {expected}, got {result.item()}"
    )
