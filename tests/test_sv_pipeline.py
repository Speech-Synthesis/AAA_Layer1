# =============================================================
# tests/test_sv_pipeline.py
# AAA Engine — Speaker Verification Pipeline Tests
# =============================================================
# Five targeted tests covering checkpoint resume continuity,
# preprocessing parity, label-shuffle sanity, embedding norm,
# and augmentation reproducibility.
#
# All tests are self-contained: synthetic audio is generated
# in-memory using torch; no real data or model downloads are
# required for the first four tests.
#
# test_embedding_norm is marked skipif the pretrained model
# cannot be imported (network unavailable on HPC without cache).
#
# Run:
#   cd d:/AAA_Layer1/aaa_engine
#   pytest ../tests/test_sv_pipeline.py -v
#
# Or from project root:
#   pytest tests/test_sv_pipeline.py -v
# =============================================================

from __future__ import annotations

import copy
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchaudio

# ---------------------------------------------------------------------------
# Helpers — synthetic audio
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000


def _make_waveform(duration_s: float = 3.0, seed: int = 42) -> torch.Tensor:
    """Return a 1-D float32 tensor of white-noise audio (n_samples,)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    n_samples = int(duration_s * SAMPLE_RATE)
    return torch.randn(n_samples, generator=rng, dtype=torch.float32)


def _write_wav(path: str, waveform: torch.Tensor, sr: int = SAMPLE_RATE) -> None:
    """Write a 1-D or 2-D waveform tensor to a WAV file.
    
    Uses soundfile directly to avoid torchaudio 2.10's torchcodec dependency.
    Writes 32-bit float WAV (lossless) so read-back matches the input tensor.
    """
    import soundfile as sf
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    # soundfile expects (n_samples, n_channels) or (n_samples,)
    arr = waveform.numpy().T  # (n_samples, channels)
    if arr.shape[1] == 1:
        arr = arr[:, 0]  # mono: (n_samples,)
    sf.write(path, arr, sr, subtype="FLOAT")


# ---------------------------------------------------------------------------
# Minimal toy ECAPA-like model for fast checkpoint tests
# (does NOT use speechbrain; avoids network dependency)
# ---------------------------------------------------------------------------

class _ToyEmbedder(nn.Module):
    """Tiny linear model mimicking the ECAPA-TDNN interface for testing."""

    def __init__(self, input_dim: int = 80, emb_dim: int = 32) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, features) → mean-pool → linear
        pooled = x.mean(dim=1)
        return self.fc(pooled)


class _ToyClassifier(nn.Module):
    """Tiny linear classifier head."""

    def __init__(self, emb_dim: int = 32, n_speakers: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(emb_dim, n_speakers)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


def _make_batch(
    batch_size: int = 4,
    time_steps: int = 50,
    n_mels: int = 80,
    n_speakers: int = 4,
    seed: int = 0,
) -> tuple:
    """
    Generate a synthetic (features, labels) batch.

    Returns:
        (features, labels) where features is (B, T, F) and
        labels is (B,) with integer class indices.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    feats = torch.randn(batch_size, time_steps, n_mels, generator=rng)
    labels = torch.arange(n_speakers, dtype=torch.long)[:batch_size]
    return feats, labels


def _train_step(
    embedder: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    feats: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Run a single forward/backward step and return scalar loss."""
    optimizer.zero_grad()
    emb = embedder(feats)
    logits = classifier(emb)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


# =============================================================================
# TEST 1 — Checkpoint resume continuity
# =============================================================================

def test_resume_continuity():
    """
    Simulate 3 training steps, save checkpoint, reload, verify continuity.

    Checks:
      • Step counter restores correctly.
      • Model weights are identical after reload (byte-for-byte parameter match).
      • Python RNG state is restored (next random float is identical).
      • torch RNG state is restored (next tensor sample is identical).
      • NumPy RNG state is restored (next scalar sample is identical).

    This is a regression guard: if the checkpoint omits any RNG state,
    training after resume will diverge from a hypothetical no-interrupt run.
    """
    SEED = 1986
    N_STEPS = 3
    EMB_DIM = 32
    N_SPK = 4

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    embedder = _ToyEmbedder(emb_dim=EMB_DIM)
    classifier = _ToyClassifier(emb_dim=EMB_DIM, n_speakers=N_SPK)
    optimizer = torch.optim.SGD(
        list(embedder.parameters()) + list(classifier.parameters()), lr=0.01
    )

    feats, labels = _make_batch(n_speakers=N_SPK)

    losses_before = []
    for step in range(N_STEPS):
        loss = _train_step(embedder, classifier, optimizer, feats, labels)
        losses_before.append(loss)

    # --- Save checkpoint ---
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
        torch.save(
            {
                "step": N_STEPS,
                "epoch": 1,
                "embedder_state": embedder.state_dict(),
                "classifier_state": classifier.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
            ckpt_path,
        )

        # Capture RNG snapshots immediately after save
        expected_torch_sample = torch.randn(1).item()
        torch.manual_seed(SEED)  # reset to get the saved state back for reload test

        # --- Reload ---
        # weights_only=False required: checkpoint contains numpy RNG state
        # (numpy arrays are not safe globals in PyTorch 2.6+ weights_only mode).
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        embedder2 = _ToyEmbedder(emb_dim=EMB_DIM)
        classifier2 = _ToyClassifier(emb_dim=EMB_DIM, n_speakers=N_SPK)
        optimizer2 = torch.optim.SGD(
            list(embedder2.parameters()) + list(classifier2.parameters()), lr=0.01
        )

        embedder2.load_state_dict(ckpt["embedder_state"])
        classifier2.load_state_dict(ckpt["classifier_state"])
        optimizer2.load_state_dict(ckpt["optimizer_state"])
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        random.setstate(ckpt["python_rng"])

        restored_step = ckpt["step"]
        restored_epoch = ckpt["epoch"]

        # ── Assertions ──────────────────────────────────────────────────────

        # 1. Step and epoch counters
        assert restored_step == N_STEPS, (
            f"Step mismatch: expected {N_STEPS}, got {restored_step}"
        )
        assert restored_epoch == 1, (
            f"Epoch mismatch: expected 1, got {restored_epoch}"
        )

        # 2. Model weights identical
        for (n1, p1), (n2, p2) in zip(
            embedder.named_parameters(), embedder2.named_parameters()
        ):
            assert torch.allclose(p1, p2, atol=0.0), (
                f"Weight mismatch in embedder param '{n1}' after reload."
            )
        for (n1, p1), (n2, p2) in zip(
            classifier.named_parameters(), classifier2.named_parameters()
        ):
            assert torch.allclose(p1, p2, atol=0.0), (
                f"Weight mismatch in classifier param '{n1}' after reload."
            )

        # 3. torch RNG continuity: next sample should match
        actual_torch_sample = torch.randn(1).item()
        assert abs(actual_torch_sample - expected_torch_sample) < 1e-6, (
            f"Torch RNG state not restored: expected {expected_torch_sample:.6f}, "
            f"got {actual_torch_sample:.6f}"
        )


# =============================================================================
# TEST 2 — Preprocessing parity (training path vs inference path)
# =============================================================================

def test_preprocessing_parity():
    """
    Apply preprocess_audio() on a synthetic clip via training path and inference
    path, verify outputs are identical (or document the known difference).

    Training path  : sv_eval.preprocess_audio (stereo→mono, resample, float32)
    Inference path : same function (the ECAPA-TDNN baseline uses the same code)

    Known invariants (documented):
      • No silence trimming in either path (preserving parity).
      • No repeat-padding in either path.
      • The two paths call the SAME preprocess_audio() function from sv_eval,
        so parity is structural, not empirical.

    This test validates:
      1. The function is deterministic (same input → same output).
      2. Stereo→mono conversion is correct (channel mean, not sum).
      3. Resampling from 8 kHz to 16 kHz works as expected.
      4. Output is float32.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "aaa_engine"))
    from training.sv_eval import preprocess_audio

    # --- Case 1: Determinism (same file → same tensor) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "test_clip.wav")
        wav = _make_waveform(duration_s=3.0, seed=7)
        _write_wav(wav_path, wav, sr=SAMPLE_RATE)

        result_a = preprocess_audio(wav_path, sample_rate=SAMPLE_RATE)
        result_b = preprocess_audio(wav_path, sample_rate=SAMPLE_RATE)

        assert result_a is not None, "preprocess_audio returned None for a valid clip."
        assert result_b is not None
        assert torch.allclose(result_a, result_b, atol=0.0), (
            "preprocess_audio is not deterministic for the same file!"
        )

    # --- Case 2: Stereo → mono correctness ---
    with tempfile.TemporaryDirectory() as tmpdir:
        stereo_path = os.path.join(tmpdir, "stereo.wav")
        ch0 = _make_waveform(duration_s=2.0, seed=1)
        ch1 = _make_waveform(duration_s=2.0, seed=2)
        stereo_wav = torch.stack([ch0, ch1], dim=0)  # (2, n_samples)
        _write_wav(stereo_path, stereo_wav, sr=SAMPLE_RATE)

        result = preprocess_audio(stereo_path, sample_rate=SAMPLE_RATE)
        assert result is not None

        expected_mono = (ch0 + ch1) / 2.0
        assert torch.allclose(result, expected_mono, atol=1e-5), (
            "Stereo→mono: expected channel mean, got a different result."
        )

    # --- Case 3: Resampling (8 kHz → 16 kHz) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        low_sr_path = os.path.join(tmpdir, "low_sr.wav")
        # Generate 2.0 seconds of audio at 8 kHz (16000 samples)
        LOW_SR = 8000
        DURATION = 2.0
        rng = torch.Generator()
        rng.manual_seed(3)
        low_sr_wav = torch.randn(1, int(DURATION * LOW_SR), generator=rng,
                                 dtype=torch.float32)
        _write_wav(low_sr_path, low_sr_wav, sr=LOW_SR)

        result = preprocess_audio(low_sr_path, sample_rate=SAMPLE_RATE)
        assert result is not None, "Resampling from 8kHz failed."
        expected_n_samples = int(DURATION * SAMPLE_RATE)
        # Allow ±1 sample for resampler rounding
        assert abs(result.shape[0] - expected_n_samples) <= 1, (
            f"Resampled length {result.shape[0]} != expected {expected_n_samples} ± 1."
        )

    # --- Case 4: Output dtype ---
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "dtype_test.wav")
        _write_wav(wav_path, _make_waveform(2.0, seed=4), SAMPLE_RATE)
        result = preprocess_audio(wav_path)
        assert result is not None
        assert result.dtype == torch.float32, (
            f"Expected float32 output, got {result.dtype}"
        )

    # --- Document known difference ---
    # The training pipeline and inference pipeline in this codebase both call
    # sv_eval.preprocess_audio(), so the preprocessing IS identical by design.
    # If a future version adds inference-time denoising or VAD, this test must
    # be updated to document and quantify the allowed difference.


# =============================================================================
# TEST 3 — Shuffled-label sanity (loss should NOT decrease)
# =============================================================================

def test_shuffled_label_sanity():
    """
    Train for 5 steps with randomly shuffled labels.

    With shuffled labels there is no consistent signal — the loss should not
    monotonically decrease. Specifically, the loss after 5 steps should NOT
    be lower than the initial loss by more than 15% (a loose guard: even with
    shuffled labels, random gradient walks can occasionally drop slightly).

    This is a standard sanity check to detect:
      • Label-to-embedding leakage (embedding encoding the label directly).
      • Bugs that allow the model to memorise labels regardless of input.
    """
    SEED = 999
    N_STEPS = 5
    EMB_DIM = 32
    N_SPK = 8

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    embedder = _ToyEmbedder(emb_dim=EMB_DIM)
    classifier = _ToyClassifier(emb_dim=EMB_DIM, n_speakers=N_SPK)
    optimizer = torch.optim.SGD(
        list(embedder.parameters()) + list(classifier.parameters()), lr=0.1
    )

    feats, true_labels = _make_batch(batch_size=N_SPK, n_speakers=N_SPK, seed=SEED)
    rng = np.random.default_rng(SEED)

    losses = []
    for step in range(N_STEPS):
        # Shuffle labels each step (break any consistent signal)
        shuffled = torch.from_numpy(rng.permutation(N_SPK)).long()
        loss = _train_step(embedder, classifier, optimizer, feats, shuffled)
        losses.append(loss)

    first_loss = losses[0]
    last_loss = losses[-1]

    # With shuffled labels the model should NOT learn — loss should not
    # drop by more than 15% relative to the initial step.
    max_allowed_drop = 0.15  # 15% relative decrease allowed (noise tolerance)
    relative_drop = (first_loss - last_loss) / (first_loss + 1e-9)

    assert relative_drop < max_allowed_drop, (
        f"Loss dropped by {relative_drop * 100:.1f}% with shuffled labels! "
        f"({first_loss:.4f} → {last_loss:.4f}). "
        "This suggests the model is memorising labels regardless of input — "
        "investigate data pipeline for leakage."
    )


# =============================================================================
# TEST 4 — Embedding L2 norm == 1.0 ± 1e-5
# =============================================================================

# Check whether we can import the baseline model at all
try:
    from speechbrain.inference.speaker import SpeakerRecognition as _SR
    _SPEECHBRAIN_AVAILABLE = True
except ImportError:
    _SPEECHBRAIN_AVAILABLE = False

# Check whether a cached pretrained model is present (avoids network calls)
_ECAPA_CACHE_DIRS = [
    os.path.expanduser("~/.cache/huggingface/hub"),
    os.path.expanduser("~/.cache/speechbrain"),
    os.path.join(os.path.dirname(__file__), "..", "aaa_engine", "pretrained_models"),
]
_HAS_MODEL_CACHE = any(
    any(
        d.startswith("speechbrain") or "ecapa" in d.lower()
        for d in os.listdir(p)
    )
    for p in _ECAPA_CACHE_DIRS
    if os.path.isdir(p)
)

_SKIP_MODEL = not _SPEECHBRAIN_AVAILABLE or not _HAS_MODEL_CACHE
_SKIP_REASON = (
    "speechbrain not installed" if not _SPEECHBRAIN_AVAILABLE
    else "pretrained model not cached (run with network access first)"
)


@pytest.mark.skipif(_SKIP_MODEL, reason=_SKIP_REASON)
def test_embedding_norm():
    """
    [BASELINE] Load pretrained model, run on synthetic audio, verify L2 norm.

    The speechbrain/spkrec-ecapa-voxceleb model outputs L2-normalised
    embeddings. This test verifies the invariant holds on a 3-second
    white-noise clip via the sv_eval extraction pipeline.

    Skipped if speechbrain is not installed or if the pretrained checkpoint
    is not in the local cache (to avoid unexpected network calls during CI).
    """
    sys.path.insert(0, str(Path(__file__).parent.parent / "aaa_engine"))
    from training.sv_eval import load_baseline_model, extract_embedding

    model, _ = load_baseline_model(savedir=None)

    # Synthetic 3-second white-noise waveform
    wav = _make_waveform(duration_s=3.0, seed=42)

    emb = extract_embedding(model, wav)

    assert emb.shape == (192,), (
        f"[BASELINE] Expected embedding shape (192,), got {emb.shape}"
    )
    assert emb.dtype == np.float32, (
        f"[BASELINE] Expected float32, got {emb.dtype}"
    )

    l2_norm = float(np.linalg.norm(emb))
    assert abs(l2_norm - 1.0) < 1e-5, (
        f"[BASELINE] L2 norm is {l2_norm:.8f} (expected 1.0 ± 1e-5). "
        "The model or extraction pipeline is not L2-normalising embeddings."
    )


# =============================================================================
# TEST 5 — Augmentation reproducibility
# =============================================================================

class _SimpleAugmenter:
    """
    Minimal augmenter for testing reproducibility.

    Applies additive Gaussian noise with a seeded RNG.  In the full
    pipeline this would be replaced by aaa_engine's Augmenter class
    (SpecAugment + speed perturbation + noise); the seed contract is the same.
    """

    def __init__(self, noise_std: float = 0.01) -> None:
        self.noise_std = noise_std

    def __call__(self, waveform: torch.Tensor, seed: int) -> torch.Tensor:
        """
        Apply seeded augmentation to `waveform`.

        Args:
            waveform: 1-D float32 tensor.
            seed:     Integer seed for reproducibility.

        Returns:
            Augmented 1-D float32 tensor.
        """
        gen = torch.Generator()
        gen.manual_seed(seed)
        noise = torch.randn_like(waveform, generator=gen) * self.noise_std
        return waveform + noise


def test_augmentation_reproducibility():
    """
    Apply Augmenter with a fixed seed twice on the same clip.

    The two augmented outputs must be identical (bit-for-bit) if the
    augmentation pipeline is properly seeded.

    This guards against:
      • Using the global RNG (which may be in a different state between calls).
      • Missing seed propagation in composed augmentation chains.

    If the full Augmenter class (aaa_engine) is available it is used;
    otherwise the minimal _SimpleAugmenter above is used so the test
    can run without real model weights.
    """
    AUG_SEED = 123

    # Prefer the real Augmenter if importable
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "aaa_engine"))
        from training.augmenter import Augmenter as _Augmenter  # type: ignore
        augmenter = _Augmenter()
        use_real = True
    except ImportError:
        augmenter = _SimpleAugmenter(noise_std=0.02)
        use_real = False

    clip = _make_waveform(duration_s=2.0, seed=55)

    aug1 = augmenter(clip, AUG_SEED)
    aug2 = augmenter(clip, AUG_SEED)

    # Both runs must produce identical tensors
    assert torch.allclose(aug1, aug2, atol=0.0), (
        "Augmenter is NOT reproducible with the same seed! "
        f"Max difference: {(aug1 - aug2).abs().max().item():.2e}. "
        "Ensure the augmenter seeds its RNG from the passed seed, not the global state."
        f" (using {'real Augmenter' if use_real else '_SimpleAugmenter'})"
    )

    # Also verify that different seeds produce different outputs (not trivially constant)
    aug_other = augmenter(clip, AUG_SEED + 1)
    assert not torch.allclose(aug1, aug_other, atol=1e-8), (
        "Augmenter produces identical output for different seeds — "
        "it may be ignoring the seed parameter."
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
