# =============================================================
# tests/test_aaasv.py
# AAA Engine – Unit tests for AAASV inference module
# =============================================================
#
# Tests are fully self-contained: no real model checkpoint is loaded.
# A _StubAAASV replaces AAASV so only the validation / data-flow logic
# is exercised.  The stub uses a seeded RNG for 192-dim embeddings.
#
# Tests:
#   1. test_embed_too_short            — wav < 1.0 s → E_TOO_SHORT
#   2. test_embed_too_long             — wav > 30.0 s → E_TOO_LONG
#   3. test_embed_silent               — RMS < -60 dBFS → E_SILENT
#   4. test_embed_stereo_converts_to_mono — stereo input does NOT error
#   5. test_embed_1s_2s_w_short_warning   — 1.0–2.0 s → ok=True + W_SHORT
#   6. test_response_envelope_fields   — AAASVResponse has all required fields
#   7. test_verify_result_fields       — VerifyResult has all required fields
#   8. test_maybe_update_gates         — update fires iff score>0.85 and q>0.7
# =============================================================

from __future__ import annotations

import sys
import os
import time
import importlib
from datetime import datetime, timezone

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable (mirrors conftest.py behaviour)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Import production dataclasses and error codes only (no model loading needed)
# ---------------------------------------------------------------------------
from aaa_engine.inference.aaasv import (
    AAASVResponse,
    EmbedResult,
    Voiceprint,
    VerifyResult,
    E_TOO_SHORT,
    E_TOO_LONG,
    E_SILENT,
    W_SHORT,
    _rms_dbfs,
    _l2_norm,
    _cosine_similarity,
    _confidence_from,
    AAASV,          # imported to call unbound _validate_and_preprocess
)

# Keep a reference to the module for helper access
_aaasv_mod = importlib.import_module("aaa_engine.inference.aaasv")

_TARGET_SR = 16_000
_EMB_DIM   = 192


# ═══════════════════════════════════════════════════════════════
# Stub AAASV — avoids any file I/O or model loading
# ═══════════════════════════════════════════════════════════════

class _StubAAASV:
    """
    Minimal stub that passes all input validation but uses a seeded RNG
    for 192-dim embeddings instead of a real model forward pass.

    It reuses the production _validate_and_preprocess() method via
    an unbound call so validation behaviour is identical to the real class.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = threshold
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    def _fake_embedding(self):
        """Return (emb_192, q_score=0.8)."""
        emb = self._rng.standard_normal(_EMB_DIM).astype(np.float32)
        return _l2_norm(emb), 0.8

    # ------------------------------------------------------------------
    def embed(self, wav: np.ndarray, sr: int) -> AAASVResponse:
        t0 = time.perf_counter()
        # Call the production validation method (unbound)
        processed, err, warnings = AAASV._validate_and_preprocess(self, wav, sr)
        lat = (time.perf_counter() - t0) * 1000.0
        if err is not None:
            return AAASVResponse(ok=False, error_code=err, result=None,
                                 warnings=warnings, latency_ms=lat)
        duration_s = len(processed) / float(_TARGET_SR)
        emb, q = self._fake_embedding()
        return AAASVResponse(
            ok=True,
            error_code=None,
            result=EmbedResult(
                embedding=emb,
                q_score=q,
                duration_s=duration_s,
                sample_rate=_TARGET_SR,
            ),
            warnings=warnings,
            latency_ms=lat,
        )

    # ------------------------------------------------------------------
    def enroll(self, wavs, srs, speaker_id):
        t0 = time.perf_counter()
        embs, qs, all_w = [], [], []
        for wav, sr in zip(wavs, srs):
            r = self.embed(wav, sr)
            for w in r.warnings:
                if w not in all_w:
                    all_w.append(w)
            if not r.ok:
                return AAASVResponse(ok=False, error_code=r.error_code,
                                     result=None, warnings=all_w,
                                     latency_ms=(time.perf_counter() - t0) * 1000)
            embs.append(r.result.embedding)
            if r.result.q_score is not None:
                qs.append(r.result.q_score)
        mean_emb = _l2_norm(np.mean(np.stack(embs), axis=0))
        now = datetime.now(timezone.utc).isoformat()
        vp = Voiceprint(speaker_id=speaker_id, embedding=mean_emb,
                        n_utterances=len(embs), created_at=now,
                        updated_at=now,
                        mean_q=float(np.mean(qs)) if qs else None)
        return AAASVResponse(ok=True, error_code=None, result=vp,
                             warnings=all_w,
                             latency_ms=(time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------
    def verify(self, wav, sr, voiceprint):
        t0 = time.perf_counter()
        r = self.embed(wav, sr)
        if not r.ok:
            return AAASVResponse(ok=False, error_code=r.error_code,
                                 result=None, warnings=r.warnings,
                                 latency_ms=(time.perf_counter() - t0) * 1000)
        er: EmbedResult = r.result
        score = _cosine_similarity(_l2_norm(er.embedding),
                                   _l2_norm(voiceprint.embedding))
        conf = _confidence_from(er.duration_s, er.q_score)
        vr = VerifyResult(score=score,
                          decision=score >= self._threshold,
                          threshold=self._threshold,
                          q_score=er.q_score,
                          confidence=conf)
        return AAASVResponse(ok=True, error_code=None, result=vr,
                             warnings=r.warnings,
                             latency_ms=(time.perf_counter() - t0) * 1000)

    # ------------------------------------------------------------------
    def maybe_update(self, voiceprint, emb, q, gates):
        """Delegate to production implementation (unbound call)."""
        return AAASV.maybe_update(self, voiceprint, emb, q, gates)


# ═══════════════════════════════════════════════════════════════
# Helper factories
# ═══════════════════════════════════════════════════════════════

def _silence(duration_s: float, sr: int = _TARGET_SR) -> np.ndarray:
    """Return a silent (all-zeros) waveform."""
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _tone(duration_s: float, sr: int = _TARGET_SR, freq: float = 440.0) -> np.ndarray:
    """Return a pure-tone waveform at the given frequency."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _stub() -> _StubAAASV:
    return _StubAAASV()


# ═══════════════════════════════════════════════════════════════
# 1.  test_embed_too_short
# ═══════════════════════════════════════════════════════════════

def test_embed_too_short():
    """wav < 1.0 s must return ok=False, error_code=E_TOO_SHORT."""
    sv = _stub()
    wav = _tone(0.5)   # 0.5 s — strictly below minimum
    resp = sv.embed(wav, _TARGET_SR)
    assert resp.ok is False
    assert resp.error_code == E_TOO_SHORT
    assert resp.result is None


# ═══════════════════════════════════════════════════════════════
# 2.  test_embed_too_long
# ═══════════════════════════════════════════════════════════════

def test_embed_too_long():
    """wav > 30.0 s must return ok=False, error_code=E_TOO_LONG."""
    sv = _stub()
    wav = _tone(31.0)   # 31 s — above maximum
    resp = sv.embed(wav, _TARGET_SR)
    assert resp.ok is False
    assert resp.error_code == E_TOO_LONG
    assert resp.result is None


# ═══════════════════════════════════════════════════════════════
# 3.  test_embed_silent
# ═══════════════════════════════════════════════════════════════

def test_embed_silent():
    """All-zeros wav must return ok=False, error_code=E_SILENT."""
    sv = _stub()
    wav = _silence(3.0)   # long enough but completely silent
    resp = sv.embed(wav, _TARGET_SR)
    assert resp.ok is False
    assert resp.error_code == E_SILENT
    assert resp.result is None


def test_rms_dbfs_silent():
    """_rms_dbfs of all-zeros should be -inf."""
    assert _rms_dbfs(np.zeros(1000, dtype=np.float32)) == float("-inf")


# ═══════════════════════════════════════════════════════════════
# 4.  test_embed_stereo_converts_to_mono
# ═══════════════════════════════════════════════════════════════

def test_embed_stereo_converts_to_mono():
    """
    Stereo input (shape: 2 × N) must NOT raise an error.
    The channels are auto-averaged to mono before validation.
    """
    sv = _stub()
    n = int(3.0 * _TARGET_SR)
    ch0 = (0.1 * np.sin(2 * np.pi * 300 * np.linspace(0, 3.0, n))).astype(np.float32)
    ch1 = (0.1 * np.sin(2 * np.pi * 500 * np.linspace(0, 3.0, n))).astype(np.float32)
    stereo = np.stack([ch0, ch1], axis=0)   # shape (2, N)
    resp = sv.embed(stereo, _TARGET_SR)
    assert resp.ok is True, (
        f"Stereo input should succeed after mono conversion; "
        f"got error_code={resp.error_code}"
    )
    assert resp.result is not None
    assert isinstance(resp.result, EmbedResult)


# ═══════════════════════════════════════════════════════════════
# 5.  test_embed_1s_2s_w_short_warning
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("duration_s", [1.0, 1.5, 1.99])
def test_embed_1s_2s_w_short_warning(duration_s):
    """
    Durations in [1.0, 2.0) must return ok=True but include W_SHORT
    in the warnings list.
    """
    sv = _stub()
    wav = _tone(duration_s)
    resp = sv.embed(wav, _TARGET_SR)
    assert resp.ok is True, (
        f"Duration {duration_s}s should succeed; got error_code={resp.error_code}"
    )
    assert W_SHORT in resp.warnings, (
        f"Expected W_SHORT in warnings for {duration_s}s; got {resp.warnings}"
    )


def test_embed_exactly_2s_no_w_short():
    """Duration ≥ 2.0 s must NOT contain W_SHORT."""
    sv = _stub()
    wav = _tone(2.0)
    resp = sv.embed(wav, _TARGET_SR)
    assert resp.ok is True
    assert W_SHORT not in resp.warnings


# ═══════════════════════════════════════════════════════════════
# 6.  test_response_envelope_fields
# ═══════════════════════════════════════════════════════════════

def test_response_envelope_fields():
    """
    AAASVResponse must expose ok, error_code, result, warnings, latency_ms.
    Types must match the dataclass definition.
    """
    sv = _stub()
    wav = _tone(3.0)
    resp = sv.embed(wav, _TARGET_SR)

    # Structural checks
    assert hasattr(resp, "ok"),          "AAASVResponse missing 'ok'"
    assert hasattr(resp, "error_code"),  "AAASVResponse missing 'error_code'"
    assert hasattr(resp, "result"),      "AAASVResponse missing 'result'"
    assert hasattr(resp, "warnings"),    "AAASVResponse missing 'warnings'"
    assert hasattr(resp, "latency_ms"),  "AAASVResponse missing 'latency_ms'"

    # Type checks
    assert isinstance(resp.ok,         bool)
    assert isinstance(resp.warnings,   list)
    assert isinstance(resp.latency_ms, float)

    # For a successful embed, error_code must be None
    assert resp.ok is True
    assert resp.error_code is None

    # EmbedResult checks
    er: EmbedResult = resp.result
    assert isinstance(er, EmbedResult)
    assert isinstance(er.embedding,    np.ndarray)
    assert er.embedding.shape == (_EMB_DIM,)
    assert isinstance(er.duration_s,   float)
    assert isinstance(er.sample_rate,  int)
    # q_score may be float or None
    assert er.q_score is None or isinstance(er.q_score, float)


# ═══════════════════════════════════════════════════════════════
# 7.  test_verify_result_fields
# ═══════════════════════════════════════════════════════════════

def test_verify_result_fields():
    """
    VerifyResult must have: score, decision, threshold, q_score, confidence.
    confidence must be one of "high" | "medium" | "low".
    """
    sv = _stub()
    wav = _tone(5.0)

    # Build a minimal voiceprint by hand
    rng = np.random.default_rng(1)
    vp_emb = _l2_norm(rng.standard_normal(_EMB_DIM).astype(np.float32))
    now = datetime.now(timezone.utc).isoformat()
    vp = Voiceprint(
        speaker_id="spk_test",
        embedding=vp_emb,
        n_utterances=3,
        created_at=now,
        updated_at=now,
        mean_q=0.75,
    )

    resp = sv.verify(wav, _TARGET_SR, vp)
    assert resp.ok is True, f"verify() failed: {resp.error_code}"

    vr: VerifyResult = resp.result
    assert isinstance(vr, VerifyResult)

    assert hasattr(vr, "score"),      "VerifyResult missing 'score'"
    assert hasattr(vr, "decision"),   "VerifyResult missing 'decision'"
    assert hasattr(vr, "threshold"),  "VerifyResult missing 'threshold'"
    assert hasattr(vr, "q_score"),    "VerifyResult missing 'q_score'"
    assert hasattr(vr, "confidence"), "VerifyResult missing 'confidence'"

    assert isinstance(vr.score,     float)
    assert isinstance(vr.decision,  bool)
    assert isinstance(vr.threshold, float)
    assert vr.confidence in {"high", "medium", "low"}, (
        f"confidence must be high/medium/low, got '{vr.confidence}'"
    )
    # score ∈ [-1, 1] for cosine similarity of L2-normalised vectors
    assert -1.0 <= vr.score <= 1.0, f"score={vr.score} out of range"


# ═══════════════════════════════════════════════════════════════
# 8.  test_maybe_update_gates
# ═══════════════════════════════════════════════════════════════

def _make_voiceprint(seed: int = 42) -> Voiceprint:
    rng = np.random.default_rng(seed)
    emb = _l2_norm(rng.standard_normal(_EMB_DIM).astype(np.float32))
    now = datetime.now(timezone.utc).isoformat()
    return Voiceprint(
        speaker_id="spk_gate_test",
        embedding=emb,
        n_utterances=5,
        created_at=now,
        updated_at=now,
        mean_q=0.75,
    )


def _make_emb(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _l2_norm(rng.standard_normal(_EMB_DIM).astype(np.float32))


@pytest.mark.parametrize("gate_score,q,should_update", [
    # Both gates pass → update (strict > 0.85 and > 0.7)
    (0.90, 0.80, True),
    (0.86, 0.71, True),
    # score gate fails (≤ 0.85) → no update
    (0.85, 0.80, False),
    (0.80, 0.80, False),
    # q gate fails (≤ 0.7) → no update
    (0.90, 0.70, False),
    (0.90, 0.60, False),
    # Both gates fail → no update
    (0.80, 0.60, False),
])
def test_maybe_update_gates(gate_score, q, should_update):
    """
    maybe_update() should modify the voiceprint ONLY when
    gates["score"] > 0.85 AND q > 0.7 (strict inequalities).
    """
    sv = _stub()
    vp = _make_voiceprint()
    emb = _make_emb()
    gates = {"score": gate_score}

    updated_vp = sv.maybe_update(vp, emb, q, gates)

    if should_update:
        # n_utterances must increase
        assert updated_vp.n_utterances == vp.n_utterances + 1, (
            f"Expected n_utterances to increment for score={gate_score}, q={q}; "
            f"got {updated_vp.n_utterances}"
        )
        # Embedding must have changed
        assert not np.allclose(updated_vp.embedding, vp.embedding), (
            f"Expected embedding to change for score={gate_score}, q={q}"
        )
    else:
        # Voiceprint must be the SAME OBJECT (no copy on no-update)
        assert updated_vp is vp, (
            f"Expected same voiceprint object (no update) for "
            f"score={gate_score}, q={q}; got a new object"
        )


# ═══════════════════════════════════════════════════════════════
# Additional: Voiceprint field completeness
# ═══════════════════════════════════════════════════════════════

def test_voiceprint_fields():
    """Voiceprint must have all required fields with correct types."""
    sv = _stub()
    wav = _tone(3.0)
    resp = sv.enroll([wav], [_TARGET_SR], "spk_001")
    assert resp.ok is True, f"enroll() failed: {resp.error_code}"
    vp: Voiceprint = resp.result
    assert isinstance(vp, Voiceprint)
    assert hasattr(vp, "speaker_id")
    assert hasattr(vp, "embedding")
    assert hasattr(vp, "n_utterances")
    assert hasattr(vp, "created_at")
    assert hasattr(vp, "updated_at")
    assert hasattr(vp, "mean_q")
    assert vp.speaker_id == "spk_001"
    assert vp.n_utterances == 1
    assert vp.embedding.shape == (_EMB_DIM,)
    # embedding must be L2-normalised (norm ≈ 1)
    assert abs(np.linalg.norm(vp.embedding) - 1.0) < 1e-5
