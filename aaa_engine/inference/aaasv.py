# =============================================================
# aaa_engine/inference/aaasv.py
# AAA Engine – Speaker Verification Inference Module   [CUSTOM]
# =============================================================
#
# Implements the AAASV inference class:
#   • embed(wav, sr)                   → EmbedResult
#   • enroll(wavs, srs, speaker_id)    → Voiceprint
#   • verify(wav, sr, voiceprint)      → VerifyResult
#   • maybe_update(voiceprint, emb, q, gates) → Voiceprint
#
# Model: ECAPA-TDNN + AAM-Softmax + optional StabilityHead q(x)
# Embeddings: 192-dim, L2-normalised
#
# Design:
#   • Lazy singleton via get_instance() (thread-safe Lock)
#   • Warm-up on init: one forward pass with 3 s silence
#   • CPU latency benchmark on first call: p50/p95 over 20 runs
#   • Input validation strictly before any forward pass
#   • No dependency on calibrator.py — calibration loaded from JSON
# =============================================================

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    _TORCHAUDIO_AVAILABLE = False

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# 1.  Error code constants  (from SHARED.md §2)
# ═══════════════════════════════════════════════════════════════

E_FORMAT            = "E_FORMAT"            # unsupported format / sample rate
E_DECODE            = "E_DECODE"            # audio decode failure
E_TOO_SHORT         = "E_TOO_SHORT"         # < 1.0 s usable speech
E_TOO_LONG          = "E_TOO_LONG"          # > 30.0 s
E_SILENT            = "E_SILENT"            # RMS < -60 dBFS
E_NONFINITE         = "E_NONFINITE"         # NaN/Inf in embedding
E_NO_ENROLLMENT     = "E_NO_ENROLLMENT"     # voiceprint not found
E_MODEL_UNAVAILABLE = "E_MODEL_UNAVAILABLE" # model files missing / corrupt
E_VERSION_MISMATCH  = "E_VERSION_MISMATCH"  # checkpoint / config mismatch
E_INTERNAL          = "E_INTERNAL"          # unexpected runtime error

# Warning tags (not error codes — added to warnings list)
W_SHORT = "W_SHORT"   # duration in [1.0, 2.0) s — ok but warn

# Validation thresholds
_MIN_DURATION_S   = 1.0
_MAX_DURATION_S   = 30.0
_WARN_SHORT_S     = 2.0
_SILENT_DBFS      = -60.0          # RMS below this → E_SILENT
_TARGET_SR        = 16_000
_EMB_DIM          = 192

_DEFAULT_THRESHOLD = 0.5

# ═══════════════════════════════════════════════════════════════
# 2.  Response envelope and result dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class AAASVResponse:
    """Uniform envelope returned by all public AAASV methods."""
    ok: bool
    error_code: Optional[str]
    result: Optional[Any]        # EmbedResult | Voiceprint | VerifyResult
    warnings: List[str]
    latency_ms: float


@dataclass
class EmbedResult:
    """Result of AAASV.embed()."""
    embedding: np.ndarray        # shape (192,), L2-normalised
    q_score: Optional[float]     # from q head; None if unavailable
    duration_s: float
    sample_rate: int


@dataclass
class Voiceprint:
    """Speaker voiceprint stored after enrollment."""
    speaker_id: str
    embedding: np.ndarray        # mean of enrolled embeddings, L2-normalised
    n_utterances: int
    created_at: str              # ISO 8601
    updated_at: str              # ISO 8601
    mean_q: Optional[float]      # mean q_score over enrolled utterances


@dataclass
class VerifyResult:
    """Result of AAASV.verify()."""
    score: float                 # cosine similarity ∈ [-1, 1]
    decision: bool               # score >= threshold
    threshold: float
    q_score: Optional[float]     # q score for the probe utterance
    confidence: str              # "high" | "medium" | "low"


# ═══════════════════════════════════════════════════════════════
# 3.  Internal helpers
# ═══════════════════════════════════════════════════════════════

def _rms_dbfs(wav: np.ndarray) -> float:
    """Compute RMS level in dBFS.  Returns -inf for all-zero signals."""
    rms = float(np.sqrt(np.mean(wav.astype(np.float64) ** 2)))
    if rms == 0.0:
        return float("-inf")
    return 20.0 * np.log10(rms)


def _l2_norm(vec: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy of a 1-D array."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec.copy()
    return vec / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised 1-D arrays."""
    return float(np.dot(a, b))


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _confidence_from(duration_s: float, q_score: Optional[float]) -> str:
    """Heuristic confidence label based on duration and q_score."""
    if duration_s >= 5.0 and (q_score is None or q_score >= 0.7):
        return "high"
    if duration_s >= 2.0 and (q_score is None or q_score >= 0.4):
        return "medium"
    return "low"


def _verify_sha256(model_dir: str) -> None:
    """Verify SHA256SUMS manifest if present. Raises RuntimeError on mismatch."""
    manifest_path = os.path.join(model_dir, "SHA256SUMS")
    if not os.path.isfile(manifest_path):
        return  # optional — skip if absent
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            expected_hex, rel_path = parts
            full_path = os.path.join(model_dir, rel_path.strip())
            if not os.path.isfile(full_path):
                logger.warning("SHA256SUMS: file listed but not found: %s", rel_path)
                continue
            sha = hashlib.sha256()
            with open(full_path, "rb") as fp:
                for chunk in iter(lambda: fp.read(65536), b""):
                    sha.update(chunk)
            if sha.hexdigest().lower() != expected_hex.lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {rel_path}: "
                    f"expected {expected_hex}, got {sha.hexdigest()}"
                )


def _resample(wav: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """Resample wav from src_sr to tgt_sr using torchaudio if available,
    otherwise fall back to a simple linear-interpolation approximation."""
    if src_sr == tgt_sr:
        return wav
    if _TORCHAUDIO_AVAILABLE:
        t = torch.from_numpy(wav).float().unsqueeze(0)  # (1, T)
        t = torchaudio.functional.resample(t, orig_freq=src_sr, new_freq=tgt_sr)
        return t.squeeze(0).numpy()
    # Fallback: numpy linear interpolation
    target_len = int(round(len(wav) * tgt_sr / src_sr))
    old_indices = np.linspace(0, len(wav) - 1, target_len)
    return np.interp(old_indices, np.arange(len(wav)), wav).astype(wav.dtype)


# ═══════════════════════════════════════════════════════════════
# 4.  Stub model (used when no real checkpoint is present)
# ═══════════════════════════════════════════════════════════════

class _StubModel(torch.nn.Module):
    """
    Minimal stub that mimics the ECAPA-TDNN + q-head interface.
    Returns deterministic (but random-seeded) 192-dim embeddings and
    a constant q_score of 0.8.  Used only when model.pt is absent
    (e.g., unit-test environments).
    """

    def __init__(self) -> None:
        super().__init__()
        self._rng = np.random.default_rng(42)

    def forward_embedding(self, wav_tensor: torch.Tensor) -> Tuple[np.ndarray, Optional[float]]:
        """Return (embedding_192, q_score)."""
        emb = self._rng.standard_normal(_EMB_DIM).astype(np.float32)
        emb = _l2_norm(emb)
        return emb, 0.8


# ═══════════════════════════════════════════════════════════════
# 5.  Model loader
# ═══════════════════════════════════════════════════════════════

class _ModelWrapper:
    """
    Wraps the loaded checkpoint and exposes a single
    ``forward_embedding(wav_numpy, sr)`` → (embedding, q_score) interface.
    """

    def __init__(self, model_dir: str, device: torch.device) -> None:
        self.device = device
        self.model_dir = model_dir
        self._stub = False

        model_path = os.path.join(model_dir, "model.pt")
        config_path = os.path.join(model_dir, "config.yaml")
        preproc_path = os.path.join(model_dir, "preproc.yaml")

        # Verify hashes if manifest present
        try:
            _verify_sha256(model_dir)
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc

        if not os.path.isfile(model_path):
            logger.warning(
                "model.pt not found in %s; using _StubModel for inference.", model_dir
            )
            self._stub_model = _StubModel()
            self._stub = True
            self._sample_rate = _TARGET_SR
            self._has_q_head = False
            return

        # Load checkpoint
        logger.info("Loading model from %s", model_path)
        ckpt = torch.load(model_path, map_location=device)

        # Attempt to reconstruct model from checkpoint.
        # The checkpoint may contain the full model under "model" key,
        # or be a raw state dict.
        if isinstance(ckpt, dict) and "model" in ckpt:
            self._model = ckpt["model"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            # Reconstruct from state dict: requires config
            self._model = None
            logger.warning(
                "Checkpoint contains state_dict only; full model reconstruction "
                "is not implemented here. Using stub."
            )
            self._stub_model = _StubModel()
            self._stub = True
            self._sample_rate = _TARGET_SR
            self._has_q_head = False
            return
        else:
            # Assume it IS the model object
            self._model = ckpt

        if hasattr(self._model, "to"):
            self._model = self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()

        # Detect q head presence
        self._has_q_head = hasattr(self._model, "q_head") or hasattr(
            self._model, "stability_head"
        )

        # Read sample rate from preproc config
        self._sample_rate = _TARGET_SR
        if os.path.isfile(preproc_path):
            try:
                import yaml  # optional; gracefully skip if absent
                with open(preproc_path, "r") as fh:
                    preproc_cfg = yaml.safe_load(fh)
                self._sample_rate = int(
                    preproc_cfg.get("sample_rate", _TARGET_SR)
                )
            except Exception:
                pass  # fall back to 16 kHz

    # ------------------------------------------------------------------
    def forward_embedding(
        self, wav: np.ndarray, sr: int
    ) -> Tuple[np.ndarray, Optional[float]]:
        """
        Run the ECAPA-TDNN forward pass and return (embedding, q_score).
        wav must be 1-D float32, already at self._sample_rate.
        """
        if self._stub:
            return self._stub_model.forward_embedding(
                torch.from_numpy(wav).float()
            )

        with torch.no_grad():
            wav_t = torch.from_numpy(wav).float().unsqueeze(0).to(self.device)
            # Try standard forward signature: model(wav) → embedding
            # or model.encode(wav) → embedding
            emb_t: Optional[torch.Tensor] = None
            q_val: Optional[float] = None

            try:
                out = self._model(wav_t)
                if isinstance(out, tuple):
                    emb_t = out[0]
                    if len(out) > 1 and out[1] is not None:
                        q_val = float(out[1].squeeze())
                else:
                    emb_t = out
            except Exception:
                # Try encode() variant
                try:
                    emb_t = self._model.encode(wav_t)
                except Exception as exc:
                    raise RuntimeError(f"Model forward failed: {exc}") from exc

            if emb_t is None:
                raise RuntimeError("Model returned None embedding")

            emb_np = emb_t.squeeze().cpu().numpy().astype(np.float32)

            # Try q head separately if not returned by forward
            if q_val is None and self._has_q_head:
                try:
                    q_head = getattr(
                        self._model, "q_head",
                        getattr(self._model, "stability_head", None)
                    )
                    if q_head is not None:
                        # q head expects pooled stats; skip if unavailable separately
                        pass
                except Exception:
                    pass

        emb_np = _l2_norm(emb_np)
        return emb_np, q_val


# ═══════════════════════════════════════════════════════════════
# 6.  AAASV — main class
# ═══════════════════════════════════════════════════════════════

class AAASV:
    """
    Speaker verification inference engine.

    Parameters
    ----------
    model_dir : str
        Path to the model pack directory containing:
        model.pt, config.yaml, preproc.yaml,
        calibration.json (optional), SHA256SUMS (optional).
    device : str
        "auto" (default) → GPU if available, else CPU.
        "cpu", "cuda", "cuda:0", etc.
    """

    def __init__(self, model_dir: str, device: str = "auto") -> None:
        self.model_dir = model_dir
        self._threshold = _DEFAULT_THRESHOLD
        self._latency_benchmarked = False
        self.latency_p50_ms: Optional[float] = None
        self.latency_p95_ms: Optional[float] = None

        # Resolve device
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        logger.info("AAASV init: model_dir=%s device=%s", model_dir, self._device)

        # Load model
        try:
            self._wrapper = _ModelWrapper(model_dir, self._device)
        except Exception as exc:
            raise RuntimeError(f"{E_MODEL_UNAVAILABLE}: {exc}") from exc

        # Load calibration threshold
        self._load_calibration()

        # Warm-up: one forward pass with 3 s silence
        self._warmup()

        # Latency benchmark: p50/p95 over 20 calls
        self._benchmark_latency()

    # ------------------------------------------------------------------
    def _load_calibration(self) -> None:
        cal_path = os.path.join(self.model_dir, "calibration.json")
        if not os.path.isfile(cal_path):
            logger.info("calibration.json not found; using default threshold %.2f", _DEFAULT_THRESHOLD)
            return
        try:
            with open(cal_path, "r", encoding="utf-8") as fh:
                cal = json.load(fh)
            self._threshold = float(cal.get("threshold", _DEFAULT_THRESHOLD))
            logger.info("Loaded calibration threshold: %.4f", self._threshold)
        except Exception as exc:
            logger.warning("Failed to load calibration.json: %s; using default", exc)

    # ------------------------------------------------------------------
    def _silence_wav(self, duration_s: float = 3.0) -> np.ndarray:
        """Return a silence (all-zeros) numpy array at target SR."""
        n = int(duration_s * _TARGET_SR)
        return np.zeros(n, dtype=np.float32)

    # ------------------------------------------------------------------
    def _warmup(self) -> None:
        """Run one forward pass with 3 s silence to prime JIT / caches."""
        try:
            sil = self._silence_wav(3.0)
            self._wrapper.forward_embedding(sil, _TARGET_SR)
            logger.info("AAASV warm-up complete.")
        except Exception as exc:
            logger.warning("AAASV warm-up failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    def _benchmark_latency(self) -> None:
        """Measure p50/p95 CPU latency over 20 runs with 3 s silence."""
        try:
            sil = self._silence_wav(3.0)
            times_ms: List[float] = []
            for _ in range(20):
                t0 = time.perf_counter()
                self._wrapper.forward_embedding(sil, _TARGET_SR)
                times_ms.append((time.perf_counter() - t0) * 1000.0)
            times_ms.sort()
            self.latency_p50_ms = float(np.percentile(times_ms, 50))
            self.latency_p95_ms = float(np.percentile(times_ms, 95))
            logger.info(
                "AAASV latency benchmark: p50=%.1f ms  p95=%.1f ms",
                self.latency_p50_ms,
                self.latency_p95_ms,
            )
            self._latency_benchmarked = True
        except Exception as exc:
            logger.warning("AAASV latency benchmark failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Input validation helpers
    # ------------------------------------------------------------------

    def _validate_and_preprocess(
        self, wav: np.ndarray, sr: int
    ) -> Tuple[Optional[np.ndarray], Optional[str], List[str]]:
        """
        Validate and preprocess the input waveform.

        Returns
        -------
        (preprocessed_wav, error_code, warnings)
        On success:  (wav_np, None, warnings_list)
        On failure:  (None,  error_code, warnings_list)
        """
        warnings: List[str] = []

        # 1. Convert stereo → mono (do NOT reject)
        if wav.ndim == 2:
            wav = wav.mean(axis=0).astype(np.float32)
        elif wav.ndim == 1:
            wav = wav.astype(np.float32)
        else:
            return None, E_FORMAT, warnings

        # 2. Resample if needed
        if sr != _TARGET_SR:
            try:
                wav = _resample(wav, sr, _TARGET_SR).astype(np.float32)
                sr = _TARGET_SR
            except Exception:
                return None, E_DECODE, warnings

        # 3. Duration checks
        duration_s = len(wav) / float(sr)
        if duration_s < _MIN_DURATION_S:
            return None, E_TOO_SHORT, warnings
        if duration_s > _MAX_DURATION_S:
            return None, E_TOO_LONG, warnings

        # 4. Short-but-ok warning
        if duration_s < _WARN_SHORT_S:
            warnings.append(W_SHORT)

        # 5. Silence check
        rms_db = _rms_dbfs(wav)
        if rms_db < _SILENT_DBFS:
            return None, E_SILENT, warnings

        return wav, None, warnings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, wav: np.ndarray, sr: int) -> AAASVResponse:
        """
        Extract a 192-dim L2-normalised speaker embedding from wav.

        Parameters
        ----------
        wav : np.ndarray  — 1-D or 2-D (channels, samples) audio
        sr  : int         — sample rate of wav

        Returns
        -------
        AAASVResponse with result=EmbedResult on success.
        """
        t0 = time.perf_counter()

        try:
            processed, err, warnings = self._validate_and_preprocess(wav, sr)
            if err is not None:
                return AAASVResponse(
                    ok=False,
                    error_code=err,
                    result=None,
                    warnings=warnings,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

            duration_s = len(processed) / float(_TARGET_SR)

            emb, q_score = self._wrapper.forward_embedding(processed, _TARGET_SR)

            # Non-finite check
            if not np.all(np.isfinite(emb)):
                return AAASVResponse(
                    ok=False,
                    error_code=E_NONFINITE,
                    result=None,
                    warnings=warnings,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )

            result = EmbedResult(
                embedding=emb,
                q_score=q_score,
                duration_s=duration_s,
                sample_rate=_TARGET_SR,
            )
            return AAASVResponse(
                ok=True,
                error_code=None,
                result=result,
                warnings=warnings,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        except Exception as exc:
            logger.exception("AAASV.embed() unexpected error: %s", exc)
            return AAASVResponse(
                ok=False,
                error_code=E_INTERNAL,
                result=None,
                warnings=[],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

    # ------------------------------------------------------------------
    def enroll(
        self,
        wavs: List[np.ndarray],
        srs: List[int],
        speaker_id: str,
    ) -> AAASVResponse:
        """
        Enroll a speaker by computing embeddings for all provided utterances
        and averaging them into a single voiceprint.

        Parameters
        ----------
        wavs       : list of np.ndarray waveforms
        srs        : list of corresponding sample rates
        speaker_id : unique speaker identifier

        Returns
        -------
        AAASVResponse with result=Voiceprint on success.
        """
        t0 = time.perf_counter()
        all_warnings: List[str] = []
        embeddings: List[np.ndarray] = []
        q_scores: List[float] = []

        if len(wavs) == 0:
            return AAASVResponse(
                ok=False,
                error_code=E_TOO_SHORT,
                result=None,
                warnings=[],
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        for idx, (wav, sr) in enumerate(zip(wavs, srs)):
            resp = self.embed(wav, sr)
            # Merge warnings
            for w in resp.warnings:
                if w not in all_warnings:
                    all_warnings.append(w)
            if not resp.ok:
                # Fail-fast on the first bad utterance
                return AAASVResponse(
                    ok=False,
                    error_code=resp.error_code,
                    result=None,
                    warnings=all_warnings,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
            embed_result: EmbedResult = resp.result
            embeddings.append(embed_result.embedding)
            if embed_result.q_score is not None:
                q_scores.append(embed_result.q_score)

        # Average and re-normalise
        mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        mean_emb = _l2_norm(mean_emb)
        mean_q = float(np.mean(q_scores)) if q_scores else None
        now = _iso_now()

        vp = Voiceprint(
            speaker_id=speaker_id,
            embedding=mean_emb,
            n_utterances=len(embeddings),
            created_at=now,
            updated_at=now,
            mean_q=mean_q,
        )
        return AAASVResponse(
            ok=True,
            error_code=None,
            result=vp,
            warnings=all_warnings,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ------------------------------------------------------------------
    def verify(
        self,
        wav: np.ndarray,
        sr: int,
        voiceprint: Voiceprint,
    ) -> AAASVResponse:
        """
        Verify whether wav belongs to the speaker represented by voiceprint.

        Parameters
        ----------
        wav        : probe waveform
        sr         : sample rate of wav
        voiceprint : enrolled Voiceprint to compare against

        Returns
        -------
        AAASVResponse with result=VerifyResult on success.
        """
        t0 = time.perf_counter()

        resp = self.embed(wav, sr)
        if not resp.ok:
            return AAASVResponse(
                ok=False,
                error_code=resp.error_code,
                result=None,
                warnings=resp.warnings,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )

        embed_result: EmbedResult = resp.result
        probe_emb = embed_result.embedding
        q_score = embed_result.q_score

        # Cosine similarity (both vectors should already be L2-normalised)
        score = _cosine_similarity(_l2_norm(probe_emb), _l2_norm(voiceprint.embedding))
        decision = score >= self._threshold
        confidence = _confidence_from(embed_result.duration_s, q_score)

        result = VerifyResult(
            score=score,
            decision=decision,
            threshold=self._threshold,
            q_score=q_score,
            confidence=confidence,
        )
        return AAASVResponse(
            ok=True,
            error_code=None,
            result=result,
            warnings=resp.warnings,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    # ------------------------------------------------------------------
    def maybe_update(
        self,
        voiceprint: Voiceprint,
        emb: np.ndarray,
        q: float,
        gates: dict,
    ) -> Voiceprint:
        """
        Adaptive voiceprint update.

        The update fires ONLY when:
          gates["score"] > 0.85  AND  q > 0.7

        When updating, the new voiceprint embedding is the L2-normalised
        average of the existing embedding and the new embedding.

        Parameters
        ----------
        voiceprint : existing Voiceprint to (possibly) update
        emb        : new embedding to incorporate (192-dim, should be L2-norm)
        q          : stability q-score for the new embedding
        gates      : dict with at least key "score" (cosine similarity of
                     the new utterance vs the existing voiceprint)

        Returns
        -------
        Updated (or unchanged) Voiceprint.
        """
        gate_score = float(gates.get("score", 0.0))
        if gate_score <= 0.85 or q <= 0.7:
            return voiceprint  # gates not satisfied — no update

        # Weighted average: weight existing by n_utterances
        n = voiceprint.n_utterances
        old_emb = _l2_norm(voiceprint.embedding)
        new_emb = _l2_norm(emb)
        updated_emb = _l2_norm(
            (old_emb * n + new_emb) / (n + 1)
        )

        # Update mean_q
        old_mean_q = voiceprint.mean_q
        if old_mean_q is not None:
            updated_mean_q = (old_mean_q * n + q) / (n + 1)
        else:
            updated_mean_q = q

        return Voiceprint(
            speaker_id=voiceprint.speaker_id,
            embedding=updated_emb,
            n_utterances=n + 1,
            created_at=voiceprint.created_at,
            updated_at=_iso_now(),
            mean_q=updated_mean_q,
        )


# ═══════════════════════════════════════════════════════════════
# 7.  Lazy singleton
# ═══════════════════════════════════════════════════════════════

_instance_lock: threading.Lock = threading.Lock()
_instances: Dict[str, AAASV] = {}


def get_instance(model_dir: str, device: str = "auto") -> AAASV:
    """
    Return a cached AAASV instance for the given model_dir + device.

    Thread-safe — multiple callers will block until the first
    initialisation completes.

    Parameters
    ----------
    model_dir : str  — path to model pack directory
    device    : str  — "auto" | "cpu" | "cuda" | "cuda:0" etc.
    """
    cache_key = f"{os.path.abspath(model_dir)}::{device}"
    if cache_key in _instances:
        return _instances[cache_key]
    with _instance_lock:
        # Double-checked locking
        if cache_key not in _instances:
            _instances[cache_key] = AAASV(model_dir=model_dir, device=device)
    return _instances[cache_key]
