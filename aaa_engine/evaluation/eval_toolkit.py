# =============================================================
# aaa_engine/evaluation/eval_toolkit.py
# AAA Engine — Phase 3.B  Unified Evaluation Toolkit
# =============================================================
# Unified evaluation code-path for B0, B1, and AAA-SV models.
#
# Public surface
# ──────────────
#   FREEZE_ENABLED              module-level flag; set True to allow test manifests
#   TrialList                   dataclass for a bucket of evaluation trial pairs
#   EvalResult                  dataclass returned by evaluate_model()
#   BucketedTrialGenerator      generates trial buckets from a manifest CSV
#   evaluate_model(...)         runs full evaluation, returns EvalResult
#
# CLI usage
# ─────────
#   python -m aaa_engine.evaluation.eval_toolkit \
#       --manifest  data/manifests/val.csv \
#       --data-root /path/to/vox1 \
#       --model-ckpt runs/.../checkpoint_best.pt \
#       --output    runs/.../eval_result.json
#
# Key design notes
# ────────────────
# • stdlib + numpy + torch only — no extra runtime deps.
# • minDCF: standard NIST formula (p_target=0.05, C_fa=1, C_miss=10).
# • Bootstrap: 1 000 iterations, sample with replacement over trial pairs.
# • DET curve: FAR/FRR at ~200 evenly-spaced thresholds.
# • sha256 of a TrialList = hash of sorted, joined "enroll|test|label" lines.
# • Manifest is "test-tagged" if its path contains the substring "test".
# • freeze guard: RuntimeError raised when test manifest used without
#   FREEZE_ENABLED = True.
# =============================================================

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level freeze flag
# ---------------------------------------------------------------------------
#: Set to True (programmatically or via --freeze CLI flag) to allow evaluation
#: on test-tagged manifests.  Defaults to False for safety.
FREEZE_ENABLED: bool = False

# ---------------------------------------------------------------------------
# NIST DCF constants
# ---------------------------------------------------------------------------
_P_TARGET: float = 0.05
_C_FA: float = 1.0
_C_MISS: float = 10.0

# ---------------------------------------------------------------------------
# Bucket dimension definitions
# ---------------------------------------------------------------------------
_DURATION_BUCKETS: List[int] = [1, 2, 3, 5]      # seconds
_SNR_BUCKETS: List[int] = [0, 5, 10, 20]          # dB
_N_DET_THRESHOLDS: int = 200
_N_BOOTSTRAP: int = 1000
_BOOTSTRAP_SEED: int = 42


# ===========================================================================
# Dataclasses
# ===========================================================================

@dataclass
class TrialList:
    """A bucket of speaker-verification trial pairs.

    Attributes
    ----------
    pairs:           List of (enroll_wav, test_wav, label) tuples.
                     label is 1 (same speaker) or 0 (different).
    sha256:          Deterministic hash of the sorted pair lines.
    source_manifest: Path to the manifest CSV used to build this list.
    bucket_name:     Human-readable bucket identifier, e.g. "duration_2s".
    n_positive:      Number of genuine (same-speaker) pairs.
    n_negative:      Number of impostor (different-speaker) pairs.
    """
    pairs: List[Tuple[str, str, int]]
    sha256: str
    source_manifest: str
    bucket_name: str
    n_positive: int
    n_negative: int


@dataclass
class EvalResult:
    """Full evaluation result for a single model / trial-list combination.

    Attributes
    ----------
    eer:                  Equal Error Rate (fraction in [0, 1]).
    frr_at_far1pct:       FRR at FAR = 1 % (fraction).
    min_dcf:              minDCF under NIST parameters.
    per_bucket_eer:       Dict mapping bucket_name → EER fraction.
    bootstrap_ci:         Dict with keys: eer_ci_low/high, frr_ci_low/high,
                          speaker_eer_ci_low/high (all fractions, 95 %).
    det_curve:            List of (FAR, FRR) points (both fractions).
    trial_list_sha256:    sha256 of the TrialList used.
    manifest_hash:        sha256 of the manifest CSV, or "unknown".
    run_id:               UUID string for this evaluation run.
    timestamp:            ISO-8601 UTC timestamp.
    model_label:          One of "B0", "B1", "AAA-SV".
    """
    eer: float
    frr_at_far1pct: float
    min_dcf: float
    per_bucket_eer: Dict[str, float]
    bootstrap_ci: Dict[str, float]
    det_curve: List[Tuple[float, float]]
    trial_list_sha256: str
    manifest_hash: str
    run_id: str
    timestamp: str
    model_label: str


# ===========================================================================
# Low-level metric helpers
# ===========================================================================

def _compute_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Return (eer, threshold_at_eer); eer is a fraction in [0, 1].

    Parameters
    ----------
    scores: float array of cosine-similarity scores (higher → more similar).
    labels: int array; 1 = same speaker, 0 = different speaker.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    # Build a dense threshold grid
    unique = np.unique(scores)
    if len(unique) > 1:
        grid = np.linspace(unique[0], unique[-1],
                           min(10_000, len(unique) * 10))
        thresholds = np.unique(np.concatenate([unique, grid]))
    else:
        thresholds = unique

    best_gap = 1.0
    best_thr = thresholds[0]
    best_far = best_frr = 1.0

    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        fa = int(((preds == 1) & (labels == 0)).sum())
        fr = int(((preds == 0) & (labels == 1)).sum())
        far = fa / n_neg
        frr = fr / n_pos
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap = gap
            best_thr = float(thr)
            best_far = far
            best_frr = frr

    eer = (best_far + best_frr) / 2.0
    return float(eer), float(best_thr)


def _compute_frr_at_far(
    scores: np.ndarray,
    labels: np.ndarray,
    target_far: float = 0.01,
) -> float:
    """Return FRR (fraction) at the threshold closest to target_far."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    thresholds = np.unique(scores)
    best_frr = float("nan")
    best_diff = float("inf")
    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        fa = int(((preds == 1) & (labels == 0)).sum())
        fr = int(((preds == 0) & (labels == 1)).sum())
        far = fa / n_neg
        frr = fr / n_pos
        diff = abs(far - target_far)
        if diff < best_diff:
            best_diff = diff
            best_frr = frr
    return float(best_frr)


def _compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = _P_TARGET,
    c_fa: float = _C_FA,
    c_miss: float = _C_MISS,
) -> float:
    """NIST SRE minDCF.

    DCF(thr) = C_miss * P_miss(thr) * P_target
             + C_fa   * P_fa(thr)   * (1 - P_target)

    minDCF = min over thresholds of DCF(thr)
             divided by the prior-only cost (normalised form).

    The normalised prior-only cost is:
        min(C_miss * p_target, C_fa * (1 - p_target))
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    thresholds = np.unique(scores)
    beta = (c_fa * (1.0 - p_target)) / (c_miss * p_target)
    prior_cost = min(c_miss * p_target, c_fa * (1.0 - p_target))

    min_cost = float("inf")
    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        fa = int(((preds == 1) & (labels == 0)).sum())
        fr = int(((preds == 0) & (labels == 1)).sum())
        p_miss = fr / n_pos
        p_fa = fa / n_neg
        dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1.0 - p_target)
        if dcf < min_cost:
            min_cost = dcf

    return float(min_cost / prior_cost) if prior_cost > 0 else float("nan")


def _compute_det_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = _N_DET_THRESHOLDS,
) -> List[Tuple[float, float]]:
    """Return a list of (FAR, FRR) points for a DET curve."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return []

    lo, hi = float(scores.min()), float(scores.max())
    if lo == hi:
        thresholds = np.array([lo])
    else:
        thresholds = np.linspace(lo, hi, n_thresholds)

    points: List[Tuple[float, float]] = []
    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        fa = int(((preds == 1) & (labels == 0)).sum())
        fr = int(((preds == 0) & (labels == 1)).sum())
        points.append((fa / n_neg, fr / n_pos))
    return points


def _bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """95% bootstrap CI for EER and FRR@FAR1%, sampling over trial pairs.

    Returns a dict with keys:
        eer_ci_low, eer_ci_high, frr_ci_low, frr_ci_high
    All values are fractions.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = len(scores)

    eer_samples: List[float] = []
    frr_samples: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s_b = scores[idx]
        l_b = labels[idx]
        eer_b, _ = _compute_eer(s_b, l_b)
        frr_b = _compute_frr_at_far(s_b, l_b, target_far=0.01)
        if not np.isnan(eer_b):
            eer_samples.append(eer_b)
        if not np.isnan(frr_b):
            frr_samples.append(frr_b)

    eer_arr = np.array(eer_samples)
    frr_arr = np.array(frr_samples)

    def _ci(arr: np.ndarray) -> Tuple[float, float]:
        if len(arr) == 0:
            return float("nan"), float("nan")
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    eer_lo, eer_hi = _ci(eer_arr)
    frr_lo, frr_hi = _ci(frr_arr)

    return {
        "eer_ci_low": eer_lo,
        "eer_ci_high": eer_hi,
        "frr_ci_low": frr_lo,
        "frr_ci_high": frr_hi,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


def _speaker_bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    speaker_ids: List[str],
    n_bootstrap: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """Speaker-level bootstrap CI: resample speakers, then pool their trials."""
    rng = np.random.default_rng(seed + 1)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    unique_speakers = list(set(speaker_ids))
    if len(unique_speakers) < 2:
        return {
            "speaker_eer_ci_low": float("nan"),
            "speaker_eer_ci_high": float("nan"),
        }

    # Map speaker → indices in score/label arrays
    spk_map: Dict[str, List[int]] = {}
    for i, spk in enumerate(speaker_ids):
        spk_map.setdefault(spk, []).append(i)

    eer_samples: List[float] = []
    for _ in range(n_bootstrap):
        sampled_spks = rng.choice(unique_speakers, size=len(unique_speakers),
                                   replace=True)
        idx: List[int] = []
        for spk in sampled_spks:
            idx.extend(spk_map[spk])
        if not idx:
            continue
        s_b = scores[idx]
        l_b = labels[idx]
        eer_b, _ = _compute_eer(s_b, l_b)
        if not np.isnan(eer_b):
            eer_samples.append(eer_b)

    arr = np.array(eer_samples)
    if len(arr) == 0:
        return {
            "speaker_eer_ci_low": float("nan"),
            "speaker_eer_ci_high": float("nan"),
        }
    return {
        "speaker_eer_ci_low": float(np.percentile(arr, 2.5)),
        "speaker_eer_ci_high": float(np.percentile(arr, 97.5)),
    }


# ===========================================================================
# sha256 helpers
# ===========================================================================

def _sha256_trial_list(pairs: List[Tuple[str, str, int]]) -> str:
    """Deterministic sha256 of a trial list.

    Hash input = sorted join of "enroll_path|test_path|label" lines.
    """
    lines = sorted(f"{enroll}|{test}|{label}" for enroll, test, label in pairs)
    content = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: str) -> str:
    """Return hex sha256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ===========================================================================
# BucketedTrialGenerator
# ===========================================================================

class BucketedTrialGenerator:
    """Generate evaluation trial pairs from a manifest CSV, grouped into buckets.

    Manifest CSV format (required columns):
        wav_path    — relative or absolute path to the audio file
        speaker_id  — speaker identity label

    Optional columns used for bucketing:
        duration_s     — clip duration in seconds (float)
        snr_db         — signal-to-noise ratio in dB (float)
        codec          — codec string, e.g. "opus", "g711"
        is_8khz_up     — 1 if the file was upsampled from 8 kHz, else 0
        device         — recording device string

    Parameters
    ----------
    manifest_csv:  Path to the manifest CSV file.
    data_root:     If set, wav_path values are interpreted relative to this root.
    seed:          Random seed for reproducible pair sampling.
    max_pairs_per_bucket: Maximum trial pairs per bucket (None = unlimited).
    """

    def __init__(
        self,
        manifest_csv: str,
        data_root: Optional[str] = None,
        seed: int = 42,
        max_pairs_per_bucket: Optional[int] = None,
    ) -> None:
        self.manifest_csv = str(manifest_csv)
        self.data_root = str(data_root) if data_root else None
        self.seed = seed
        self.max_pairs_per_bucket = max_pairs_per_bucket
        self._records: List[Dict] = []
        self._manifest_hash: str = "unknown"
        self._rng = np.random.default_rng(seed)
        self._load_manifest()

    # ------------------------------------------------------------------
    def _load_manifest(self) -> None:
        """Read and validate the manifest CSV."""
        self._manifest_hash = _sha256_file(self.manifest_csv)
        with open(self.manifest_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"Manifest CSV has no header: {self.manifest_csv}")
            required = {"wav_path", "speaker_id"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"Manifest CSV missing required columns: {missing}"
                )
            for row in reader:
                wav = row["wav_path"].strip()
                if self.data_root:
                    wav = str(Path(self.data_root) / wav)
                row["_wav_path_resolved"] = wav
                self._records.append(row)
        logger.info(
            "Loaded manifest: %d records from %s (sha256=%s…)",
            len(self._records), self.manifest_csv, self._manifest_hash[:12],
        )

    # ------------------------------------------------------------------
    @property
    def manifest_hash(self) -> str:
        return self._manifest_hash

    # ------------------------------------------------------------------
    def _make_pairs(
        self,
        records: List[Dict],
        bucket_name: str,
    ) -> TrialList:
        """Generate same-speaker and cross-speaker pairs from a record subset."""
        # Group by speaker
        spk_map: Dict[str, List[str]] = {}
        for rec in records:
            spk = rec["speaker_id"].strip()
            spk_map.setdefault(spk, []).append(rec["_wav_path_resolved"])

        same_pairs: List[Tuple[str, str, int]] = []
        diff_pairs: List[Tuple[str, str, int]] = []

        # Same-speaker pairs
        for spk, wavs in spk_map.items():
            for w1, w2 in itertools.combinations(wavs, 2):
                same_pairs.append((w1, w2, 1))

        # Cross-speaker pairs (all speaker combos, one pair each)
        speakers = list(spk_map.keys())
        for s1, s2 in itertools.combinations(speakers, 2):
            w1 = spk_map[s1][0]
            w2 = spk_map[s2][0]
            diff_pairs.append((w1, w2, 0))

        # Optionally cap
        if self.max_pairs_per_bucket is not None:
            half = self.max_pairs_per_bucket // 2
            if len(same_pairs) > half:
                idx = self._rng.choice(len(same_pairs), half, replace=False)
                same_pairs = [same_pairs[i] for i in idx]
            if len(diff_pairs) > half:
                idx = self._rng.choice(len(diff_pairs), half, replace=False)
                diff_pairs = [diff_pairs[i] for i in idx]

        all_pairs = same_pairs + diff_pairs
        sha = _sha256_trial_list(all_pairs)

        return TrialList(
            pairs=all_pairs,
            sha256=sha,
            source_manifest=self.manifest_csv,
            bucket_name=bucket_name,
            n_positive=len(same_pairs),
            n_negative=len(diff_pairs),
        )

    # ------------------------------------------------------------------
    def generate_all(self) -> Dict[str, TrialList]:
        """Generate all bucket trial lists.

        Returns
        -------
        dict mapping bucket_name → TrialList.
        Always includes "all" (the full manifest), plus per-dimension buckets
        for duration, SNR, codec, 8kHz-upsampled, and device.
        """
        buckets: Dict[str, TrialList] = {}

        # ── "all" bucket ──────────────────────────────────────────────────
        buckets["all"] = self._make_pairs(self._records, "all")

        # ── Helper: resolve optional float/int column ─────────────────────
        def _get_float(rec: Dict, col: str) -> Optional[float]:
            val = rec.get(col, "").strip()
            if val == "":
                return None
            try:
                return float(val)
            except ValueError:
                return None

        def _get_str(rec: Dict, col: str) -> Optional[str]:
            val = rec.get(col, "").strip()
            return val if val else None

        # ── Duration buckets ──────────────────────────────────────────────
        for dur in _DURATION_BUCKETS:
            subset = [
                r for r in self._records
                if (d := _get_float(r, "duration_s")) is not None
                and round(d) == dur
            ]
            if subset:
                buckets[f"duration_{dur}s"] = self._make_pairs(
                    subset, f"duration_{dur}s"
                )

        # ── SNR buckets ───────────────────────────────────────────────────
        for snr in _SNR_BUCKETS:
            subset = [
                r for r in self._records
                if (s := _get_float(r, "snr_db")) is not None
                and round(s) == snr
            ]
            if subset:
                buckets[f"snr_{snr}dB"] = self._make_pairs(
                    subset, f"snr_{snr}dB"
                )

        # ── Codec buckets ─────────────────────────────────────────────────
        codecs: set = {_get_str(r, "codec") for r in self._records} - {None}
        for codec in codecs:
            subset = [
                r for r in self._records
                if _get_str(r, "codec") == codec
            ]
            if subset:
                name = f"codec_{codec}"
                buckets[name] = self._make_pairs(subset, name)

        # ── 8kHz-upsampled bucket ─────────────────────────────────────────
        subset_8k = [
            r for r in self._records
            if _get_float(r, "is_8khz_up") == 1.0
        ]
        if subset_8k:
            buckets["8khz_upsampled"] = self._make_pairs(
                subset_8k, "8khz_upsampled"
            )

        # ── Device-mismatch buckets ───────────────────────────────────────
        devices: set = {_get_str(r, "device") for r in self._records} - {None}
        for device in devices:
            subset = [
                r for r in self._records
                if _get_str(r, "device") == device
            ]
            if subset:
                name = f"device_{device}"
                buckets[name] = self._make_pairs(subset, name)

        logger.info("Generated %d trial buckets.", len(buckets))
        return buckets


# ===========================================================================
# evaluate_model
# ===========================================================================

def evaluate_model(
    embedding_fn: Callable[[str], np.ndarray],
    trial_list: TrialList,
    freeze: bool = False,
    model_label: str = "AAA-SV",
    manifest_path: Optional[str] = None,
) -> "EvalResult":
    """Evaluate a speaker-verification model against a trial list.

    Parameters
    ----------
    embedding_fn:   Callable mapping wav_path (str) → 192-dim numpy array.
                    The array must be L2-normalised (unit norm).
    trial_list:     TrialList dataclass instance (from BucketedTrialGenerator).
    freeze:         If True, test-tagged manifests are allowed.
                    If False and the manifest is test-tagged, raises RuntimeError.
    model_label:    One of "B0", "B1", "AAA-SV".
    manifest_path:  Optional path to the source manifest CSV; used for
                    test-manifest detection and manifest_hash in EvalResult.

    Returns
    -------
    EvalResult dataclass.

    Raises
    ------
    RuntimeError: When a test-tagged manifest is used without freeze=True
                  or FREEZE_ENABLED=True.
    """
    # ── Freeze guard ──────────────────────────────────────────────────────
    _effective_freeze = freeze or FREEZE_ENABLED
    _manifest_to_check = manifest_path or trial_list.source_manifest
    if _is_test_manifest(_manifest_to_check) and not _effective_freeze:
        raise RuntimeError(
            "Test manifests require --freeze flag."
        )

    # ── Compute manifest hash ─────────────────────────────────────────────
    if manifest_path and os.path.isfile(manifest_path):
        m_hash = _sha256_file(manifest_path)
    elif trial_list.source_manifest and os.path.isfile(trial_list.source_manifest):
        m_hash = _sha256_file(trial_list.source_manifest)
    else:
        m_hash = "unknown"

    logger.info(
        "evaluate_model: bucket=%s, pairs=%d, model=%s",
        trial_list.bucket_name, len(trial_list.pairs), model_label,
    )

    # ── Extract embeddings (cached) ───────────────────────────────────────
    unique_paths: set = set()
    for enroll, test, _ in trial_list.pairs:
        unique_paths.add(enroll)
        unique_paths.add(test)

    emb_cache: Dict[str, np.ndarray] = {}
    for path in unique_paths:
        try:
            emb = embedding_fn(path)
            emb_arr = np.asarray(emb, dtype=np.float64)
            # Ensure unit norm
            norm = float(np.linalg.norm(emb_arr))
            if norm > 1e-9:
                emb_arr = emb_arr / norm
            emb_cache[path] = emb_arr
        except Exception as exc:
            logger.warning("SKIP embedding for %s: %s", path, exc)

    # ── Compute cosine scores ─────────────────────────────────────────────
    scores: List[float] = []
    labels: List[int] = []
    enroll_spks: List[str] = []   # for speaker-level bootstrap
    skipped = 0

    for enroll, test, label in trial_list.pairs:
        e1 = emb_cache.get(enroll)
        e2 = emb_cache.get(test)
        if e1 is None or e2 is None:
            skipped += 1
            continue
        # Cosine similarity = dot product of unit-norm vectors
        score = float(np.dot(e1, e2))
        score = max(-1.0, min(1.0, score))
        scores.append(score)
        labels.append(int(label))
        enroll_spks.append(str(enroll))   # proxy for speaker identity

    if skipped:
        logger.warning("Skipped %d pairs due to missing embeddings.", skipped)

    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    # ── Primary metrics ───────────────────────────────────────────────────
    eer, _eer_thr = _compute_eer(scores_arr, labels_arr)
    frr_at_far1 = _compute_frr_at_far(scores_arr, labels_arr, target_far=0.01)
    min_dcf = _compute_min_dcf(scores_arr, labels_arr)
    det_curve = _compute_det_curve(scores_arr, labels_arr)

    # ── Bootstrap CIs ─────────────────────────────────────────────────────
    boot_ci = _bootstrap_ci(scores_arr, labels_arr)
    spk_ci = _speaker_bootstrap_ci(scores_arr, labels_arr, enroll_spks)
    combined_ci = {**boot_ci, **spk_ci}

    # ── Per-bucket EER breakdown ──────────────────────────────────────────
    # For the main evaluate_model call we only have one bucket; callers who
    # want full per-bucket breakdown iterate over BucketedTrialGenerator.generate_all().
    per_bucket_eer: Dict[str, float] = {
        trial_list.bucket_name: float(eer),
    }

    result = EvalResult(
        eer=float(eer),
        frr_at_far1pct=float(frr_at_far1),
        min_dcf=float(min_dcf),
        per_bucket_eer=per_bucket_eer,
        bootstrap_ci=combined_ci,
        det_curve=det_curve,
        trial_list_sha256=trial_list.sha256,
        manifest_hash=m_hash,
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_label=model_label,
    )

    logger.info(
        "EER=%.4f  FRR@FAR1%%=%.4f  minDCF=%.4f  bucket=%s",
        eer, frr_at_far1, min_dcf, trial_list.bucket_name,
    )
    return result


# ===========================================================================
# Helpers
# ===========================================================================

def _is_test_manifest(path: str) -> bool:
    """Return True if the manifest path contains the substring 'test'."""
    return "test" in Path(path).as_posix().lower()


# ===========================================================================
# CLI
# ===========================================================================

def _build_embedding_fn(ckpt_path: str) -> Callable[[str], np.ndarray]:
    """Build an embedding function from a checkpoint.

    Supports loading:
      - B1 / AAA-SV checkpoints saved with state dict key "model" or "model_state_dict"
      - Pretrained baseline models via SpeechBrain
    """
    import torch
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization

    compute_features = Fbank(n_mels=80)
    mean_var_norm = InputNormalization(norm_type="sentence", std_norm=False)
    compute_features.to(device)
    mean_var_norm.to(device)

    model = ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        groups=[1, 1, 1, 1, 1],
        attention_channels=128,
        lin_neurons=192,
    )

    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        if "model" in state:
            state_dict = state["model"]
        elif "model_state_dict" in state:
            state_dict = state["model_state_dict"]
        else:
            state_dict = state
    else:
        state_dict = state

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    _SAMPLE_RATE = 16000

    def embedding_fn(wav_path: str) -> np.ndarray:
        import soundfile as sf

        try:
            data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(data.T)
        except Exception:
            wav, sr = torchaudio.load(wav_path)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != _SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, _SAMPLE_RATE)(wav)
        wav = wav.to(device)

        with torch.no_grad():
            feats = compute_features(wav)
            lens = torch.ones(wav.shape[0], device=device)
            feats = mean_var_norm(feats, lens)
            emb = model(feats)
            if emb.dim() == 3:
                emb = emb.squeeze(1)

        emb_np = emb.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb_np)
        if norm > 1e-9:
            emb_np = emb_np / norm
        return emb_np

    return embedding_fn


def _cli_main(argv: Optional[List[str]] = None) -> None:
    global FREEZE_ENABLED

    parser = argparse.ArgumentParser(
        description="AAA-SV Evaluation Toolkit — unified eval for B0/B1/AAA-SV"
    )
    parser.add_argument(
        "--manifest", required=True,
        help="Path to manifest CSV (columns: wav_path, speaker_id).",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Root directory prepended to wav_path values.",
    )
    parser.add_argument(
        "--model-ckpt", required=True,
        help="Path to model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file for the EvalResult.",
    )
    parser.add_argument(
        "--model-label", default="AAA-SV",
        choices=["B0", "B1", "AAA-SV"],
        help="Label identifying the model variant.",
    )
    parser.add_argument(
        "--freeze", action="store_true",
        help="Allow evaluation on test-tagged manifests.",
    )
    parser.add_argument(
        "--bucket", default="all",
        help="Which bucket to evaluate (default: all).",
    )
    args = parser.parse_args(argv)

    if args.freeze:
        FREEZE_ENABLED = True

    # Build trial list
    gen = BucketedTrialGenerator(
        manifest_csv=args.manifest,
        data_root=args.data_root,
    )
    all_buckets = gen.generate_all()
    if args.bucket not in all_buckets:
        raise ValueError(
            f"Bucket '{args.bucket}' not found. "
            f"Available: {list(all_buckets.keys())}"
        )
    trial_list = all_buckets[args.bucket]

    # Build embedding function
    embedding_fn = _build_embedding_fn(args.model_ckpt)

    # Run evaluation
    result = evaluate_model(
        embedding_fn=embedding_fn,
        trial_list=trial_list,
        freeze=args.freeze,
        model_label=args.model_label,
        manifest_path=args.manifest,
    )

    # Serialise and save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_dict = asdict(result)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result_dict, fh, indent=2, default=str)

    print(f"\nEvalResult written to: {output_path}")
    print(f"  EER        : {result.eer:.4f} ({result.eer*100:.2f}%)")
    print(f"  FRR@FAR1%  : {result.frr_at_far1pct:.4f}")
    print(f"  minDCF     : {result.min_dcf:.4f}")
    print(f"  model      : {result.model_label}")
    print(f"  run_id     : {result.run_id}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _cli_main()
