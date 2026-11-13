# =============================================================
# aaa_engine/calibration/calibrator.py
# AAA Engine – Score Calibration Module   [Phase 3-C]
# =============================================================
#
# Implements (spec AAA-SV.md §20):
#   • ASNorm  — Adaptive Score Normalisation (Top-K cohort)
#   • LogisticCalibrator — logistic regression calibrator; minimises Cllr
#   • select_threshold   — FAR-based operating-point selection
#   • save_calibration / load_calibration — JSON artefact I/O
#
# Dependencies: stdlib + numpy + scipy only (no sklearn).
# =============================================================

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _l2_norm(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised version of v (safe against zero-norm)."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(np.dot(_l2_norm(a), _l2_norm(b)))


def _topk_cosine_stats(
    query: np.ndarray,
    cohort: np.ndarray,
    K: int,
) -> tuple[float, float]:
    """
    Compute mean and std of the top-K cosine similarities between *query*
    and each row of *cohort*.

    Parameters
    ----------
    query  : 1-D embedding (D,)
    cohort : 2-D array   (N, D)
    K      : number of top cohort vectors to retain

    Returns
    -------
    (mean_topk, std_topk)
    """
    q_norm = _l2_norm(query)
    # Normalise each cohort row
    norms = np.linalg.norm(cohort, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    cohort_norm = cohort / norms

    sims = cohort_norm @ q_norm  # (N,)

    actual_k = min(K, len(sims))
    top_indices = np.argpartition(sims, -actual_k)[-actual_k:]
    top_sims = sims[top_indices]

    mean = float(np.mean(top_sims))
    std = float(np.std(top_sims, ddof=0))
    if std < 1e-12:
        std = 1e-12  # guard against division by zero
    return mean, std


# ──────────────────────────────────────────────────────────────────────────────
# AS-Norm
# ──────────────────────────────────────────────────────────────────────────────

class ASNorm:
    """
    Adaptive Score Normalisation (Top-K).

    Parameters
    ----------
    cohort_embeddings : np.ndarray of shape (N, D)
        Embeddings drawn exclusively from **training** speakers.
        Never include validation or test speakers.
    K : int
        Number of top cohort vectors used per normalisation (default 300).
    """

    def __init__(self, cohort_embeddings: np.ndarray, K: int = 300) -> None:
        if cohort_embeddings.ndim != 2:
            raise ValueError(
                "cohort_embeddings must be 2-D, got shape "
                f"{cohort_embeddings.shape}"
            )
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.cohort = np.asarray(cohort_embeddings, dtype=np.float32)
        self.K = K

    # ------------------------------------------------------------------
    def normalize(
        self,
        enroll_emb: np.ndarray,
        test_emb: np.ndarray,
    ) -> float:
        """
        Compute AS-Norm score for one enrol/test pair.

        raw_score = cosine_similarity(enroll_emb, test_emb)

        norm_score = 0.5 * (
            (raw_score - mean_e) / std_e +
            (raw_score - mean_t) / std_t
        )

        where mean_e / std_e are the Top-K statistics of enrol vs cohort
        and   mean_t / std_t are the Top-K statistics of test vs cohort.

        Parameters
        ----------
        enroll_emb : 1-D enrolment embedding
        test_emb   : 1-D test embedding

        Returns
        -------
        Normalised score (float).
        """
        enroll_emb = np.asarray(enroll_emb, dtype=np.float32).ravel()
        test_emb = np.asarray(test_emb, dtype=np.float32).ravel()

        raw_score = _cosine_similarity(enroll_emb, test_emb)
        mean_e, std_e = _topk_cosine_stats(enroll_emb, self.cohort, self.K)
        mean_t, std_t = _topk_cosine_stats(test_emb, self.cohort, self.K)

        norm_score = 0.5 * (
            (raw_score - mean_e) / std_e +
            (raw_score - mean_t) / std_t
        )
        return float(norm_score)


# ──────────────────────────────────────────────────────────────────────────────
# Cllr helper
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def compute_cllr(log_odds: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Cllr (log-likelihood-ratio cost) from log-odds and binary labels.

    Cllr = 0.5 * (
        mean_{positives} [ log2(1 + exp(-log_odds_i)) ]  +
        mean_{negatives} [ log2(1 + exp(+log_odds_i)) ]
    )

    Parameters
    ----------
    log_odds : 1-D array — log-odds = log P(target) / P(non-target)
    labels   : 1-D binary array (1 = genuine, 0 = impostor)

    Returns
    -------
    Cllr (float; 0 = perfect, 1 = chance uninformative system)
    """
    log_odds = np.asarray(log_odds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    pos_mask = labels == 1
    neg_mask = labels == 0

    _ln2 = math.log(2)

    def _softplus_neg(x: np.ndarray) -> np.ndarray:
        """log(1 + exp(-x)) computed stably."""
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(-x, 0)

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        raise ValueError("labels must contain both positive (1) and negative (0) samples")

    cost_pos = np.mean(_softplus_neg(log_odds[pos_mask])) / _ln2
    cost_neg = np.mean(_softplus_neg(-log_odds[neg_mask])) / _ln2

    return float(0.5 * (cost_pos + cost_neg))


# ──────────────────────────────────────────────────────────────────────────────
# Logistic Calibrator
# ──────────────────────────────────────────────────────────────────────────────

_FEATURE_NAMES_WITH_Q = [
    "normalized_score",
    "q",
    "log_duration",
    "snr_norm",
    "channel_class",
]
_FEATURE_NAMES_NO_Q = [
    "normalized_score",
    "log_duration",
    "snr_norm",
    "channel_class",
]


class LogisticCalibrator:
    """
    Logistic calibrator trained to minimise Cllr (spec §20).

    Feature vector (use_q=True):
        [normalized_score, q, log(duration_seconds), snr_db/20, channel_class]

    Feature vector (use_q=False):
        [normalized_score, log(duration_seconds), snr_db/20, channel_class]

    Parameters
    ----------
    use_q : bool
        Whether to include the stability head output q in the feature vector.
    """

    def __init__(self, use_q: bool = True) -> None:
        self.use_q = use_q
        self.feature_names: list[str] = (
            _FEATURE_NAMES_WITH_Q if use_q else _FEATURE_NAMES_NO_Q
        )
        self._coef: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._cllr: Optional[float] = None

    # ------------------------------------------------------------------
    def _build_features(self, features: np.ndarray) -> np.ndarray:
        """Return (N, n_features) float64 array, validated."""
        X = np.asarray(features, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        expected = len(self.feature_names)
        if X.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} features {self.feature_names}, "
                f"got {X.shape[1]}"
            )
        return X

    # ------------------------------------------------------------------
    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticCalibrator":
        """
        Fit the logistic calibrator by minimising Cllr.

        Parameters
        ----------
        features : (N, n_features) array — pre-built feature vectors
        labels   : (N,) binary array (1=genuine, 0=impostor)

        Returns
        -------
        self
        """
        X = self._build_features(features)
        y = np.asarray(labels, dtype=np.int32)

        n_feat = X.shape[1]
        # Initial params: [coef_0, ..., coef_{n-1}, intercept]
        x0 = np.zeros(n_feat + 1)

        def _objective(params: np.ndarray) -> float:
            coef = params[:n_feat]
            intercept = params[n_feat]
            log_odds = X @ coef + intercept
            return compute_cllr(log_odds, y)

        result = minimize(
            _objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-10, "gtol": 1e-7},
        )

        self._coef = result.x[:n_feat]
        self._intercept = float(result.x[n_feat])

        # Store achieved Cllr on training data
        log_odds_train = X @ self._coef + self._intercept
        self._cllr = compute_cllr(log_odds_train, y)

        return self

    # ------------------------------------------------------------------
    def _make_feature_vector(
        self,
        score: float,
        q: Optional[float] = None,
        log_duration: Optional[float] = None,
        snr_norm: Optional[float] = None,
        channel_class: Optional[float] = None,
    ) -> np.ndarray:
        """
        Construct the feature vector from individual fields.

        Missing optional fields default to 0.0 (safe neutral values).
        """
        if self.use_q:
            vec = np.array([
                float(score),
                float(q) if q is not None else 0.0,
                float(log_duration) if log_duration is not None else 0.0,
                float(snr_norm) if snr_norm is not None else 0.0,
                float(channel_class) if channel_class is not None else 0.0,
            ], dtype=np.float64)
        else:
            vec = np.array([
                float(score),
                float(log_duration) if log_duration is not None else 0.0,
                float(snr_norm) if snr_norm is not None else 0.0,
                float(channel_class) if channel_class is not None else 0.0,
            ], dtype=np.float64)
        return vec

    # ------------------------------------------------------------------
    def calibrate(
        self,
        score: float,
        q: Optional[float] = None,
        log_duration: Optional[float] = None,
        snr_norm: Optional[float] = None,
        channel_class: Optional[float] = None,
    ) -> float:
        """
        Calibrate a single score.

        Parameters
        ----------
        score         : AS-Norm (or raw cosine) score
        q             : stability head output ∈ [0,1] (optional)
        log_duration  : log(duration_seconds)
        snr_norm      : snr_db / 20.0
        channel_class : 0=clean, 1=phone, 2=mic

        Returns
        -------
        Calibrated log-odds (float); positive → target, negative → non-target.
        """
        if self._coef is None:
            raise RuntimeError("Calibrator has not been fitted yet. Call fit() first.")

        vec = self._make_feature_vector(score, q, log_duration, snr_norm, channel_class)
        log_odds = float(vec @ self._coef) + self._intercept
        return log_odds

    # ------------------------------------------------------------------
    @property
    def cllr(self) -> Optional[float]:
        """Cllr achieved on the training data (available after fit)."""
        return self._cllr

    @property
    def coef_(self) -> Optional[np.ndarray]:
        return self._coef

    @property
    def intercept_(self) -> float:
        return self._intercept


# ──────────────────────────────────────────────────────────────────────────────
# Threshold Selection
# ──────────────────────────────────────────────────────────────────────────────

def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    target_far: float = 0.01,
) -> dict:
    """
    Select the decision threshold that achieves FAR ≤ *target_far* on the
    provided calibration trials.

    Parameters
    ----------
    scores     : 1-D array of calibrated scores (or log-odds)
    labels     : 1-D binary array (1=genuine, 0=impostor)
    target_far : desired False-Accept Rate upper bound (default 0.01)

    Returns
    -------
    dict with keys:
        threshold     : float
        realized_far  : float
        realized_frr  : float
        n_impostor    : int
        n_genuine     : int
        [validated]   : bool  (only when target_far == 0.001)
        [warning]     : str   (only when n_impostor < 3000 and target_far == 0.001)
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]

    n_genuine = int(len(genuine_scores))
    n_impostor = int(len(impostor_scores))

    if n_genuine == 0:
        raise ValueError("No genuine trials found in labels.")
    if n_impostor == 0:
        raise ValueError("No impostor trials found in labels.")

    # Candidate thresholds: all unique score values, sorted descending
    # We sweep thresholds from high to low; at each step FAR increases.
    candidates = np.sort(np.unique(scores))[::-1]

    best_threshold = float(candidates[0])
    best_far = 0.0
    best_frr = 1.0

    for t in candidates:
        far = float(np.mean(impostor_scores >= t))
        if far <= target_far:
            frr = float(np.mean(genuine_scores < t))
            best_threshold = float(t)
            best_far = far
            best_frr = frr
        else:
            # Once FAR exceeds target, stop (we want FAR ≤ target_far)
            break

    result: dict = {
        "threshold": best_threshold,
        "realized_far": best_far,
        "realized_frr": best_frr,
        "n_impostor": n_impostor,
        "n_genuine": n_genuine,
    }

    if math.isclose(target_far, 0.001):
        if n_impostor < 3000:
            result["validated"] = False
            result["warning"] = (
                "< 3000 independent speaker-level impostor trials"
            )
        else:
            result["validated"] = True

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Manifest Guard
# ──────────────────────────────────────────────────────────────────────────────

_FORBIDDEN_MANIFEST_TOKENS = ("train", "val", "test")


def _guard_manifest(manifest_path: str) -> None:
    """
    Raise ValueError if the manifest filename contains a forbidden token
    ('train', 'val', 'test').  The calibration manifest must be named 'cal.*'.
    """
    basename = os.path.basename(manifest_path).lower()
    for token in _FORBIDDEN_MANIFEST_TOKENS:
        if token in basename:
            raise ValueError(
                f"Calibration manifest path '{manifest_path}' contains "
                f"forbidden token '{token}'. "
                "Only 'cal' manifests are permitted for calibration "
                "to prevent data leakage."
            )


# ──────────────────────────────────────────────────────────────────────────────
# Calibration JSON I/O
# ──────────────────────────────────────────────────────────────────────────────

def save_calibration(calibrator: LogisticCalibrator, path: str, **kwargs) -> None:
    """
    Persist calibration artefact to *path* as JSON (spec §20).

    Required kwargs (passed through from the CLI or calling code):
        model_label        : str
        as_norm_K          : int
        threshold          : float
        realized_far       : float
        cohort_size        : int
        cal_manifest_hash  : str (SHA-256 hex of calibration manifest)

    Parameters
    ----------
    calibrator : fitted LogisticCalibrator
    path       : destination file path (e.g. 'runs/.../calibration.json')
    **kwargs   : extra metadata fields (see above)
    """
    if calibrator.coef_ is None:
        raise RuntimeError("Calibrator must be fitted before saving.")

    record: dict = {
        "model_label": kwargs.get("model_label", "aaa_sv"),
        "as_norm_K": int(kwargs.get("as_norm_K", 300)),
        "threshold": float(kwargs.get("threshold", 0.0)),
        "realized_far": float(kwargs.get("realized_far", float("nan"))),
        "cllr": float(calibrator.cllr) if calibrator.cllr is not None else None,
        "feature_names": calibrator.feature_names,
        "logistic_coef": calibrator.coef_.tolist(),
        "logistic_intercept": float(calibrator.intercept_),
        "cohort_size": int(kwargs.get("cohort_size", 0)),
        "cal_manifest_hash": kwargs.get("cal_manifest_hash", ""),
        "timestamp": kwargs.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    }

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)


def load_calibration(path: str) -> dict:
    """
    Load a calibration artefact from *path*.

    Returns
    -------
    dict with the same fields that save_calibration() wrote.
    """
    with open(path, "r", encoding="utf-8") as fh:
        record = json.load(fh)
    return record


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AAA-SV Calibration: fit AS-Norm + logistic calibrator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cal-manifest",
        required=True,
        help="Path to calibration trial manifest CSV "
             "(columns: enroll_path, test_path, label). "
             "Must be a 'cal' manifest (not train/val/test).",
    )
    parser.add_argument(
        "--cohort-embeddings",
        required=True,
        help="Path to .npy file with cohort embeddings (N, D) from training speakers.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for calibration.json artefact.",
    )
    parser.add_argument(
        "--as-norm-k",
        type=int,
        default=300,
        help="Top-K for AS-Norm.",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=0.01,
        help="Target FAR for threshold selection.",
    )
    parser.add_argument(
        "--model-label",
        default="aaa_sv",
        help="Label identifying this model checkpoint.",
    )
    parser.add_argument(
        "--no-q",
        action="store_true",
        help="Disable stability head q feature in calibrator.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # ── 1. Guard manifest ────────────────────────────────────────────────────
    _guard_manifest(args.cal_manifest)

    # ── 2. Load cohort embeddings ────────────────────────────────────────────
    cohort_embs = np.load(args.cohort_embeddings)
    if cohort_embs.ndim != 2:
        raise ValueError(
            f"cohort_embeddings must be 2-D, got shape {cohort_embs.shape}"
        )

    # ── 3. Load manifest ─────────────────────────────────────────────────────
    import csv

    trials = []
    with open(args.cal_manifest, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            trials.append(row)

    if not trials:
        raise ValueError(f"Manifest '{args.cal_manifest}' is empty.")

    cal_hash = _sha256_of_file(args.cal_manifest)

    # ── 4. Build AS-Norm + compute normalised scores ─────────────────────────
    asnorm = ASNorm(cohort_embs, K=args.as_norm_k)

    # For the CLI we expect embeddings to be precomputed and stored;
    # in production the manifest would point to .npy embedding files or
    # provide raw paths that the embedding extractor would process.
    # Here we implement the standard pattern: manifest has columns
    #   enroll_emb_path, test_emb_path, label
    # (or just: label, with embeddings inlined — depends on run config).
    # We handle both: try 'enroll_emb_path'/'test_emb_path' first, then
    # fall back to raw cosine from a 'score' column.

    normalised_scores = []
    labels = []

    for row in trials:
        label = int(row.get("label", row.get("target", 0)))
        labels.append(label)

        if "enroll_emb_path" in row and "test_emb_path" in row:
            enroll_emb = np.load(row["enroll_emb_path"])
            test_emb = np.load(row["test_emb_path"])
            norm_score = asnorm.normalize(enroll_emb, test_emb)
        elif "score" in row:
            norm_score = float(row["score"])
        else:
            raise ValueError(
                "Manifest must have either 'enroll_emb_path'/'test_emb_path' "
                "columns or a pre-computed 'score' column."
            )
        normalised_scores.append(norm_score)

    normalised_scores_arr = np.array(normalised_scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)

    # ── 5. Build feature matrix ───────────────────────────────────────────────
    use_q = not args.no_q
    cal = LogisticCalibrator(use_q=use_q)

    # CLI mode: only normalized_score is available from manifest;
    # other features default to 0.  The caller can extend this by
    # providing richer manifests.
    n = len(normalised_scores)
    n_feat = len(cal.feature_names)
    X = np.zeros((n, n_feat), dtype=np.float64)
    X[:, 0] = normalised_scores_arr  # first feature is always normalized_score

    # Optionally fill q, log_duration, snr_norm, channel_class if present
    _col_map = {
        "q": 1 if use_q else None,
        "log_duration": 2 if use_q else 1,
        "snr_norm": 3 if use_q else 2,
        "channel_class": 4 if use_q else 3,
    }
    for col_name, col_idx in _col_map.items():
        if col_idx is not None and col_name in (trials[0] if trials else {}):
            X[:, col_idx] = [float(r.get(col_name, 0.0)) for r in trials]

    # ── 6. Fit calibrator ─────────────────────────────────────────────────────
    cal.fit(X, labels_arr)

    # ── 7. Select threshold ───────────────────────────────────────────────────
    calibrated = X @ cal.coef_ + cal.intercept_
    thr_info = select_threshold(calibrated, labels_arr, target_far=args.target_far)

    # ── 8. Save artefact ──────────────────────────────────────────────────────
    save_calibration(
        cal,
        args.output,
        model_label=args.model_label,
        as_norm_K=args.as_norm_k,
        threshold=thr_info["threshold"],
        realized_far=thr_info["realized_far"],
        cohort_size=len(cohort_embs),
        cal_manifest_hash=cal_hash,
    )

    print(f"[calibrator] Saved calibration artefact → {args.output}")
    print(f"             Cllr        = {cal.cllr:.4f}")
    print(f"             Threshold   = {thr_info['threshold']:.4f}")
    print(f"             Realized FAR= {thr_info['realized_far']:.4f}")
    print(f"             Realized FRR= {thr_info['realized_frr']:.4f}")


if __name__ == "__main__":
    main()
