# =============================================================
# training/sv_eval.py
# AAA Engine — B0 Baseline Speaker Verification Evaluation
# =============================================================
# Zero-shot evaluation of the pretrained ECAPA-TDNN
# from speechbrain/spkrec-ecapa-voxceleb (unchanged weights).
#
# This script is FROZEN after Day 3. The pretrained model is used
# as-is — no fine-tuning, no weight updates. All reported numbers
# are on whatever trial list / manifest is passed in.
#
# Key invariants
#   • Manifest sha256 is verified against config['manifest_hash']
#     if the hash is provided — a mismatch is a hard failure.
#   • NO silence trimming (training/inference parity — see note).
#   • NO repeat-padding (clips shorter than 1 s are skipped).
#   • All randomness is seeded.
#   • Metrics marked TBD until measured on the full VoxCeleb1-O list.
#
# Usage:
#   cd d:/AAA_Layer1
#   python -m aaa_engine.training.sv_eval \
#       --config configs/eval_config.yaml \
#       --trial-list path/to/trials.txt \
#       --manifest path/to/manifest.jsonl
# =============================================================

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import yaml
from scipy.spatial.distance import cosine as cosine_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_EMB_DIM = 192
SAMPLE_RATE = 16000
DURATION_MIN_S = 1.0
DURATION_MAX_S = 10.0


# ---------------------------------------------------------------------------
# Library version logging
# ---------------------------------------------------------------------------
def _log_versions() -> Dict[str, str]:
    """Record and print key library versions."""
    import speechbrain
    import scipy

    versions = {
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "speechbrain": speechbrain.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    print("\nLibrary versions:")
    for lib, ver in versions.items():
        print(f"  {lib}: {ver}")
    return versions


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_baseline_model(savedir: Optional[str] = None):
    """
    Load the pretrained ECAPA-TDNN from HuggingFace Hub.

    The model weights are NOT modified in any way. This is the
    published speechbrain/spkrec-ecapa-voxceleb checkpoint, used
    here as the B0 baseline for zero-shot evaluation.

    Args:
        savedir: Local cache directory. If None, uses HF default cache.

    Returns:
        SpeakerRecognition model (eval mode, on selected device).
    """
    try:
        from speechbrain.inference.speaker import SpeakerRecognition
    except ImportError:
        from speechbrain.pretrained import SpeakerRecognition

    versions = _log_versions()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    run_opts = {"device": device}
    source = "speechbrain/spkrec-ecapa-voxceleb"

    logger.info("Loading %s ...", source)
    model = SpeakerRecognition.from_hparams(
        source=source,
        savedir=savedir,
        run_opts=run_opts,
    )
    model.eval()

    # ── Sanity check: embedding dimension ──────────────────────────────────
    # Run a dummy forward pass on a 3-second zero tensor to confirm
    # (a) the model loads without error,
    # (b) the output embedding dimension is exactly 192 (ECAPA-TDNN lin_neurons),
    # (c) the L2 norm of the output is ≈ 1.0 (model outputs normalised embeddings).
    dummy_wav = torch.zeros(1, SAMPLE_RATE * 3)
    with torch.no_grad():
        dummy_emb = model.encode_batch(dummy_wav)
    dummy_emb_np = dummy_emb.squeeze().cpu().numpy()

    actual_dim = dummy_emb_np.shape[-1]
    if actual_dim != EXPECTED_EMB_DIM:
        raise RuntimeError(
            f"Embedding dim mismatch: expected {EXPECTED_EMB_DIM}, "
            f"got {actual_dim}. Wrong model checkpoint?"
        )

    l2_norm = float(np.linalg.norm(dummy_emb_np))
    if abs(l2_norm - 1.0) > 0.05:
        logger.warning(
            "L2 norm of dummy embedding is %.5f (expected ≈ 1.0). "
            "Model may not be normalising internally — sv_eval will normalise.",
            l2_norm,
        )
    else:
        logger.info("L2 norm check OK: %.6f ≈ 1.0", l2_norm)

    print(
        "\nLoaded speechbrain/spkrec-ecapa-voxceleb unchanged\n"
        f"  emb_dim={actual_dim}  l2_norm={l2_norm:.6f}  device={device}"
    )
    return model, versions


# ---------------------------------------------------------------------------
# Audio preprocessing
# ---------------------------------------------------------------------------
def preprocess_audio(
    path: str,
    sample_rate: int = SAMPLE_RATE,
) -> Optional[torch.Tensor]:
    """
    Load and preprocess a single audio file.

    Processing steps (order matters):
        1. Load via torchaudio.load
        2. Stereo→mono (channel mean)
        3. Resample to `sample_rate` if needed
        4. Cast to float32

    Explicitly NOT applied:
        • Silence trimming — intentionally omitted to preserve
          training/inference parity.  Trimming during inference only
          would create a systematic mismatch versus the ECAPA-TDNN
          training pipeline, biasing embedding distributions.
        • Repeat-padding — clips shorter than DURATION_MIN_S are skipped
          rather than padded, to avoid artificially inflating clip content.

    Args:
        path:        Absolute path to audio file (.wav / .flac / .mp3).
        sample_rate: Target sample rate (default 16 000 Hz).

    Returns:
        1-D float32 tensor of shape (n_samples,), or None if skipped.
    """
    path = str(path)
    if not os.path.isfile(path):
        logger.warning("SKIP (file not found): %s", path)
        return None

    try:
        # Primary: soundfile (no torchcodec dependency, works on all platforms)
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        # soundfile returns (n_samples, n_channels) → transpose to (n_channels, n_samples)
        waveform = torch.from_numpy(data.T)
    except Exception:
        # Fallback: torchaudio (requires torchcodec on torchaudio>=2.5)
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as exc:
            logger.warning("SKIP (load error) %s: %s", path, exc)
            return None

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Float32
    waveform = waveform.to(torch.float32)

    # Validate duration
    duration_s = waveform.shape[-1] / sample_rate
    if not (DURATION_MIN_S <= duration_s <= DURATION_MAX_S):
        logger.warning(
            "SKIP (duration %.2f s OOB [%.1f, %.1f]): %s",
            duration_s, DURATION_MIN_S, DURATION_MAX_S, path,
        )
        return None

    return waveform.squeeze(0)  # → shape (n_samples,)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def extract_embedding(model, waveform: torch.Tensor) -> np.ndarray:
    """
    Extract an L2-normalised 192-dim embedding from a waveform.

    Args:
        model:    Loaded SpeakerRecognition model.
        waveform: 1-D float32 tensor of shape (n_samples,).

    Returns:
        float32 numpy array of shape (192,), L2-normalised.
    """
    # encode_batch expects (batch, time) — add batch dim
    wav_batch = waveform.unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(wav_batch)

    emb_np = emb.squeeze().cpu().numpy().astype(np.float32)

    # L2 normalise (model usually does this internally, but we enforce it)
    norm = np.linalg.norm(emb_np)
    if norm > 1e-9:
        emb_np = emb_np / norm

    assert emb_np.shape == (EXPECTED_EMB_DIM,), (
        f"Unexpected embedding shape: {emb_np.shape}"
    )
    return emb_np


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _cosine_score(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Return cosine *similarity* (not distance) in [0, 1]."""
    return float(np.clip(1.0 - cosine_distance(emb1, emb2), -1.0, 1.0))


def compute_eer(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute EER (%), threshold at EER, and FRR@FAR1%.

    Adapted from training/evaluate.py with:
      • EER returned as a *percentage* (multiply by 100).
      • frr_at_far1pct also returned.

    Args:
        scores: 1-D float array of cosine similarity scores.
        labels: 1-D int array; 1 = same speaker, 0 = different.

    Returns:
        (eer_pct, threshold_at_eer, frr_at_far1pct)
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))

    if n_pos == 0 or n_neg == 0:
        logger.warning("Cannot compute EER: need both positive and negative pairs.")
        return float("nan"), float("nan"), float("nan")

    # Build a fine threshold grid from unique score values
    unique_thresholds = np.unique(scores)
    if len(unique_thresholds) > 1:
        fine = np.linspace(
            unique_thresholds[0],
            unique_thresholds[-1],
            min(10_000, len(unique_thresholds) * 10),
        )
        thresholds = np.unique(np.concatenate([unique_thresholds, fine]))
    else:
        thresholds = unique_thresholds

    best_eer_abs = 1.0
    best_threshold = thresholds[0]
    best_far = 1.0
    best_frr = 1.0

    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        fa = int(np.sum((preds == 1) & (labels == 0)))
        fr = int(np.sum((preds == 0) & (labels == 1)))
        far = fa / n_neg
        frr = fr / n_pos
        if abs(far - frr) < best_eer_abs:
            best_eer_abs = abs(far - frr)
            best_threshold = thr
            best_far = far
            best_frr = frr

    eer_pct = ((best_far + best_frr) / 2.0) * 100.0
    frr_at_far1 = compute_frr_at_far(scores, labels, target_far=0.01)
    return eer_pct, float(best_threshold), frr_at_far1


def compute_frr_at_far(
    scores: np.ndarray,
    labels: np.ndarray,
    target_far: float = 0.01,
) -> float:
    """
    Sweep thresholds to find where FAR ≈ target_far, return FRR there.

    Args:
        scores:     Cosine similarity scores.
        labels:     Ground-truth labels (1=same, 0=diff).
        target_far: Target FAR (default 0.01 → FAR=1%).

    Returns:
        FRR (as a fraction, not percentage) at the threshold where
        FAR ≈ target_far.  Returns NaN if not computable.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    thresholds = np.unique(scores)
    best_frr = float("nan")
    best_far_diff = float("inf")

    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        fa = int(np.sum((preds == 1) & (labels == 0)))
        fr = int(np.sum((preds == 0) & (labels == 1)))
        far = fa / n_neg
        frr = fr / n_pos
        diff = abs(far - target_far)
        if diff < best_far_diff:
            best_far_diff = diff
            best_frr = frr

    return best_frr


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap confidence intervals for EER and FRR@FAR1%.

    Resamples trial *pairs* (with replacement) n_bootstrap times and
    computes EER and FRR@FAR1% for each resample.

    Args:
        scores:      Cosine similarity scores.
        labels:      Ground-truth labels.
        n_bootstrap: Number of bootstrap iterations (default 1000).
        seed:        Random seed for reproducibility.

    Returns:
        dict with keys:
          eer_mean, eer_std, eer_ci_low, eer_ci_high (all in %)
          frr_at_far1_mean, frr_at_far1_ci_low, frr_at_far1_ci_high (fractions)
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    n = len(scores)

    eer_samples = []
    frr_samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s_b = scores[idx]
        l_b = labels[idx]
        eer_b, _, frr_b = compute_eer(s_b, l_b)
        if not np.isnan(eer_b):
            eer_samples.append(eer_b)
        if not np.isnan(frr_b):
            frr_samples.append(frr_b)

    eer_arr = np.array(eer_samples)
    frr_arr = np.array(frr_samples)

    return {
        "eer_mean": float(np.mean(eer_arr)) if len(eer_arr) else float("nan"),
        "eer_std": float(np.std(eer_arr)) if len(eer_arr) else float("nan"),
        "eer_ci_low": float(np.percentile(eer_arr, 2.5)) if len(eer_arr) else float("nan"),
        "eer_ci_high": float(np.percentile(eer_arr, 97.5)) if len(eer_arr) else float("nan"),
        "frr_at_far1_mean": float(np.mean(frr_arr)) if len(frr_arr) else float("nan"),
        "frr_at_far1_ci_low": float(np.percentile(frr_arr, 2.5)) if len(frr_arr) else float("nan"),
        "frr_at_far1_ci_high": float(np.percentile(frr_arr, 97.5)) if len(frr_arr) else float("nan"),
        "n_bootstrap": n_bootstrap,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Trial list / manifest I/O
# ---------------------------------------------------------------------------
def _sha256_file(path: str) -> str:
    """Return hex sha256 of a file's contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trial_list(path: str) -> Tuple[List[Dict], str]:
    """
    Load a speaker verification trial list.

    Accepts two formats:
      • VoxCeleb-O whitespace-delimited:  ``label path1 path2``
        where label is 1 (same speaker) or 0 (different).
      • CSV with header row containing columns: label, wav1, wav2

    Args:
        path: Path to the trial file.

    Returns:
        (trials, file_sha256)
        trials: list of {'label': int, 'wav1': str, 'wav2': str}
    """
    path = str(path)
    file_hash = _sha256_file(path)
    trials = []

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        f.seek(0)

        # Detect CSV vs space-delimited
        if first_line.startswith("label") and "," in first_line:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                trials.append({
                    "label": int(row["label"]),
                    "wav1": row["wav1"].strip(),
                    "wav2": row["wav2"].strip(),
                })
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    logger.warning("Malformed trial line (expected 3 fields): %s", line)
                    continue
                trials.append({
                    "label": int(parts[0]),
                    "wav1": parts[1],
                    "wav2": parts[2],
                })

    logger.info("Loaded %d trials from %s (sha256=%s…)", len(trials), path, file_hash[:12])
    return trials, file_hash


def load_manifest(path: Optional[str]) -> Tuple[Dict, Optional[str]]:
    """
    Load a JSONL manifest file.

    Args:
        path: Path to the manifest JSONL file, or None.

    Returns:
        (records_dict, file_sha256_or_None)
        records_dict: keyed by 'audio_path'
    """
    if path is None:
        logger.info("No manifest provided — running without manifest context.")
        return {}, None

    path = str(path)
    file_hash = _sha256_file(path)
    records = {}

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Manifest line %d JSON error: %s", lineno, exc)
                continue
            audio_path = rec.get("audio_path", "")
            if audio_path:
                records[audio_path] = rec

    logger.info(
        "Loaded manifest: %d records, sha256=%s…", len(records), file_hash[:12]
    )
    return records, file_hash


# ---------------------------------------------------------------------------
# Main evaluation orchestrator
# ---------------------------------------------------------------------------
def run_evaluation(
    config: Dict,
    trial_list_path: str,
    manifest_path: Optional[str],
) -> Dict:
    """
    Full zero-shot baseline evaluation pipeline.

    Steps:
      1.  Verify manifest sha256 matches config['manifest_hash'] (hard fail).
      2.  Load model.
      3.  Load trial list → compute trial_hash.
      4.  Extract embeddings for all unique files (caching).
      5.  Compute cosine similarity scores.
      6.  Compute EER, FRR@FAR1%, bootstrap CI.
      7.  Return results dict.

    Args:
        config:          Parsed eval_config.yaml as dict.
        trial_list_path: Path to the trial list file.
        manifest_path:   Path to the JSONL manifest (or None).

    Returns:
        results dict suitable for save_results().
    """
    # ── 1. Manifest hash guard ──────────────────────────────────────────────
    manifest_records, manifest_hash = load_manifest(manifest_path)
    expected_hash = config.get("manifest_hash")

    if expected_hash and manifest_hash:
        if manifest_hash != expected_hash:
            raise RuntimeError(
                f"[SECURITY] Manifest sha256 mismatch!\n"
                f"  Expected : {expected_hash}\n"
                f"  Actual   : {manifest_hash}\n"
                "The manifest file has changed since the config was frozen. "
                "Do NOT proceed — investigate data integrity before re-running."
            )
        logger.info("[OK] Manifest sha256 verified.")
    elif expected_hash and not manifest_hash:
        raise RuntimeError(
            "Config specifies manifest_hash but no manifest was provided."
        )

    # ── 2. Load model ──────────────────────────────────────────────────────
    savedir = config.get("pretrained_savedir")
    model, lib_versions = load_baseline_model(savedir=savedir)

    # ── 3. Trial list ──────────────────────────────────────────────────────
    trials, trial_hash = load_trial_list(trial_list_path)
    n_pos = sum(1 for t in trials if t["label"] == 1)
    n_neg = sum(1 for t in trials if t["label"] == 0)
    logger.info("Trials: %d total, %d positive, %d negative", len(trials), n_pos, n_neg)

    # ── 4. Extract embeddings (with caching) ───────────────────────────────
    unique_files = list({t["wav1"] for t in trials} | {t["wav2"] for t in trials})
    logger.info("Unique audio files to embed: %d", len(unique_files))

    emb_cache: Dict[str, np.ndarray] = {}
    skipped_files = []
    t0 = time.time()

    for i, fpath in enumerate(unique_files):
        waveform = preprocess_audio(fpath, sample_rate=SAMPLE_RATE)
        if waveform is None:
            skipped_files.append(fpath)
            continue
        emb_cache[fpath] = extract_embedding(model, waveform)
        if (i + 1) % 100 == 0 or (i + 1) == len(unique_files):
            elapsed = time.time() - t0
            logger.info(
                "  Embedded %d/%d files (%.1f s elapsed)",
                i + 1, len(unique_files), elapsed,
            )

    if skipped_files:
        logger.warning("Skipped %d files (duration OOB or load error).", len(skipped_files))

    # ── 5. Compute scores ──────────────────────────────────────────────────
    scores = []
    labels = []
    skipped_trials = 0

    for trial in trials:
        e1 = emb_cache.get(trial["wav1"])
        e2 = emb_cache.get(trial["wav2"])
        if e1 is None or e2 is None:
            skipped_trials += 1
            continue
        scores.append(_cosine_score(e1, e2))
        labels.append(trial["label"])

    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.int32)
    logger.info(
        "Scored %d trial pairs (%d skipped due to missing embeddings).",
        len(scores), skipped_trials,
    )

    # ── 6. Metrics ─────────────────────────────────────────────────────────
    eer_pct, eer_thr, frr_at_far1 = compute_eer(scores_arr, labels_arr)

    n_boot = config.get("n_bootstrap", 1000)
    boot_seed = config.get("bootstrap_seed", 42)
    ci = bootstrap_ci(scores_arr, labels_arr, n_bootstrap=n_boot, seed=boot_seed)

    # ── 7. Bundle results ──────────────────────────────────────────────────
    results = {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": "speechbrain/spkrec-ecapa-voxceleb (pretrained, unchanged)",
        "lib_versions": lib_versions,
        "trial_list": {
            "path": str(trial_list_path),
            "sha256": trial_hash,
            "n_trials": len(trials),
            "n_scored": len(scores),
            "n_skipped": skipped_trials,
            "n_positive": n_pos,
            "n_negative": n_neg,
        },
        "manifest": {
            "path": str(manifest_path) if manifest_path else "none",
            "sha256": manifest_hash if manifest_hash else "none",
        },
        "metrics": {
            "eer_pct": eer_pct,
            "eer_threshold": eer_thr,
            "frr_at_far1pct": frr_at_far1,
        },
        "bootstrap_ci": ci,
        "skipped_files": skipped_files,
    }

    logger.info(
        "Results — EER: %.2f%%  FRR@FAR1%%: %.4f  "
        "95%% CI EER: [%.2f%%, %.2f%%]",
        eer_pct,
        frr_at_far1,
        ci["eer_ci_low"],
        ci["eer_ci_high"],
    )
    return results


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def save_results(results: Dict, run_id: str, output_dir: str) -> None:
    """
    Persist evaluation results.

    Writes two files inside ``{output_dir}/{run_id}/``:
      • ``baseline_smoke_result.md``  — human-readable report.
      • ``results.jsonl``             — machine-readable, append-mode.

    Args:
        results:    Dict returned by run_evaluation().
        run_id:     Unique run identifier (UUID string).
        output_dir: Root output directory (e.g. 'runs').
    """
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tl = results["trial_list"]
    mf = results["manifest"]
    m = results["metrics"]
    ci = results["bootstrap_ci"]
    v = results["lib_versions"]

    eer_val = f"{m['eer_pct']:.4f}" if not np.isnan(m["eer_pct"]) else "NaN"
    frr_val = f"{m['frr_at_far1pct'] * 100:.4f}" if not np.isnan(m["frr_at_far1pct"]) else "NaN"
    ci_lo = f"{ci['eer_ci_low']:.4f}" if not np.isnan(ci["eer_ci_low"]) else "NaN"
    ci_hi = f"{ci['eer_ci_high']:.4f}" if not np.isnan(ci["eer_ci_high"]) else "NaN"
    frr_ci_lo = f"{ci['frr_at_far1_ci_low'] * 100:.4f}" if not np.isnan(ci["frr_at_far1_ci_low"]) else "NaN"
    frr_ci_hi = f"{ci['frr_at_far1_ci_high'] * 100:.4f}" if not np.isnan(ci["frr_at_far1_ci_high"]) else "NaN"

    md_content = f"""\
# Baseline Smoke Result
run_id: {run_id}
timestamp: {results['timestamp']}
provenance: {results['provenance']}

## Versions
torch: {v.get('torch', 'unknown')}
torchaudio: {v.get('torchaudio', 'unknown')}
speechbrain: {v.get('speechbrain', 'unknown')}

## Trial List
path: {tl['path']}
sha256: {tl['sha256']}
n_trials: {tl['n_trials']}
n_positive: {tl['n_positive']}
n_negative: {tl['n_negative']}

## Manifest
path: {mf['path']}
sha256: {mf['sha256']}

## Results
EER: TBD  [not yet measured on full VoxCeleb1-O — subset result only]
  — subset n_trials: {tl['n_scored']}
  — subset EER: {eer_val}%
  — bootstrap 95% CI: [{ci_lo}%, {ci_hi}%]
FRR@FAR1%: {frr_val}%  CI: [{frr_ci_lo}%, {frr_ci_hi}%]

## Reference
Published EER on VoxCeleb1-O (full list, pretrained model): ~0.87%
IMPORTANT: A subset result is NOT comparable to the full-list number.
A large mismatch on the full list indicates a pipeline bug, not a model issue.
"""

    md_path = run_dir / "baseline_smoke_result.md"
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Wrote report: %s", md_path)

    jsonl_path = run_dir / "results.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(results, default=str) + "\n")
    logger.info("Appended to: %s", jsonl_path)


# ---------------------------------------------------------------------------
# Bucketed trial generator
# ---------------------------------------------------------------------------
def generate_trial_buckets(
    manifest: Dict,
    speakers: List[str],
    output_dir: str,
    seed: int = 42,
) -> Dict[str, str]:
    """
    Generate per-bucket trial files from a manifest.

    Bucket dimensions:
      • duration : 1, 2, 3, 5 seconds (rounded to nearest integer second)
      • snr      : 0, 5, 10, 20 dB (requires 'snr_db' field in manifest)
      • device   : unique device values (requires 'device' field)
      • codec    : unique codec values (requires 'codec' field)

    For each bucket, same-speaker and cross-speaker pairs are generated
    from the filtered manifest records. Each trial file is written to
    ``{output_dir}/trials_{bucket_dim}_{bucket_val}.txt`` in
    VoxCeleb-O format (``label path1 path2``).

    Speaker-level bootstrap (EER) groups trials by enroll speaker and
    resamples *speakers* rather than individual pairs — set seed here
    for reproducibility.

    Args:
        manifest:   Dict returned by load_manifest() (keyed by audio_path).
        speakers:   List of speaker IDs to include.
        output_dir: Directory to write trial files.
        seed:       Random seed for pair sampling.

    Returns:
        Dict mapping bucket_name → sha256 of written trial file.
    """
    rng = np.random.default_rng(seed)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    # Group manifest records by speaker
    spk_to_records: Dict[str, List[Dict]] = {}
    for rec in manifest.values():
        spk = rec.get("speaker_id")
        if spk and spk in speakers:
            spk_to_records.setdefault(spk, []).append(rec)

    speaker_list = sorted(spk_to_records.keys())

    def _make_pairs(filtered_records: Dict[str, List[Dict]]) -> List[Tuple[int, str, str]]:
        """Generate same-speaker (+) and cross-speaker (-) pairs."""
        pairs: List[Tuple[int, str, str]] = []
        spks = sorted(filtered_records.keys())

        # Same-speaker pairs
        for spk in spks:
            recs = filtered_records[spk]
            if len(recs) < 2:
                continue
            paths = [r["audio_path"] for r in recs]
            indices = list(range(len(paths)))
            rng.shuffle(indices)
            for i in range(0, len(indices) - 1, 2):
                pairs.append((1, paths[indices[i]], paths[indices[i + 1]]))

        # Cross-speaker pairs (roughly same count)
        n_cross = len(pairs)
        for _ in range(n_cross):
            if len(spks) < 2:
                break
            s1, s2 = rng.choice(spks, size=2, replace=False)
            r1 = rng.choice(filtered_records[s1])
            r2 = rng.choice(filtered_records[s2])
            pairs.append((0, r1["audio_path"], r2["audio_path"]))

        return pairs

    written: Dict[str, str] = {}

    # ── Duration buckets ───────────────────────────────────────────────────
    for dur_s in [1, 2, 3, 5]:
        filtered: Dict[str, List[Dict]] = {}
        for spk in speaker_list:
            recs = [
                r for r in spk_to_records[spk]
                if int(round(r.get("duration_s", 0))) == dur_s
            ]
            if recs:
                filtered[spk] = recs

        if not filtered:
            logger.info("Duration bucket %ds: no matching records — skipping.", dur_s)
            continue

        pairs = _make_pairs(filtered)
        trial_path = output_dir_p / f"trials_duration_{dur_s}s.txt"
        with open(trial_path, "w", encoding="utf-8") as f:
            for label, p1, p2 in pairs:
                f.write(f"{label} {p1} {p2}\n")

        h = _sha256_file(str(trial_path))
        written[f"duration_{dur_s}s"] = h
        logger.info(
            "Duration bucket %ds: %d pairs → %s (sha256=%s…)",
            dur_s, len(pairs), trial_path, h[:12],
        )

    # ── SNR buckets ────────────────────────────────────────────────────────
    # Only generated if at least one manifest record contains 'snr_db'
    has_snr = any("snr_db" in r for r in manifest.values())
    if not has_snr:
        logger.info("SNR metadata absent in manifest — skipping SNR buckets.")
    else:
        for snr_db in [0, 5, 10, 20]:
            filtered = {}
            for spk in speaker_list:
                recs = [
                    r for r in spk_to_records[spk]
                    if int(round(r.get("snr_db", -999))) == snr_db
                ]
                if recs:
                    filtered[spk] = recs
            if not filtered:
                logger.info("SNR bucket %ddB: no matching records — skipping.", snr_db)
                continue
            pairs = _make_pairs(filtered)
            trial_path = output_dir_p / f"trials_snr_{snr_db}dB.txt"
            with open(trial_path, "w", encoding="utf-8") as f:
                for label, p1, p2 in pairs:
                    f.write(f"{label} {p1} {p2}\n")
            h = _sha256_file(str(trial_path))
            written[f"snr_{snr_db}dB"] = h
            logger.info(
                "SNR bucket %ddB: %d pairs → %s (sha256=%s…)",
                snr_db, len(pairs), trial_path, h[:12],
            )

    # ── Device buckets ─────────────────────────────────────────────────────
    has_device = any(r.get("device") is not None for r in manifest.values())
    if not has_device:
        logger.info("Device metadata absent in manifest — skipping device buckets.")
    else:
        devices = sorted({
            r["device"] for r in manifest.values()
            if r.get("device") is not None
        })
        for dev in devices:
            filtered = {}
            for spk in speaker_list:
                recs = [r for r in spk_to_records[spk] if r.get("device") == dev]
                if recs:
                    filtered[spk] = recs
            if not filtered:
                continue
            pairs = _make_pairs(filtered)
            safe_dev = str(dev).replace(" ", "_").replace("/", "-")
            trial_path = output_dir_p / f"trials_device_{safe_dev}.txt"
            with open(trial_path, "w", encoding="utf-8") as f:
                for label, p1, p2 in pairs:
                    f.write(f"{label} {p1} {p2}\n")
            h = _sha256_file(str(trial_path))
            written[f"device_{safe_dev}"] = h
            logger.info(
                "Device bucket '%s': %d pairs → %s (sha256=%s…)",
                dev, len(pairs), trial_path, h[:12],
            )

    # ── Codec buckets ──────────────────────────────────────────────────────
    has_codec = any(r.get("codec") is not None for r in manifest.values())
    if not has_codec:
        logger.info("Codec metadata absent in manifest — skipping codec buckets.")
    else:
        codecs = sorted({
            r["codec"] for r in manifest.values()
            if r.get("codec") is not None
        })
        for codec in codecs:
            filtered = {}
            for spk in speaker_list:
                recs = [r for r in spk_to_records[spk] if r.get("codec") == codec]
                if recs:
                    filtered[spk] = recs
            if not filtered:
                continue
            pairs = _make_pairs(filtered)
            safe_codec = str(codec).replace(" ", "_").replace("/", "-")
            trial_path = output_dir_p / f"trials_codec_{safe_codec}.txt"
            with open(trial_path, "w", encoding="utf-8") as f:
                for label, p1, p2 in pairs:
                    f.write(f"{label} {p1} {p2}\n")
            h = _sha256_file(str(trial_path))
            written[f"codec_{safe_codec}"] = h
            logger.info(
                "Codec bucket '%s': %d pairs → %s (sha256=%s…)",
                codec, len(pairs), trial_path, h[:12],
            )

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot EER evaluation of speechbrain/spkrec-ecapa-voxceleb. "
            "Config is frozen after Day 3 — do not edit eval_config.yaml post-freeze."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to eval_config.yaml (frozen after Day 3).",
    )
    parser.add_argument(
        "--trial-list",
        required=True,
        help="Path to trial list file (VoxCeleb-O format or CSV).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Path to JSONL manifest file. Optional for smoke runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir from config.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Override auto-generated run UUID.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_file():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = args.output_dir or config.get("output_dir", "runs")
    results = run_evaluation(
        config=config,
        trial_list_path=args.trial_list,
        manifest_path=args.manifest,
    )

    run_id = args.run_id or results["run_id"]
    save_results(results, run_id=run_id, output_dir=output_dir)
    print(f"\nEvaluation complete. Run ID: {run_id}")
    print(f"  EER: {results['metrics']['eer_pct']:.4f}%")
    print(f"  FRR@FAR1%: {results['metrics']['frr_at_far1pct'] * 100:.4f}%")
    print(f"  Results saved to: {output_dir}/{run_id}/")


if __name__ == "__main__":
    main()
