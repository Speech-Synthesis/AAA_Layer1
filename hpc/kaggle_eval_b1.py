#!/usr/bin/env python3
"""
kaggle_eval_b1.py
=================
Kaggle-compatible evaluation script for the AAA-SV B1 checkpoint.

Usage
-----
  python hpc/kaggle_eval_b1.py --ckpt /path/to/checkpoint_best.pt

Arguments
---------
  --ckpt   Path to the B1 checkpoint (.pt file) produced by train_b1.py.
           The checkpoint dict is expected to contain at minimum:
             {"model": <embedding_model state dict>, ...}
  --device cpu | cuda  (default: auto-detect)
  --seed   RNG seed for the synthetic dataset (default: 42)
  --n-spk  Number of synthetic speakers (default: 20)
  --n-utt  Utterances per speaker (default: 5)
  --sr     Sample rate in Hz (default: 16000)
  --dur    Utterance duration in seconds (default: 2.0)

Output
------
Prints EER, FRR@FAR1%, and minDCF to stdout.
No VoxCeleb or any real audio dataset is required.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torchaudio


# ---------------------------------------------------------------------------
# Ensure repo root is importable when run from workspace root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Metric helpers (self-contained — no VoxCeleb dependency)
# ---------------------------------------------------------------------------

def _compute_eer(scores: np.ndarray, labels: np.ndarray):
    """Equal Error Rate via linear threshold sweep."""
    thresholds = np.unique(scores)
    best_eer, best_thr = 1.0, 0.0
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        n_neg = np.sum(labels == 0)
        n_pos = np.sum(labels == 1)
        far = fp / n_neg if n_neg > 0 else 0.0
        frr = fn / n_pos if n_pos > 0 else 0.0
        eer_cand = abs(far - frr)
        if eer_cand < abs(best_eer - best_eer) + 1e-9:
            best_eer = (far + frr) / 2.0
            best_thr = thr
    # More robust: interpolate crossing
    thresholds_sorted = np.sort(np.unique(scores))
    fars, frrs = [], []
    for thr in thresholds_sorted:
        preds = (scores >= thr).astype(int)
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        n_neg = max(np.sum(labels == 0), 1)
        n_pos = max(np.sum(labels == 1), 1)
        fars.append(fp / n_neg)
        frrs.append(fn / n_pos)
    fars = np.array(fars)
    frrs = np.array(frrs)
    diff = frrs - fars
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) > 0:
        i = sign_changes[0]
        d1, d2 = diff[i], diff[i + 1]
        alpha = d1 / (d1 - d2) if (d1 - d2) != 0 else 0.5
        eer = fars[i] + alpha * (fars[i + 1] - fars[i])
        thr = thresholds_sorted[i] + alpha * (thresholds_sorted[i + 1] - thresholds_sorted[i])
    else:
        # No crossing found — use midpoint
        eer = (fars[-1] + frrs[0]) / 2.0
        thr = thresholds_sorted[-1]
    return float(eer), float(thr)


def _compute_frr_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float = 0.01) -> float:
    """FRR at the threshold where FAR ≤ target_far."""
    thresholds = np.sort(np.unique(scores))[::-1]
    n_neg = max(np.sum(labels == 0), 1)
    n_pos = max(np.sum(labels == 1), 1)
    best_frr = 1.0
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        far = fp / n_neg
        frr = fn / n_pos
        if far <= target_far:
            best_frr = frr
            break
    return float(best_frr)


def _compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> float:
    """NIST SRE-style minDCF."""
    thresholds = np.sort(np.unique(scores))[::-1]
    n_neg = max(np.sum(labels == 0), 1)
    n_pos = max(np.sum(labels == 1), 1)
    min_dcf = float("inf")
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        far = fp / n_neg
        frr = fn / n_pos
        dcf = c_miss * p_target * frr + c_fa * (1 - p_target) * far
        if dcf < min_dcf:
            min_dcf = dcf
    return float(min_dcf) if min_dcf != float("inf") else 1.0


# ---------------------------------------------------------------------------
# Build embedding function (wraps eval_toolkit._build_embedding_fn)
# ---------------------------------------------------------------------------

def build_embedding_fn(ckpt_path: str, device: str):
    """Load B1 checkpoint and return an embedding function.

    The returned function accepts a path to a WAV file (or a temp file) and
    returns a L2-normalised 192-dim embedding as np.float32 ndarray.
    """
    from aaa_engine.evaluation.eval_toolkit import _build_embedding_fn
    # _build_embedding_fn auto-detects device; we set torch default device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU.")
        device = "cpu"
    # Temporarily override CUDA availability detection if needed
    return _build_embedding_fn(ckpt_path)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def make_synthetic_dataset(
    n_spk: int = 20,
    n_utt: int = 5,
    sr: int = 16000,
    dur: float = 2.0,
    seed: int = 42,
    tmpdir: str | None = None,
):
    """Generate Gaussian noise utterances as WAV files.

    Returns
    -------
    wav_paths : list[str]  — paths indexed by [speaker_idx * n_utt + utt_idx]
    speaker_ids : list[str]
    """
    rng = np.random.default_rng(seed)
    n_samples = int(sr * dur)
    wav_paths: list[str] = []
    speaker_ids: list[str] = []
    tmpdir = tmpdir or tempfile.mkdtemp(prefix="kaggle_eval_b1_")

    for spk in range(n_spk):
        spk_id = f"spk{spk:04d}"
        spk_dir = os.path.join(tmpdir, spk_id)
        os.makedirs(spk_dir, exist_ok=True)
        # Each speaker has a fixed "vocal tract" offset vector for identity
        spk_signature = rng.standard_normal(n_samples) * 0.3
        for utt in range(n_utt):
            noise = rng.standard_normal(n_samples).astype(np.float32) * 0.5
            wav_data = (spk_signature + noise).clip(-1.0, 1.0).astype(np.float32)
            wav_tensor = torch.from_numpy(wav_data).unsqueeze(0)  # [1, T]
            path = os.path.join(spk_dir, f"utt{utt:03d}.wav")
            torchaudio.save(path, wav_tensor, sr)
            wav_paths.append(path)
            speaker_ids.append(spk_id)

    return wav_paths, speaker_ids, tmpdir


def build_trial_list(wav_paths: list[str], speaker_ids: list[str], rng_seed: int = 42):
    """Build genuine and impostor trial pairs.

    For each speaker, pairs all utterances as genuine.
    Randomly samples an equal number of cross-speaker impostor pairs.
    """
    rng = np.random.default_rng(rng_seed + 1)

    # Group by speaker
    from collections import defaultdict
    spk_to_utts: dict[str, list[str]] = defaultdict(list)
    for path, spk in zip(wav_paths, speaker_ids):
        spk_to_utts[spk].append(path)

    trials: list[tuple[str, str, int]] = []

    # Genuine pairs
    for spk, utts in spk_to_utts.items():
        for i in range(len(utts)):
            for j in range(i + 1, len(utts)):
                trials.append((utts[i], utts[j], 1))

    n_genuine = len(trials)

    # Impostor pairs
    spk_list = list(spk_to_utts.keys())
    impostor_count = 0
    max_impostors = n_genuine * 3  # up to 3× impostors
    attempts = 0
    while impostor_count < n_genuine and attempts < max_impostors * 10:
        attempts += 1
        spk_a, spk_b = rng.choice(len(spk_list), size=2, replace=False)
        sa, sb = spk_list[spk_a], spk_list[spk_b]
        ua = rng.choice(spk_to_utts[sa])
        ub = rng.choice(spk_to_utts[sb])
        trials.append((ua, ub, 0))
        impostor_count += 1

    rng.shuffle(trials)
    return trials


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Kaggle-compatible B1 checkpoint evaluation on synthetic data"
    )
    parser.add_argument("--ckpt", required=True, help="Path to B1 checkpoint (.pt)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-spk", type=int, default=20, help="Synthetic speakers")
    parser.add_argument("--n-utt", type=int, default=5, help="Utterances per speaker")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--dur", type=float, default=2.0, help="Utterance duration (s)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device != "auto":
        device = args.device

    print(f"[kaggle_eval_b1] checkpoint : {args.ckpt}")
    print(f"[kaggle_eval_b1] device     : {device}")
    print(f"[kaggle_eval_b1] synthetic  : {args.n_spk} speakers × {args.n_utt} utterances")

    # --- Inspect checkpoint keys ---
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        print(f"[kaggle_eval_b1] ckpt keys  : {list(ckpt.keys())}")
    else:
        print(f"[kaggle_eval_b1] ckpt type  : {type(ckpt)}")

    # --- Build embedding function via eval_toolkit ---
    print("[kaggle_eval_b1] Building embedding function …")
    embedding_fn = build_embedding_fn(args.ckpt, device)
    print("[kaggle_eval_b1] Embedding function ready.")

    # --- Synthetic dataset ---
    print("[kaggle_eval_b1] Generating synthetic dataset …")
    wav_paths, speaker_ids, tmpdir = make_synthetic_dataset(
        n_spk=args.n_spk,
        n_utt=args.n_utt,
        sr=args.sr,
        dur=args.dur,
        seed=args.seed,
    )
    trials = build_trial_list(wav_paths, speaker_ids, rng_seed=args.seed)
    n_genuine = sum(1 for _, _, l in trials if l == 1)
    n_impostor = sum(1 for _, _, l in trials if l == 0)
    print(f"[kaggle_eval_b1] Trials: {len(trials)} total ({n_genuine} genuine, {n_impostor} impostor)")

    # --- Extract embeddings ---
    print("[kaggle_eval_b1] Extracting embeddings …")
    emb_cache: dict[str, np.ndarray] = {}
    all_paths = list({p for pair in trials for p in pair[:2]})
    for i, path in enumerate(all_paths):
        emb_cache[path] = embedding_fn(path)
        if (i + 1) % 20 == 0 or (i + 1) == len(all_paths):
            print(f"  {i+1}/{len(all_paths)} embeddings extracted", end="\r")
    print()

    # --- Score trials ---
    scores = np.array([
        float(np.dot(emb_cache[e], emb_cache[t]))
        for e, t, _ in trials
    ])
    labels = np.array([l for _, _, l in trials])

    # --- Compute metrics ---
    eer, eer_thr = _compute_eer(scores, labels)
    frr_at_far1 = _compute_frr_at_far(scores, labels, target_far=0.01)
    min_dcf = _compute_min_dcf(scores, labels)

    print()
    print("=" * 50)
    print("  AAA-SV B1 Evaluation Results (synthetic)")
    print("=" * 50)
    print(f"  EER         : {eer:.4f}  ({eer*100:.2f}%)")
    print(f"  FRR@FAR1%   : {frr_at_far1:.4f}")
    print(f"  minDCF      : {min_dcf:.4f}")
    print(f"  EER thr     : {eer_thr:.4f}")
    print("=" * 50)
    print(f"  [note] Synthetic Gaussian noise — metrics reflect random baseline.")
    print(f"  Temp files at: {tmpdir}")

    # Cleanup is left intentional so caller can inspect files if needed
    return 0


if __name__ == "__main__":
    sys.exit(main())
