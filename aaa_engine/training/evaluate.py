# =============================================================
# training/evaluate.py
# AAA Engine - Training Pipeline
# Evaluate trained ECAPA-TDNN model using EER metric
# =============================================================
# Loads the trained model, extracts embeddings for test pairs,
# and computes Equal Error Rate (EER).
#
# EER is the standard metric for speaker verification:
#   - FAR (False Accept Rate) = wrongly accepted impostors
#   - FRR (False Reject Rate) = wrongly rejected genuine speakers
#   - EER = the point where FAR == FRR
#   - Lower EER = better model
#
# Usage:
#   cd aaa_engine
#   python -m training.evaluate
# =============================================================

import os
import csv
import argparse

import numpy as np
import torch
import soundfile as sf
from scipy.spatial.distance import cosine

# ── Configuration ────────────────────────────────────────────
MANIFEST_DIR   = os.path.join(os.path.dirname(__file__), 'manifests')
PAIRS_CSV      = os.path.join(MANIFEST_DIR, 'test_pairs.csv')
MODEL_DIR      = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'ecapa_sample', '1986', 'save'
)
SAMPLE_RATE    = 16000


def load_trained_model(model_dir: str):
    """
    Load the trained ECAPA-TDNN model from checkpoint.

    Reconstructs the model architecture and loads the best checkpoint
    weights from the training output directory.

    Args:
        model_dir: Path to the checkpoint save directory

    Returns:
        (embedding_model, compute_features, mean_var_norm) tuple
    """
    import speechbrain as sb
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.lobes.features import Fbank
    from speechbrain.processing.features import InputNormalization

    # Reconstruct model architecture (must match train_hparams.yaml)
    compute_features = Fbank(n_mels=80)
    embedding_model = ECAPA_TDNN(
        input_size=80,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        groups=[1, 1, 1, 1, 1],
        attention_channels=128,
        lin_neurons=192,
    )
    mean_var_norm = InputNormalization(norm_type='sentence', std_norm=False)

    # Find the best checkpoint
    model_dir = os.path.abspath(model_dir)
    ckpt_dirs = [
        d for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('CKPT')
    ]

    if not ckpt_dirs:
        raise FileNotFoundError(
            f"No checkpoints found in {model_dir}\n"
            f"Run training first: python -m training.train_ecapa training/train_hparams.yaml"
        )

    # Use the latest checkpoint (or the one kept as best)
    latest_ckpt = sorted(ckpt_dirs)[-1]
    ckpt_path = os.path.join(model_dir, latest_ckpt)

    print(f"  Loading checkpoint: {latest_ckpt}")

    # Load embedding model weights
    emb_ckpt = os.path.join(ckpt_path, 'embedding_model.ckpt')
    if os.path.exists(emb_ckpt):
        embedding_model.load_state_dict(
            torch.load(emb_ckpt, map_location='cpu')
        )
        print("  [OK] Embedding model weights loaded")
    else:
        raise FileNotFoundError(f"Embedding checkpoint not found: {emb_ckpt}")

    # Load normalizer weights if available
    norm_ckpt = os.path.join(ckpt_path, 'normalizer.ckpt')
    if os.path.exists(norm_ckpt):
        mean_var_norm.load_state_dict(
            torch.load(norm_ckpt, map_location='cpu'),
            strict=False
        )
        print("  [OK] Normalizer weights loaded")

    embedding_model.eval()
    return embedding_model, compute_features, mean_var_norm


def extract_embedding_from_file(
    filepath: str,
    embedding_model,
    compute_features,
    mean_var_norm
) -> np.ndarray:
    """
    Extract a 192-dim embedding from an audio file.

    Steps:
        1. Load audio file as waveform
        2. Extract Mel filterbank features
        3. Apply mean-variance normalization
        4. Run through ECAPA-TDNN → 192-dim embedding
        5. L2-normalize the embedding

    Args:
        filepath:         Path to WAV file
        embedding_model:  Loaded ECAPA-TDNN model
        compute_features: Fbank feature extractor
        mean_var_norm:    Input normalizer

    Returns:
        L2-normalized numpy array of shape (192,)
    """
    import torchaudio

    # Load audio
    waveform, sr = torchaudio.load(filepath)

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Extract features
    with torch.no_grad():
        feats = compute_features(waveform)
        lens = torch.tensor([1.0])
        feats = mean_var_norm(feats, lens)
        embedding = embedding_model(feats)

    # Flatten and L2 normalize
    emb_np = embedding.squeeze().numpy()
    norm = np.linalg.norm(emb_np)
    if norm > 0:
        emb_np = emb_np / norm

    return emb_np


def compute_eer(scores: list, labels: list) -> tuple:
    """
    Compute Equal Error Rate (EER) from similarity scores and labels.

    Sweeps through all possible thresholds to find the point
    where False Accept Rate (FAR) equals False Reject Rate (FRR).

    Args:
        scores: List of cosine similarity scores
        labels: List of ground truth labels (1=same, 0=different)

    Returns:
        (eer, threshold_at_eer) tuple
    """
    scores = np.array(scores)
    labels = np.array(labels)

    # Sort by scores
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Total positives and negatives
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        print("  [!] Cannot compute EER: need both positive and negative pairs")
        return 0.0, 0.0

    # Sweep thresholds
    best_eer = 1.0
    best_threshold = 0.0

    thresholds = np.unique(sorted_scores)
    # Add some intermediate thresholds for finer granularity
    if len(thresholds) > 1:
        fine_thresholds = np.linspace(
            thresholds[0], thresholds[-1], min(1000, len(thresholds) * 10)
        )
        thresholds = np.unique(np.concatenate([thresholds, fine_thresholds]))

    for threshold in thresholds:
        # Predictions: score >= threshold → ACCEPT (predict same speaker)
        predictions = (scores >= threshold).astype(int)

        # False Accept: predicted same, but actually different
        fa = np.sum((predictions == 1) & (labels == 0))
        far = fa / n_neg

        # False Reject: predicted different, but actually same
        fr = np.sum((predictions == 0) & (labels == 1))
        frr = fr / n_pos

        # EER is where FAR ~ FRR
        eer = abs(far - frr)
        if eer < best_eer:
            best_eer = eer
            best_threshold = threshold
            best_far = far
            best_frr = frr

    # The actual EER value is the average of FAR and FRR at the optimal point
    eer_value = (best_far / n_neg + best_frr / n_pos) / 2.0

    return eer_value, best_threshold


def evaluate(
    pairs_csv: str = PAIRS_CSV,
    model_dir: str = MODEL_DIR
) -> None:
    """
    Main evaluation pipeline.

    1. Load the trained model from checkpoint
    2. Read test pairs CSV
    3. Extract embeddings for all test audio files
    4. Compute cosine similarity for each pair
    5. Calculate and report EER

    Args:
        pairs_csv: Path to test_pairs.csv (label, wav1, wav2)
        model_dir: Path to training checkpoint directory
    """
    print("=" * 65)
    print("  AAA Engine - Model Evaluation (EER)")
    print("=" * 65)

    # Step 1: Load model
    print("\n[Step 1] Loading trained model...")
    embedding_model, compute_features, mean_var_norm = load_trained_model(
        model_dir
    )

    # Step 2: Read test pairs
    print("\n[Step 2] Reading test pairs...")
    pairs_csv = os.path.abspath(pairs_csv)

    if not os.path.exists(pairs_csv):
        raise FileNotFoundError(
            f"Test pairs not found: {pairs_csv}\n"
            f"Run 'python -m training.prepare_data' first."
        )

    pairs = []
    with open(pairs_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                'label': int(row['label']),
                'wav1': row['wav1'],
                'wav2': row['wav2'],
            })

    print(f"  Total pairs: {len(pairs)}")
    print(f"  Positive (same speaker): {sum(1 for p in pairs if p['label'] == 1)}")
    print(f"  Negative (diff speaker): {sum(1 for p in pairs if p['label'] == 0)}")

    # Step 3: Extract embeddings and compute scores
    print("\n[Step 3] Computing similarity scores...")

    # Cache embeddings to avoid redundant computation
    embedding_cache = {}
    scores = []
    labels = []

    for i, pair in enumerate(pairs):
        wav1 = pair['wav1']
        wav2 = pair['wav2']

        # Extract or retrieve cached embeddings
        if wav1 not in embedding_cache:
            embedding_cache[wav1] = extract_embedding_from_file(
                wav1, embedding_model, compute_features, mean_var_norm
            )
        if wav2 not in embedding_cache:
            embedding_cache[wav2] = extract_embedding_from_file(
                wav2, embedding_model, compute_features, mean_var_norm
            )

        emb1 = embedding_cache[wav1]
        emb2 = embedding_cache[wav2]

        # Cosine similarity
        similarity = 1.0 - cosine(emb1, emb2)
        similarity = float(np.clip(similarity, 0.0, 1.0))

        scores.append(similarity)
        labels.append(pair['label'])

        if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
            print(f"  Processed {i + 1}/{len(pairs)} pairs...")

    # Step 4: Compute EER
    print("\n[Step 4] Computing Equal Error Rate (EER)...")
    eer, threshold = compute_eer(scores, labels)

    # Score statistics
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Equal Error Rate (EER):     {eer * 100:.2f}%")
    print(f"  Threshold at EER:           {threshold:.4f}")
    print(f"  Total test pairs:           {len(pairs)}")
    print(f"  Unique embeddings cached:   {len(embedding_cache)}")
    print("-" * 65)
    print(f"  Positive pair scores:  mean={np.mean(pos_scores):.4f}, "
          f"std={np.std(pos_scores):.4f}")
    print(f"  Negative pair scores:  mean={np.mean(neg_scores):.4f}, "
          f"std={np.std(neg_scores):.4f}")
    print(f"  Score separation:      {np.mean(pos_scores) - np.mean(neg_scores):.4f}")
    print("=" * 65)

    if eer < 0.20:
        print("  [OK] Model is producing discriminative embeddings!")
    elif eer < 0.40:
        print("  ~ Model shows some discrimination. More training data may help.")
    else:
        print("  [!] High EER - model needs more training data or epochs.")


# ── Main Entry Point ─────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained ECAPA-TDNN model (EER)'
    )
    parser.add_argument(
        '--pairs-csv', type=str, default=PAIRS_CSV,
        help='Path to test_pairs.csv'
    )
    parser.add_argument(
        '--model-dir', type=str, default=MODEL_DIR,
        help='Path to training checkpoint directory'
    )
    args = parser.parse_args()

    evaluate(
        pairs_csv=args.pairs_csv,
        model_dir=args.model_dir
    )
