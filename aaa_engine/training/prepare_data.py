# =============================================================
# training/prepare_data.py
# AAA Engine - Training Pipeline
# Prepare SpeechBrain-compatible CSV manifests from audio dataset
# =============================================================
# Scans a speaker dataset directory and creates:
#   - train.csv  (70% of utterances)
#   - dev.csv    (15% of utterances)
#   - test.csv   (15% of utterances)
#   - test_pairs.csv (positive + negative pairs for EER eval)
#
# Expected input structure:
#   dataset_dir/
#     speaker_0001/
#       utterance_01.wav
#       ...
#     speaker_0002/
#       ...
#
# CSV format (SpeechBrain standard):
#   ID, duration, wav, start, stop, spk_id
#
# Usage:
#   cd aaa_engine
#   python -m training.prepare_data
# =============================================================

import os
import csv
import random
import argparse
import soundfile as sf
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'sample_dataset')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), 'manifests')

# Split ratios
TRAIN_RATIO = 0.70
DEV_RATIO   = 0.15
TEST_RATIO  = 0.15

# Test pairs configuration
NUM_POSITIVE_PAIRS = 100   # Same-speaker pairs
NUM_NEGATIVE_PAIRS = 100   # Different-speaker pairs


def scan_dataset(dataset_dir: str) -> dict:
    """
    Scan dataset directory and collect all audio files per speaker.

    Args:
        dataset_dir: Root directory containing speaker subdirectories

    Returns:
        Dictionary mapping speaker_id → list of (filepath, duration) tuples
    """
    dataset_dir = os.path.abspath(dataset_dir)
    speakers = defaultdict(list)

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Run 'python -m training.generate_sample_dataset' first."
        )

    for spk_folder in sorted(os.listdir(dataset_dir)):
        spk_path = os.path.join(dataset_dir, spk_folder)
        if not os.path.isdir(spk_path):
            continue

        for audio_file in sorted(os.listdir(spk_path)):
            if not audio_file.endswith('.wav'):
                continue

            filepath = os.path.join(spk_path, audio_file)
            try:
                info = sf.info(filepath)
                duration = info.duration
                speakers[spk_folder].append((filepath, duration))
            except Exception as e:
                print(f"  [!] Skipping {filepath}: {e}")

    return dict(speakers)


def create_csv_manifest(
    entries: list,
    output_path: str,
    dataset_dir: str
) -> None:
    """
    Write a SpeechBrain-compatible CSV manifest file.

    CSV columns: ID, duration, wav, start, stop, spk_id

    The 'wav' column uses a {data_root} placeholder so SpeechBrain
    can resolve paths at runtime via the 'replacements' parameter.

    Args:
        entries:     List of (speaker_id, filepath, duration) tuples
        output_path: Path to write the CSV file
        dataset_dir: Root dataset directory (for relative paths)
    """
    dataset_dir = os.path.abspath(dataset_dir)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'duration', 'wav', 'start', 'stop', 'spk_id'])

        for speaker_id, filepath, duration in entries:
            # Create a relative path with {data_root} placeholder
            rel_path = os.path.relpath(filepath, dataset_dir)
            wav_path = os.path.join('{data_root}', rel_path).replace('\\', '/')

            # Unique ID from speaker + filename
            basename = os.path.splitext(os.path.basename(filepath))[0]
            entry_id = f"{speaker_id}_{basename}"

            # Start/stop in samples (full utterance)
            num_samples = int(duration * 16000)

            writer.writerow([
                entry_id,
                f"{duration:.4f}",
                wav_path,
                0,
                num_samples,
                speaker_id
            ])


def create_test_pairs(
    test_entries: list,
    output_path: str,
    num_positive: int = NUM_POSITIVE_PAIRS,
    num_negative: int = NUM_NEGATIVE_PAIRS,
    seed: int = 42
) -> None:
    """
    Generate test pairs CSV for verification evaluation.

    Creates positive (same speaker) and negative (different speaker) pairs
    for computing EER (Equal Error Rate).

    CSV columns: label, wav1, wav2
      - label: 1 = same speaker, 0 = different speaker

    Args:
        test_entries:  List of (speaker_id, filepath, duration) tuples
        output_path:   Path to write the pairs CSV
        num_positive:  Number of same-speaker pairs
        num_negative:  Number of different-speaker pairs
        seed:          Random seed
    """
    rng = random.Random(seed)

    # Group by speaker
    speaker_files = defaultdict(list)
    for spk_id, filepath, _ in test_entries:
        speaker_files[spk_id].append(filepath)

    # Filter speakers with at least 2 test utterances (for positive pairs)
    valid_speakers = [
        spk for spk, files in speaker_files.items() if len(files) >= 2
    ]

    if len(valid_speakers) < 2:
        print("  [!] Not enough speakers for test pairs. Skipping.")
        return

    pairs = []

    # Positive pairs (same speaker)
    for _ in range(num_positive):
        spk = rng.choice(valid_speakers)
        f1, f2 = rng.sample(speaker_files[spk], 2)
        pairs.append((1, f1, f2))

    # Negative pairs (different speakers)
    for _ in range(num_negative):
        spk1, spk2 = rng.sample(valid_speakers, 2)
        f1 = rng.choice(speaker_files[spk1])
        f2 = rng.choice(speaker_files[spk2])
        pairs.append((0, f1, f2))

    # Shuffle pairs
    rng.shuffle(pairs)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'wav1', 'wav2'])
        for label, wav1, wav2 in pairs:
            writer.writerow([label, wav1, wav2])


def prepare_data(
    dataset_dir: str = DATASET_DIR,
    output_dir: str = OUTPUT_DIR,
    seed: int = 42
) -> None:
    """
    Main data preparation pipeline.

    1. Scans the dataset directory for speakers and audio files
    2. Splits data into train/dev/test sets
    3. Creates CSV manifests for each split
    4. Creates test pairs CSV for verification evaluation

    Args:
        dataset_dir: Root directory containing speaker folders
        output_dir:  Directory to write CSV manifest files
        seed:        Random seed for reproducible splits
    """
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = os.path.abspath(output_dir)

    print("=" * 65)
    print("  AAA Engine - Data Preparation")
    print("=" * 65)
    print(f"  Dataset: {dataset_dir}")
    print(f"  Output:  {output_dir}")
    print(f"  Split:   {TRAIN_RATIO:.0%} train / {DEV_RATIO:.0%} dev / {TEST_RATIO:.0%} test")
    print("=" * 65)

    # Step 1: Scan dataset
    print("\n[Step 1] Scanning dataset directory...")
    speakers = scan_dataset(dataset_dir)
    total_utterances = sum(len(utts) for utts in speakers.values())
    print(f"  Found {len(speakers)} speakers, {total_utterances} total utterances")

    # Step 2: Split per speaker (stratified - each speaker appears in all splits)
    print("\n[Step 2] Creating train/dev/test splits...")
    rng = random.Random(seed)

    train_entries = []
    dev_entries = []
    test_entries = []

    for spk_id, utterances in sorted(speakers.items()):
        # Shuffle utterances for this speaker
        shuffled = list(utterances)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_dev = max(1, int(n * DEV_RATIO))
        # Remaining goes to test
        n_test = n - n_train - n_dev

        for filepath, dur in shuffled[:n_train]:
            train_entries.append((spk_id, filepath, dur))
        for filepath, dur in shuffled[n_train:n_train + n_dev]:
            dev_entries.append((spk_id, filepath, dur))
        for filepath, dur in shuffled[n_train + n_dev:]:
            test_entries.append((spk_id, filepath, dur))

    print(f"  Train: {len(train_entries)} utterances")
    print(f"  Dev:   {len(dev_entries)} utterances")
    print(f"  Test:  {len(test_entries)} utterances")

    # Step 3: Write CSV manifests
    print("\n[Step 3] Writing CSV manifests...")
    os.makedirs(output_dir, exist_ok=True)

    train_csv = os.path.join(output_dir, 'train.csv')
    dev_csv = os.path.join(output_dir, 'dev.csv')
    test_csv = os.path.join(output_dir, 'test.csv')
    pairs_csv = os.path.join(output_dir, 'test_pairs.csv')

    create_csv_manifest(train_entries, train_csv, dataset_dir)
    print(f"  [OK] {train_csv}")

    create_csv_manifest(dev_entries, dev_csv, dataset_dir)
    print(f"  [OK] {dev_csv}")

    create_csv_manifest(test_entries, test_csv, dataset_dir)
    print(f"  [OK] {test_csv}")

    # Step 4: Generate test pairs for EER evaluation
    print("\n[Step 4] Generating test pairs for evaluation...")
    create_test_pairs(test_entries, pairs_csv, seed=seed)
    print(f"  [OK] {pairs_csv}")

    print("\n" + "=" * 65)
    print("  [OK] Data preparation complete!")
    print("=" * 65)


# ── Main Entry Point ─────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare SpeechBrain CSV manifests from speaker dataset'
    )
    parser.add_argument(
        '--dataset-dir', type=str, default=DATASET_DIR,
        help='Root directory of speaker dataset'
    )
    parser.add_argument(
        '--output-dir', type=str, default=OUTPUT_DIR,
        help='Output directory for CSV manifests'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for splits'
    )
    args = parser.parse_args()

    prepare_data(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        seed=args.seed
    )
