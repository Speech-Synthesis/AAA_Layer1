# =============================================================
# training/export_model.py
# AAA Engine - Training Pipeline
# Export trained model to production directory
# =============================================================
# Copies the best checkpoint weights and generates the required
# configuration files for SpeechBrain's SpeakerRecognition.from_hparams()
# to load the model in production.
#
# After export:
#   - Set env var: ECAPA_MODEL_PATH=pretrained_models/ecapa-tdnn-custom
#   - The speaker_verifier.py will automatically use the trained model
#
# Usage:
#   cd aaa_engine
#   python -m training.export_model
# =============================================================

import os
import shutil
import argparse

# ── Configuration ────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'results', 'ecapa_sample', '1986', 'save'
)
EXPORT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'pretrained_models', 'ecapa-tdnn-custom'
)


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """
    Find the best (latest) checkpoint directory.

    SpeechBrain saves checkpoints as CKPT+<timestamp> directories.
    The checkpointer keeps only the best one by default.

    Args:
        checkpoint_dir: Path to the save/ directory from training

    Returns:
        Absolute path to the best checkpoint directory
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}\n"
            f"Run training first: python -m training.train_ecapa training/train_hparams.yaml"
        )

    ckpt_dirs = [
        d for d in os.listdir(checkpoint_dir)
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith('CKPT')
    ]

    if not ckpt_dirs:
        raise FileNotFoundError(
            f"No CKPT directories found in {checkpoint_dir}\n"
            f"Training may not have completed successfully."
        )

    # Sort by name (timestamp-based) and take the latest
    best = sorted(ckpt_dirs)[-1]
    return os.path.join(checkpoint_dir, best)


def generate_hyperparams_yaml(export_dir: str) -> None:
    """
    Generate a hyperparams.yaml that SpeechBrain's
    SpeakerRecognition.from_hparams() can use to load the model.

    This file defines the model architecture and the paths to
    checkpoint files, enabling the model to be loaded for inference
    without the full training configuration.

    Args:
        export_dir: Directory where the model is being exported
    """
    yaml_content = """\
# =============================================================
# AAA Engine - Custom-Trained ECAPA-TDNN
# Hyperparameters for SpeakerRecognition.from_hparams() loading
# =============================================================
# This model was fine-tuned from speechbrain/spkrec-ecapa-voxceleb
# on a custom speaker dataset using the AAA training pipeline.
# =============================================================

# Feature extraction
compute_features: !new:speechbrain.lobes.features.Fbank
    n_mels: 80

# ECAPA-TDNN encoder (same architecture as VoxCeleb pre-trained)
embedding_model: !new:speechbrain.lobes.models.ECAPA_TDNN.ECAPA_TDNN
    input_size: 80
    channels: [1024, 1024, 1024, 1024, 3072]
    kernel_sizes: [5, 3, 3, 3, 1]
    dilations: [1, 2, 3, 4, 1]
    groups: [1, 1, 1, 1, 1]
    attention_channels: 128
    lin_neurons: 192

# Mean-variance normalization
mean_var_norm: !new:speechbrain.processing.features.InputNormalization
    norm_type: sentence
    std_norm: False

# Label encoder (needed by SpeakerRecognition interface)
label_encoder: !new:speechbrain.dataio.encoder.CategoricalEncoder

# Module registry
modules:
    compute_features: !ref <compute_features>
    embedding_model: !ref <embedding_model>
    mean_var_norm: !ref <mean_var_norm>

# Pretrainer: maps loadable components to their checkpoint files
pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    loadables:
        embedding_model: !ref <embedding_model>
        mean_var_norm: !ref <mean_var_norm>
    paths:
        embedding_model: !ref <embedding_model.ckpt>
        mean_var_norm: !ref <mean_var_norm_emb.ckpt>
"""
    yaml_path = os.path.join(export_dir, 'hyperparams.yaml')
    if os.path.exists(yaml_path) or os.path.islink(yaml_path):
        try:
            os.remove(yaml_path)
        except Exception:
            pass
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)


def generate_custom_module(export_dir: str) -> None:
    """
    Generate custom.py module file required by SpeechBrain.

    This file is required for the from_hparams() loading mechanism
    to find custom model classes (though we use standard SpeechBrain
    classes, the file must exist).

    Args:
        export_dir: Directory where the model is being exported
    """
    custom_content = """\
# =============================================================
# AAA Engine - Custom-Trained ECAPA-TDNN
# Custom module file (required by SpeechBrain from_hparams)
# =============================================================
# This model uses standard SpeechBrain ECAPA-TDNN components.
# No custom classes are needed, but this file must exist for
# the SpeakerRecognition.from_hparams() loading mechanism.
# =============================================================

# Re-export standard SpeechBrain components
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN, Classifier
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
"""
    custom_path = os.path.join(export_dir, 'custom.py')
    if os.path.exists(custom_path) or os.path.islink(custom_path):
        try:
            os.remove(custom_path)
        except Exception:
            pass
    with open(custom_path, 'w', encoding='utf-8') as f:
        f.write(custom_content)


def export_model(
    checkpoint_dir: str = CHECKPOINT_DIR,
    export_dir: str = EXPORT_DIR
) -> None:
    """
    Export the trained model to the production directory.

    Steps:
        1. Find the best checkpoint from training
        2. Copy embedding_model.ckpt to export directory
        3. Copy normalizer.ckpt as mean_var_norm_emb.ckpt
        4. Generate hyperparams.yaml for from_hparams() loading
        5. Generate custom.py module file

    After export, set the environment variable:
        ECAPA_MODEL_PATH=pretrained_models/ecapa-tdnn-custom

    Args:
        checkpoint_dir: Path to training save/ directory
        export_dir:     Target directory for the exported model
    """
    export_dir = os.path.abspath(export_dir)

    print("=" * 65)
    print("  AAA Engine - Model Export")
    print("=" * 65)

    # Step 1: Find best checkpoint
    print("\n[Step 1] Finding best checkpoint...")
    best_ckpt = find_best_checkpoint(checkpoint_dir)
    print(f"  [OK] Best checkpoint: {os.path.basename(best_ckpt)}")

    # Step 2: Prepare export directory
    print("\n[Step 2] Preparing export directory...")
    os.makedirs(export_dir, exist_ok=True)
    print(f"  [OK] Export to: {export_dir}")

    # Step 3: Copy checkpoint files
    print("\n[Step 3] Copying model weights...")

    # Copy embedding model
    src_emb = os.path.join(best_ckpt, 'embedding_model.ckpt')
    dst_emb = os.path.join(export_dir, 'embedding_model.ckpt')
    if os.path.exists(src_emb):
        if os.path.exists(dst_emb) or os.path.islink(dst_emb):
            try:
                os.remove(dst_emb)
            except Exception:
                pass
        shutil.copy2(src_emb, dst_emb)
        size_mb = os.path.getsize(dst_emb) / (1024 * 1024)
        print(f"  [OK] embedding_model.ckpt ({size_mb:.1f} MB)")
    else:
        print(f"  [!] embedding_model.ckpt not found in checkpoint")

    # Copy normalizer
    src_norm = os.path.join(best_ckpt, 'normalizer.ckpt')
    dst_norm = os.path.join(export_dir, 'mean_var_norm_emb.ckpt')
    if os.path.exists(src_norm):
        if os.path.exists(dst_norm) or os.path.islink(dst_norm):
            try:
                os.remove(dst_norm)
            except Exception:
                pass
        shutil.copy2(src_norm, dst_norm)
        print(f"  [OK] mean_var_norm_emb.ckpt")
    else:
        print(f"  [!] normalizer.ckpt not found - using default normalization")

    # Copy classifier (optional, for reference)
    src_cls = os.path.join(best_ckpt, 'classifier.ckpt')
    dst_cls = os.path.join(export_dir, 'classifier.ckpt')
    if os.path.exists(src_cls):
        if os.path.exists(dst_cls) or os.path.islink(dst_cls):
            try:
                os.remove(dst_cls)
            except Exception:
                pass
        shutil.copy2(src_cls, dst_cls)
        print(f"  [OK] classifier.ckpt")

    # Copy label encoder if exists
    src_le = os.path.join(os.path.dirname(best_ckpt), 'label_encoder.txt')
    dst_le = os.path.join(export_dir, 'label_encoder.txt')
    if os.path.exists(src_le):
        if os.path.exists(dst_le) or os.path.islink(dst_le):
            try:
                os.remove(dst_le)
            except Exception:
                pass
        shutil.copy2(src_le, dst_le)
        print(f"  [OK] label_encoder.txt")

    # Step 4: Generate config files
    print("\n[Step 4] Generating configuration files...")
    generate_hyperparams_yaml(export_dir)
    print(f"  [OK] hyperparams.yaml")
    generate_custom_module(export_dir)
    print(f"  [OK] custom.py")

    # Summary
    print("\n" + "=" * 65)
    print("  [OK] Model exported successfully!")
    print("=" * 65)
    print(f"\n  Export location: {export_dir}")
    print(f"\n  To use the trained model in production:")
    print(f"    set ECAPA_MODEL_PATH=pretrained_models/ecapa-tdnn-custom")
    print(f"    set ECAPA_SAVE_DIR=pretrained_models/ecapa-tdnn-custom")
    print(f"\n  Then start the API server as usual:")
    print(f"    uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    print("=" * 65)


# ── Main Entry Point ─────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export trained ECAPA-TDNN model to production'
    )
    parser.add_argument(
        '--checkpoint-dir', type=str, default=CHECKPOINT_DIR,
        help='Path to training checkpoint (save/) directory'
    )
    parser.add_argument(
        '--export-dir', type=str, default=EXPORT_DIR,
        help='Target directory for the exported model'
    )
    args = parser.parse_args()

    export_model(
        checkpoint_dir=args.checkpoint_dir,
        export_dir=args.export_dir
    )
