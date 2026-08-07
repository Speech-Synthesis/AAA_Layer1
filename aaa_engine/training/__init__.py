# =============================================================
# AAA Engine — Training Pipeline
# Package Initialization
# =============================================================
# This package contains the complete training pipeline for
# fine-tuning the ECAPA-TDNN speaker verification model.
#
# Modules:
#   generate_sample_dataset — Create synthetic speaker audio
#   prepare_data            — Build SpeechBrain CSV manifests
#   train_ecapa             — Fine-tune ECAPA-TDNN model
#   evaluate                — Compute EER on test pairs
#   export_model            — Export trained model to production
# =============================================================
