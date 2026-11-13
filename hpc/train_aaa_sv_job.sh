#!/bin/bash
# =============================================================
# hpc/train_aaa_sv_job.sh
# PBS job script — AAA-SV Joint Training (B1 + Stability Head q)
# =============================================================
# Submit AFTER checkpoint_best.pt exists from the B1 job:
#
#   # Confirm B1 checkpoint exists first:
#   ls -lh runs/*/checkpoint_best.pt
#
#   # Then submit:
#   qsub hpc/train_aaa_sv_job.sh
#
# Or supply the B1 checkpoint path explicitly:
#   qsub -v B1_CKPT=/path/to/checkpoint_best.pt hpc/train_aaa_sv_job.sh
# =============================================================

#PBS -N aaa_sv_train
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o logs/train_aaa_sv.out
#PBS -e logs/train_aaa_sv.err

# ── Environment ──────────────────────────────────────────────
cd "$PBS_O_WORKDIR"
mkdir -p logs runs

module load cuda/11.6 2>/dev/null || module load cuda/11.8 2>/dev/null || true

eval "$(~/micromamba/bin/micromamba shell hook --shell bash 2>/dev/null || micromamba shell hook --shell bash)"
micromamba activate speaker_sv

# ── Data paths ───────────────────────────────────────────────
# Set DATA_ROOT to the VoxCeleb1 dev wav root on the cluster.
export DATA_ROOT=/home/n_harini/datasets/voxceleb1_sv/vox1_dev_wav
export MUSAN_ROOT=/home/n_harini/datasets/musan         # update if present
export RIRS_ROOT=/home/n_harini/datasets/RIRS_NOISES    # update if present
export MODEL_CACHE=/home/n_harini/.cache/huggingface
export PROJECT_ROOT="$PBS_O_WORKDIR"
export HF_DATASETS_OFFLINE=1

# ── Locate the most recent B1 checkpoint_best.pt ─────────────
# Override by passing: qsub -v B1_CKPT=/full/path
if [ -z "$B1_CKPT" ]; then
    B1_CKPT=$(ls -t runs/*/checkpoint_best.pt 2>/dev/null | head -1)
fi

if [ -z "$B1_CKPT" ]; then
    echo "[ERROR] No B1 checkpoint found. Run the B1 job first or set B1_CKPT." >&2
    exit 1
fi
echo "[INFO] Using B1 checkpoint: $B1_CKPT"

# ── Launch AAA-SV training ───────────────────────────────────
python -m aaa_engine.training.train_aaa_sv \
    --config configs/train_aaa_sv.yaml \
    --b1-checkpoint "$B1_CKPT"
