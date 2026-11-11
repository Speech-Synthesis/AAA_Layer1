#!/usr/bin/env bash
# =============================================================
# train_job_b1.sh
# AAA Engine – B1 Fine-Tune – HPC PBS Job Script
# =============================================================
# Submit with:
#   qsub train_job_b1.sh
#
# FILL_IN markers indicate values you must supply.
# Never hard-code personal paths, passwords, or account names.
#
# Required environment variables (set in your ~/.bashrc or via
# 'export' before qsub, or in the PBS -v list below):
#
#   PROJECT_ROOT  — absolute path to this repository on the cluster
#                   e.g. /scratch/username/AAA_Layer1
#   DATA_ROOT     — absolute path to VoxCeleb1 dev wav root
#   MUSAN_ROOT    — absolute path to MUSAN corpus root
#   RIRS_ROOT     — absolute path to RIRS_NOISES corpus root
#   MODEL_CACHE   — local directory for pretrained model weights
#                   e.g. /scratch/username/model_cache
# =============================================================

# ── PBS directives ────────────────────────────────────────────

# FILL_IN: Your HPC project / account code
#PBS -A FILL_IN_ACCOUNT_CODE

# FILL_IN: Queue / partition name (check with: qstat -Q)
#PBS -q FILL_IN_QUEUE_NAME

# FILL_IN: Walltime limit (HH:MM:SS).  VoxCeleb1 fine-tune
#          typically takes 8–24 h depending on GPU and epochs.
#PBS -l walltime=24:00:00

# FILL_IN: Resource request.  Adjust ncpus, ngpus, mem to match
#          your cluster's available node types.
#          Example for a DGX-A100 node:
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb

# Job name visible in qstat
#PBS -N aaa_b1_finetune

# FILL_IN: Email address for job state notifications (optional)
#PBS -M FILL_IN_YOUR_EMAIL@example.com
#PBS -m abe

# Join stdout and stderr into separate named files (see below)
#PBS -j n

# =============================================================
# From here on this is regular bash.
# =============================================================
set -euo pipefail

# ── Validate required environment variables ───────────────────
: "${PROJECT_ROOT:?PROJECT_ROOT must be set (e.g. /scratch/user/AAA_Layer1)}"
: "${DATA_ROOT:?DATA_ROOT must be set (e.g. /scratch/user/voxceleb1/dev/wav)}"
: "${MUSAN_ROOT:?MUSAN_ROOT must be set (e.g. /scratch/user/musan)}"
: "${RIRS_ROOT:?RIRS_ROOT must be set (e.g. /scratch/user/RIRS_NOISES)}"
: "${MODEL_CACHE:?MODEL_CACHE must be set (e.g. /scratch/user/model_cache)}"

# ── Redirect stdout / stderr to per-job log files ────────────
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
exec 1>"${LOG_DIR}/job_${PBS_JOBID}.out" 2>"${LOG_DIR}/job_${PBS_JOBID}.err"

echo "============================================================"
echo "  AAA Engine B1 Fine-Tune Job"
echo "  Started:      $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "  PBS Job ID:   ${PBS_JOBID}"
echo "  Node list:    $(cat ${PBS_NODEFILE} 2>/dev/null || echo 'N/A')"
echo "  PROJECT_ROOT: ${PROJECT_ROOT}"
echo "  DATA_ROOT:    ${DATA_ROOT}"
echo "============================================================"

# ── Load system modules ───────────────────────────────────────
# FILL_IN: Uncomment and adjust to the CUDA version on your cluster.
# Check available versions with:  module avail cuda
# module load cuda/12.1   # FILL_IN: check available with: module avail cuda

# FILL_IN: Uncomment if cuDNN is a separate module on your cluster.
# module load cudnn/8.9    # FILL_IN: check available with: module avail cudnn

# FILL_IN: Uncomment if ffmpeg is a module on your cluster.
# module load ffmpeg/6.0   # FILL_IN: for codec augmentation

# ── Activate Python environment ───────────────────────────────
# FILL_IN: Replace 'aaa_env' with your conda/micromamba env name.
# conda activate aaa_env   # FILL_IN: your env name
# OR for micromamba:
# micromamba activate aaa_env  # FILL_IN: your env name

# ── Change to project directory ───────────────────────────────
cd "${PROJECT_ROOT}"

# ── Create checkpoint output directory ───────────────────────
# Checkpoints are written to runs/ inside PROJECT_ROOT,
# which is outside Git's tracking directory by design.
mkdir -p "${PROJECT_ROOT}/runs"
mkdir -p "${PROJECT_ROOT}/logs"

# ── Automatic resume ──────────────────────────────────────────
# The training script detects checkpoint_last.pt automatically
# by run_id (derived from config hash + git SHA).  No explicit
# --resume flag is needed; the script checks on startup.
#
# If you want to force a fresh run despite an existing checkpoint,
# delete or rename the relevant runs/<run_id>/ directory.

echo "[job] Checking for existing checkpoints …"
# Count existing run directories as a quick summary
N_RUNS=$(find "${PROJECT_ROOT}/runs" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l)
echo "[job] Existing run directories: ${N_RUNS}"

# ── Run training ──────────────────────────────────────────────
echo "[job] Launching train_b1.py …"

python -m aaa_engine.training.train_b1 \
    --config configs/train_hpc.yaml

EXIT_CODE=$?

echo "============================================================"
if [ "${EXIT_CODE}" -eq 0 ]; then
    echo "  Training COMPLETED successfully."
else
    echo "  Training FAILED with exit code ${EXIT_CODE}."
fi
echo "  Finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

exit "${EXIT_CODE}"
