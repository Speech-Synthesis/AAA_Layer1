#!/bin/bash
#PBS -N aaa_ecapa_train
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o logs/job.out
#PBS -e logs/job.err
#PBS -m abe
#PBS -M your.email@university.edu

# ============================================================
# PBS DIRECTIVES EXPLAINED:
# -N : Job name (appears in qstat)
# -q : Queue name (gpu, cpu, short, long - check with: qstat -Q)
# -l select=1:ncpus=8:ngpus=1:mem=32gb
#      select=1    : 1 node
#      ncpus=8     : 8 CPU cores
#      ngpus=1     : 1 GPU
#      mem=32gb    : 32GB RAM
# -l walltime=HH:MM:SS : Maximum runtime
# -o : stdout log file
# -e : stderr log file
# -m abe : Email on (a)bort, (b)egin, (e)nd
# -M : Email address
# ============================================================

# Print job info
echo "============================================================"
echo "  AAA Engine - ECAPA-TDNN Training Job"
echo "============================================================"
echo "Job Started:         $(date)"
echo "Job ID:              $PBS_JOBID"
echo "Node:                $(hostname)"
echo "Working Directory:   $PBS_O_WORKDIR"
echo "============================================================"

# Change to project directory
cd $PBS_O_WORKDIR

# Create logs directory if it doesn't exist
mkdir -p logs checkpoints

# Load required modules (check available modules with: module avail)
module load cuda/11.6
# module load cudnn/8.4  # if needed

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate myproject

# Print environment info
echo ""
echo "Environment Information"
echo "-----------------------"
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "import speechbrain; print(f'SpeechBrain: {speechbrain.__version__}')"
echo ""

# Show GPU status
nvidia-smi
echo ""

# ============================================================
# RUN TRAINING
# ============================================================
echo "Starting ECAPA-TDNN Training..."
echo "============================================================"

cd aaa_engine

python -m training.train_ecapa training/train_hparams.yaml \
    --data_folder=../data \
    --number_of_epochs=20 \
    --batch_size=32

# ============================================================
# RUN EVALUATION
# ============================================================
echo ""
echo "============================================================"
echo "Starting Model Evaluation (EER)..."
echo "============================================================"

python -m training.evaluate

# ============================================================
# EXPORT MODEL TO PRODUCTION
# ============================================================
echo ""
echo "============================================================"
echo "Exporting Model to Production..."
echo "============================================================"

python -m training.export_model

# ============================================================
# CLEANUP & EXIT
# ============================================================
echo ""
echo "============================================================"
echo "Job Finished: $(date)"
echo "============================================================"
