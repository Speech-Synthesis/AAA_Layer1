# HPC Deep Learning Project Setup Guide

> [!NOTE]
> This guide covers end-to-end setup for running deep learning training jobs on an HPC cluster using **PBS** (Portable Batch System) and **Micromamba**.

---

## 1. Initial Environment Setup (Run Once)

### Step 1 — Install Micromamba

```bash
cd ~
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mkdir -p ~/.local/bin
mv bin/micromamba ~/.local/bin/
rmdir bin
```

Add to `~/.bashrc`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(micromamba shell hook --shell bash)"' >> ~/.bashrc
source ~/.bashrc
```

---

### Step 2 — Create Project Environment

> [!TIP]
> Replace `myproject` with your actual project name. Replace Python version and packages as needed.

```bash
micromamba create -n myproject python=3.10 -c conda-forge -y
micromamba activate myproject
```

#### Install PyTorch

> [!IMPORTANT]
> Check your HPC's CUDA version first with `nvidia-smi` — look for **"CUDA Version: XX.X"**.

| CUDA Version | Install Command |
|:---:|---|
| **11.6** | `pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116` |
| **11.8** | `pip install torch==2.0.0+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118` |
| **12.1** | `pip install torch torchvision torchaudio` |

```bash
# Install project dependencies
pip install -r requirements.txt
```

---

### Step 3 — Clone Your Project

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/YOUR_PROJECT.git
cd YOUR_PROJECT
```

---

## 2. PBS Job Script Template

Create `train_job.sh`:

```bash
#!/bin/bash
#PBS -N my_training_job
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=32gb
#PBS -l walltime=24:00:00
#PBS -o logs/job.out
#PBS -e logs/job.err
#PBS -m abe
#PBS -M your.email@university.edu
```

### PBS Directives Explained

| Directive | Purpose |
|---|---|
| `-N` | Job name (appears in `qstat`) |
| `-q` | Queue name (`gpu`, `cpu`, `short`, `long` — check with `qstat -Q`) |
| `-l select=1:ncpus=8:ngpus=1:mem=32gb` | **1 node**, **8 CPU cores**, **1 GPU**, **32 GB RAM** |
| `-l walltime=HH:MM:SS` | Maximum runtime |
| `-o` | Stdout log file |
| `-e` | Stderr log file |
| `-m abe` | Email on **(a)**bort, **(b)**egin, **(e)**nd |
| `-M` | Email address |

### Job Body

```bash
# Print job info
echo "============================================================"
echo "Job Started: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Node: $(hostname)"
echo "Working Directory: $PBS_O_WORKDIR"
echo "============================================================"

# Change to project directory
cd $PBS_O_WORKDIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (check available with: module avail)
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
echo ""

# Show GPU status
nvidia-smi
echo ""

# ============================================================
# RUN YOUR TRAINING SCRIPT
# ============================================================
echo "Starting Training..."
echo "============================================================"

python train.py \
    --config configs/config.json \
    --checkpoint_dir checkpoints \
    --log_dir logs

# ============================================================
# CLEANUP & EXIT
# ============================================================
echo ""
echo "============================================================"
echo "Job Finished: $(date)"
echo "============================================================"
```

---

## 3. Directory Structure

```
~/your_project/
├── train.py                 # Your training script
├── model.py                 # Your model
├── dataset.py               # Your dataset
├── configs/
│   └── config.json          # Hyperparameters
├── checkpoints/             # Saved models
├── logs/                    # Job logs
├── data/                    # Your data
├── train_job.sh             # PBS job script
└── requirements.txt         # Python dependencies
```

```bash
# Create required directories
mkdir -p checkpoints logs data
```

---

## 4. Essential PBS Commands

| Action | Command |
|---|---|
| **Submit a job** | `qsub train_job.sh` |
| **All jobs** | `qstat` |
| **Your jobs only** | `qstat -u $USER` |
| **Detailed job info** | `qstat -f JOB_ID` |
| **Live stdout** | `tail -f logs/job.out` |
| **Live stderr** | `tail -f logs/job.err` |
| **Cancel a job** | `qdel JOB_ID` |
| **Available queues** | `qstat -Q` |
| **Queue details** | `qstat -Qf gpu` |
| **Node status** | `pbsnodes -a` |
| **Interactive GPU session** | `qsub -I -q gpu -l select=1:ncpus=4:ngpus=1:mem=16gb -l walltime=2:00:00` |

---

## 5. Common Configurations

### A. Multi-GPU Training (Single Node)

```bash
#PBS -l select=1:ncpus=16:ngpus=4:mem=64gb
```

```python
# In your Python script:
model = torch.nn.DataParallel(model)
```

### B. Large Memory Jobs

```bash
#PBS -l select=1:ncpus=8:ngpus=1:mem=128gb
```

### C. Long Running Jobs

```bash
#PBS -l walltime=72:00:00
#PBS -r y  # Make job rerunnable if node fails
```

### D. Array Jobs (Multiple Experiments)

```bash
#PBS -J 1-5  # Run 5 jobs with PBS_ARRAY_INDEX = 1,2,3,4,5
```

```bash
# In script:
CONFIG_ID=$PBS_ARRAY_INDEX
python train.py --config configs/exp_${CONFIG_ID}.json
```

---

## 6. Training Script Best Practices

Add these patterns to your `train.py`:

### 1 — Automatic GPU Selection

```python
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 2 — Reproducibility

```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
```

### 3 — Checkpointing (Resume from Interruption)

```python
def save_checkpoint(model, optimizer, epoch, step, path):
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Saved checkpoint: {path}")

def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resumed from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        return checkpoint['epoch'], checkpoint['step']
    return 0, 0
```

### 4 — Memory Management

```python
# Clear cache periodically
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
# model.gradient_checkpointing_enable()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 5 — Graceful Interruption Handling

> [!WARNING]
> PBS sends `SIGTERM` before killing a job. Catching this signal lets you save a checkpoint before the process exits.

```python
import signal
import sys

def signal_handler(signum, frame):
    print("\nInterrupted! Saving checkpoint...")
    save_checkpoint(model, optimizer, epoch, step, "checkpoints/interrupted.pt")
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)  # PBS sends SIGTERM
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
```

---

## 7. Troubleshooting

| Problem | Solution |
|---|---|
| **Job stuck in queue?** | `qstat -f JOB_ID \| grep -E "job_state\|comment"` |
| **Out of memory?** | Reduce batch size or use gradient accumulation |
| **CUDA out of memory?** | `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` |
| **Module not found?** | `module avail cuda` → `module load cuda/11.6` |
| **Permission denied on logs?** | `chmod +x train_job.sh` and `mkdir -p logs` |
| **Check disk quota** | `quota -s` / `du -sh ~/*` |

---

## 8. Quick Start Checklist

```bash
# 1. Setup (first time only)
cd ~
git clone https://github.com/YOUR_USER/YOUR_PROJECT.git
cd YOUR_PROJECT

# 2. Create directories
mkdir -p logs checkpoints

# 3. Make job script executable
chmod +x train_job.sh

# 4. Edit job script with your settings
nano train_job.sh

# 5. Submit job
qsub train_job.sh

# 6. Monitor
watch -n 5 qstat -u $USER    # Job status every 5s
tail -f logs/job.err          # Live error log
```

> [!TIP]
> Bookmark this guide for quick reference when submitting jobs on the cluster.
