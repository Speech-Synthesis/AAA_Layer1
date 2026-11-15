#!/bin/bash
#PBS -N aaa_sv_resume_proof
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:30:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/resume_proof_${PBS_JOBID}.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
source ~/.bashrc
micromamba activate speaker_sv

echo "=== Kill-and-Resume Proof ===" | tee docs/evidence/day4/resume_proof.txt
echo "Job: $PBS_JOBID  Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a docs/evidence/day4/resume_proof.txt

# Use the local smoke config (tiny synthetic dataset, no GPU needed)
python -m aaa_engine.training.train_b1 \
  --config configs/train_local_smoke.yaml \
  --max-epochs 1 2>&1 | head -50 | tee -a docs/evidence/day4/resume_proof.txt

echo "--- Checkpoint written ---" | tee -a docs/evidence/day4/resume_proof.txt
ls -lh runs/*/checkpoint_last.pt 2>/dev/null | tail -1 | tee -a docs/evidence/day4/resume_proof.txt

echo "--- Resume test: test_resume_continuity() ---" | tee -a docs/evidence/day4/resume_proof.txt
python -m pytest tests/test_sv_pipeline.py::test_resume_continuity -v 2>&1 | tee -a docs/evidence/day4/resume_proof.txt

echo "RESUME PROOF END: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a docs/evidence/day4/resume_proof.txt
