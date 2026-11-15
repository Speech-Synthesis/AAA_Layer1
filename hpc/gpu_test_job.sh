#!/bin/bash
#PBS -N aaa_sv_gpu_tests
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1
#PBS -l walltime=01:00:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/gpu_test_${PBS_JOBID}.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
source ~/.bashrc
micromamba activate speaker_sv

echo "=== AAA-SV GPU Test Suite ==="
echo "Job: $PBS_JOBID  Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

git pull origin feat/navadeep-aaa-sv
mkdir -p docs/evidence/day4

python -m pytest tests/ -v 2>&1 | tee docs/evidence/day4/gpu_test_output.txt
EXIT_CODE=${PIPESTATUS[0]}
echo "EXIT_CODE: $EXIT_CODE" >> docs/evidence/day4/gpu_test_output.txt

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $EXIT_CODE
