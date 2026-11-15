#!/bin/bash
#PBS -N aaa_sv_eval_val
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=04:00:00
#PBS -q workq
#PBS -j oe
#PBS -o logs/eval_val_${PBS_JOBID}.log

set -euo pipefail
cd "$PBS_O_WORKDIR"
source ~/.bashrc
micromamba activate speaker_sv

echo "=== B1 Val Evaluation Job ==="
echo "Job ID: $PBS_JOBID"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Node: $(hostname)"
echo "Checkpoint SHA256:"
sha256sum runs/20261113-1718_09dc907_5e10bf/checkpoint_best.pt

git pull origin feat/navadeep-aaa-sv

mkdir -p docs/evidence/day4

python -m aaa_engine.evaluation.eval_toolkit \
  --manifest data/manifests/val.csv \
  --data-root /home/n_harini/datasets/voxceleb1_sv/vox1_dev_wav \
  --model-ckpt runs/20261113-1718_09dc907_5e10bf/checkpoint_best.pt \
  --output docs/evidence/day4/val_eval_result.json \
  --model-label B1

echo "=== Result ==="
cat docs/evidence/day4/val_eval_result.json | python -c "
import sys, json
d = json.load(sys.stdin)
print('EER:        ', d.get('eer'))
print('FRR@FAR1%:  ', d.get('frr_at_far1pct'))
print('minDCF:     ', d.get('min_dcf'))
print('run_id:     ', d.get('run_id'))
print('trial_sha:  ', d.get('trial_list_sha256','')[:16])
"

sha256sum docs/evidence/day4/val_eval_result.json > docs/evidence/day4/val_eval_result.sha256

echo "End: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
