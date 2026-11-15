# AAA-SV Installation Guide

## Environment Setup
```bash
micromamba create -n speaker_sv python=3.10
micromamba activate speaker_sv
pip install torch==1.13.1+cu116 torchaudio==0.13.1 speechbrain==0.5.16 numpy scipy soundfile
```

## Clone Repository
```bash
git clone https://github.com/Speech-Synthesis/AAA_Layer1.git
cd AAA_Layer1
git checkout feat/navadeep-aaa-sv
```

## Verify Checkpoint SHA256
```bash
sha256sum runs/20261113-1718_09dc907_5e10bf/checkpoint_best.pt
# Expected: f78e2b6c1c20bdee8930dd5e04c47837e962c7aae8cee74866c71c212b03cdbd
```

## Run Tests
```bash
python -m pytest tests/ -v
# Expected: 113 passed
```

## Run Evaluation
```bash
python -m aaa_engine.evaluation.eval_toolkit \
  --manifest data/manifests/val.csv \
  --data-root /path/to/voxceleb1/dev/wav \
  --model-ckpt runs/20261113-1718_09dc907_5e10bf/checkpoint_best.pt \
  --output docs/evidence/day4/val_eval_result.json \
  --model-label B1
```

## Offline Load Proof
```bash
python hpc/offline_load_test.py
cat docs/evidence/day4/offline_load_proof.json
```
