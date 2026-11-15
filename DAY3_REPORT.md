# AAA-SV — Day 3 Individual Report
**Member:** Navadeep Maka  
**Role:** AAA-SV Speaker Verification Model Engineer  
**Workstream:** AAA-SV (Model Checkpoints, Full Integration, Model Card, Delivery Package)  
**Sprint Day:** Day 3 — Friday 13 Nov 2026  
**Branch:** `feat/navadeep-aaa-sv`  
**Report Version:** 1.0  

---

## 1. Executive Summary & Goals

Day 3 focused on:
1. **Cluster GPU Training Milestone:** Verifying completion of B1 fine-tuning on the Amrita HPC cluster, capturing official checkpoint hashes, and auditing loss curves.
2. **Model Package & Checkpoint Staging:** Assembling the final model artifact directory structure (`model.pt`, `config.yaml`, `preproc.yaml`, `calibration.json`, `MODEL_CARD.md`, `SHA256SUMS`).
3. **Full Regression Testing:** Running the unified test suite across all subsystems (113/113 passed).
4. **Handoff & Peer Review Preparation:** Packaging evidence and instructions for independent review by Karthik and API consumption by Surya.

---

## 2. HPC Training Audit & Results

### 2.1 B1 Fine-Tuning Run (`93483.amritahpc`)
- **Execution Target:** Tesla V100-PCIE-16GB, 4 CPUs, 32 GB host memory.
- **Dataset:** VoxCeleb1 dev partition (1,211 speakers, 112,470 train utterances).
- **Manifest Hash:** `1a3c010b39da65d63701dfa3ef8307d64690f09cf6e3b256eb5b4b968942bc3f`
- **Leakage Check:** 0 overlapping speakers between training data and locked evaluation IDs.
- **Epoch 1 Training Dynamics:**
  - Steps completed: 3,514 global steps.
  - Initial Loss: 1.5819 (Step 1)
  - Convergence trajectory: Loss decreased steadily across batches: $1.3230 \to 1.0699 \to 0.9923 \to 0.9286$.
  - Learning rates: Backbone $1.0 \times 10^{-4}$ (frozen epoch 1 warmup), AAM head $1.0 \times 10^{-3}$.
  - Margin ramp: $m$ ramped smoothly from $0.00 \to 0.05$ rad over the warmup schedule.

### 2.2 Checkpoint Verification
The job completed Epoch 1 and wrote atomic checkpoints to cluster storage:
- `checkpoint_best.pt`: 83 MB, saved at `runs/20261113-1718_09dc907_5e10bf/checkpoint_best.pt`
- `checkpoint_last.pt`: 83 MB, saved at `runs/20261113-1718_09dc907_5e10bf/checkpoint_last.pt`
- Run artifacts captured: `git_diff.txt`, `gpu_info.txt`, `manifest_hash.txt`, `pip_freeze.txt`, `run_config.yaml`, `seed.txt`, `metrics.jsonl`, `metrics.csv`.

---

## 3. End-to-End System Validation

The full test suite was executed locally on `feat/navadeep-aaa-sv`:

```bash
python -m pytest tests/ -v
```

### Complete Test Results:
```
tests/test_layer1.py (34 tests) .................................. [ 30%]
tests/test_stability_head.py (10 tests) ..........                [ 39%]
tests/test_calibrator.py (27 tests) ...........................   [ 63%]
tests/test_eval_toolkit.py (18 tests) ..................          [ 79%]
tests/test_aaasv.py (19 tests) ...................                [ 95%]
tests/test_sv_pipeline.py (5 tests) .....                         [100%]

======================= 113 passed, 1 warning in 8.72s ========================
```

---

## 4. Final Deliverables Summary

1. **`MODEL_CARD.md`**: Complete architectural and performance specifications, intended usage, and biometric privacy disclosures.
2. **`DATASET_PLAN.md`**: Data lineage, partition protocols, and speaker isolation guarantees.
3. **`INDIVIDUAL_REPORT_NAVADEEP_FINAL.md`**: Consolidated 4-day technical mission summary.
4. **Developer Test Reports**: Day 1, Day 2, and Day 3 test reports logging all 113 unit tests.
5. **Inference & Calibration Codebase**: Ready for immediate import in `AAA_Baselines` or deployment as Layer 1 microservice.
