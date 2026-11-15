# AAA-SV — Day 4 Individual Report

**Member:** Navadeep Maka  
**Role:** AAA-SV Speaker Verification Model Engineer  
**Workstream:** AAA-SV (Model Checkpoints, Evaluation Toolkit, Final Deliverables, Handoff)  
**Sprint Day:** Day 4 — Saturday 15 Nov 2026  
**Branch:** `feat/navadeep-aaa-sv`  
**Report Version:** 1.0  

---

## 1. Daily Goals

1. **Complete Evaluation Pipeline:** Resolve runtime issues in `eval_toolkit.py` and execute validation split evaluation on the HPC cluster.
2. **Fix `eval_toolkit` Bugs:** Address feature extraction incompatibilities and checkpoint state dict unpacking.
3. **Compile All Final Deliverables:** Author complete model card, integration acceptance specs, offline load proof, and peer review packages.
4. **Coordinate Handoff & Review:** Deliver frozen checklist to Karthik for peer review and provide adapter specifications to Surya.

---

## 2. Work Completed & Accomplishments

1. **Evaluation Toolkit Debugging & Fixes (3 Iterations):**
   - **Commit `f7ecead`:** Resolved missing mel filterbank transformation by introducing `speechbrain.processing.features.Fbank`.
   - **Commit `cab13f4`:** Corrected feature dimension ordering transpose `(batch, time, freq)` expected by the ECAPA-TDNN backbone.
   - **Commit `460b2f0`:** Migrated feature extraction to standard `torchaudio.transforms.MelSpectrogram` (80 mel bins, 25ms window, 10ms hop) to ensure robust environment consistency between training and evaluation pipelines.
   - **Checkpoint Unpacking Resolution:** Addressed `train_b1.py` checkpoint structure mismatch where weights were nested under key `"model"`.

2. **Checkpoint SHA256 Integrity Verification:**
   - Captured and verified official hashes for checkpoints under `runs/20261113-1718_09dc907_5e10bf/`:
     - `checkpoint_best.pt`: `f78e2b6c1c20bdee8930dd5e04c47837e962c7aae8cee74866c71c212b03cdbd`
     - `checkpoint_last.pt`: `260db647fc93f6fbade2c4b11937a255c681974358ba01a7087f37d66e10e67a`
   - Checksums permanently recorded in `RELEASE_ARTIFACTS/SHA256SUMS.txt` and `docs/evidence/day1/checkpoint_sha256.txt`.

3. **Active Cluster Jobs:**
   - **Job `93483.amritahpc`:** B1 training job completed successfully (1 epoch warmup, loss $1.58 \to 0.92$).
   - **Job `93490.amritahpc`:** `aaa_sv_train` (joint training with Stability Head $q(x)$) active and running (~12h runtime on Tesla V100 GPU).

4. **Offline Load Proof:**
   - Authored and executed `hpc/offline_load_test.py`.
   - Verified that `AAASV` wrapper initializes with local staged files and runs inference with zero network connectivity.
   - Result captured in `docs/evidence/day4/offline_load_proof.json` (`outcome: PASS`).

5. **Deliverables Documentation Authored:**
   - `MODEL_CARD.md`: Overwritten with strict provenance, honest performance declarations, and error code taxonomy.
   - `docs/evidence/day4/EVAL_STATUS.md`: Full evaluation history, bug fixes, and transparent status.
   - `docs/evidence/day4/OFFLINE_LOAD_PROOF.md`: Step-by-step offline load proof report.
   - `docs/evidence/day4/INTEGRATION_ACCEPTANCE.md`: Public contract specifications and error taxonomy for Surya.
   - `docs/evidence/day4/KARTHIK_PEER_REVIEW_PACKAGE.md`: Frozen 14-item verification checklist for Karthik.
   - `docs/evidence/day4/FINAL_STATUS.md`: Comprehensive deliverable status matrix.
   - `docs/evidence/day4/NEEDS_SURYA.md`: Clear requests for dataset deviation sign-off, Common Voice Set B, and Vox1-O test staging.
   - Release artifacts: `REGISTRY_MANIFEST.md`, `INSTALL_README.md`, `SHA256SUMS.txt`, and `calibration.json`.

6. **Unit Test Suite:**
   - All 113/113 local unit tests passing on Windows CPU (`pytest tests/ -v`).

---

## 3. Blockers Encountered & Resolutions

- **`eval_toolkit` Feature Extraction Bug:**
  - Required 3 iterations to fix across multiple commits (`f7ecead`, `cab13f4`, `460b2f0`). SpeechBrain Fbank had breaking differences between PyTorch 1.13 and 2.x; permanently solved by using PyTorch native `torchaudio.transforms.MelSpectrogram`.
- **Checkpoint Key Mismatch:**
  - `train_b1.py` checkpoint saved weights in a dictionary under `"model"` rather than directly or under `"state_dict"`. Updated evaluation loading logic to inspect both `"model"` and `"state_dict"`.
- **HPC Cluster Contention:**
  - GPU node heavily contested and running `aasist_dy` and `aaa_sv_train` jobs, delaying the rerun of the 733k trial pair evaluation job.
  - Staged Kaggle fallback script (`hpc/kaggle_eval_b1.py`) to decouple evaluation from HPC cluster queuing.

---

## 4. Honest Model Status

- **Classification:** Strictly labeled `[BASELINE] backbone (frozen ep1 warmup), ArcFace head trained` — **NOT [FINE-TUNED]**.
- **Epochs:** Exactly 1 warmup epoch completed. Full backbone fine-tuning (~24h compute) was not executed.
- **ArcFace Margin:** Margin ramp completed at $m=0.05$ out of $0.20$ rad target.
- **Quantitative Metrics:** EER, FRR@FAR=1%, and minDCF are **NOT YET MEASURED** pending the completion of the HPC evaluation job. No synthetic metrics have been invented.
