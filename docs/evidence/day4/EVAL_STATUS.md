# AAA-SV Evaluation Status Report — Day 4

**Date:** Saturday 15 November 2026  
**Engineer:** Navadeep Maka (Lead Model Engineer)  
**Run ID:** `20261113-1718_09dc907_5e10bf`  
**Model Branch:** `feat/navadeep-aaa-sv`  
**Checkpoint Tested:** `checkpoint_best.pt` (SHA256: `f78e2b6c1c20bdee8930dd5e04c47837e962c7aae8cee74866c71c212b03cdbd`)

---

## 1. Evaluation Attempt Summary

An offline evaluation run was attempted on the Amrita HPC cluster using the VoxCeleb1 validation split:
- **Utterances:** 2,422 validation utterances from 1,211 speakers
- **Trial Pairs:** 733,866 trial pairs (exhaustive cross-speaker and same-speaker evaluation)
- **Manifest File:** `data/manifests/val.csv` (SHA256 starts with `9739f2a8...`)
- **Evaluation Tool:** `aaa_engine/evaluation/eval_toolkit.py`

---

## 2. Bug History & Root Cause Analysis

During execution on the HPC cluster, evaluation encountered several runtime issues in `eval_toolkit.py`:

1. **Commit `f7ecead` — Missing SpeechBrain Fbank Feature Extraction:**
   - *Issue:* The initial evaluation script called a feature extraction routine that assumed raw waveform inputs without proper mel-spectrogram transformation.
   - *Fix:* Added `speechbrain.processing.features.Fbank` transformation.

2. **Commit `cab13f4` — Feature Dimension Transpose Mismatch:**
   - *Issue:* SpeechBrain ECAPA-TDNN expects feature inputs formatted as `(batch, time, freq)`, whereas the extracted tensor had dimension layout `(batch, freq, time)`.
   - *Fix:* Added transposition to match ECAPA-TDNN input expectations.

3. **Commit `460b2f0` — Migration to `torchaudio.transforms.MelSpectrogram`:**
   - *Issue:* Inconsistent SpeechBrain Fbank API behaviour across environment versions on the HPC node caused unexpected parameter exceptions.
   - *Fix:* Replaced SpeechBrain Fbank with standard `torchaudio.transforms.MelSpectrogram` (80 mel bins, 25ms window, 10ms hop), ensuring robust, repeatable feature extraction matching `train_b1.py`.

4. **New Checkpoint Key Mismatch Identified:**
   - *Issue:* `train_b1.py` saves the checkpoint as a dictionary with key `"model"` containing the model `state_dict` (along with `"epoch"`, `"optimizer"`, `"margin"`, etc.), rather than a raw model object or standard `"state_dict"` key. The eval loader expected direct weights or `"state_dict"`.
   - *Resolution:* Fixed checkpoint unpacking logic in `eval_toolkit.py` to inspect and load `state_dict` from `"model"`, `"state_dict"`, or raw dictionary formats.

---

## 3. Current HPC Status

- **Evaluation Completion:** Eval run has not yet completed on the HPC cluster; fix being applied in the current commit sequence.
- **Active Job 93490.amritahpc:** `aaa_sv_train` (AAA-SV joint training with Stability Head $q(x)$) has been running for ~12h on Tesla V100 GPU.
- **Completed Job 93483.amritahpc:** B1 training job completed (1 epoch warmup, loss $1.58 \to 0.92$).
- **Kaggle Fallback Script:** A standalone evaluation notebook script `hpc/kaggle_eval_b1.py` has been prepared in the repository for offloading evaluation if HPC queue remains occupied.

---

## 4. Quantitative Metrics Declaration

> [!CAUTION]
> **EER, FRR@FAR=1%, and minDCF: NOT YET MEASURED.**  
> In accordance with sprint integrity rules, no synthetic, estimated, or placeholder numbers are presented.

| Metric | Target | Status |
|---|---|---|
| **EER (Equal Error Rate)** | $\le 2.0\%$ | **NOT YET MEASURED** |
| **FRR @ FAR = 1%** | $\le 3.0\%$ | **NOT YET MEASURED** |
| **minDCF ($P_{\text{target}}=0.05$)** | $\le 0.20$ | **NOT YET MEASURED** |

Numerical verification will be published immediately upon completion of the pending evaluation job.
