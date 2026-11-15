# AAA-SV — Day 2 Individual Report
**Member:** Navadeep Maka  
**Workstream:** AAA-SV (Speaker Verification — Model, Training, Calibration, Evaluation)  
**Sprint Day:** Day 2 — Thursday 12 Nov 2026  
**Branch:** `feat/navadeep-aaa-sv`  
**Report Version:** 1.0  

---

## 1. Executive Summary & Goals

Day 2 objectives centered on implementing the core machine learning innovations specified in `docs/mission_pack/02_MODEL_ARCHITECTURE/AAA-SV.md` and `SHARED.md`:
1. **Auxiliary Stability Head $q(x)$ and Joint Loss:** Design a detached pooling head that predicts speaker embedding stability under degradation without destabilizing the ECAPA-TDNN backbone.
2. **Comprehensive Evaluation Toolkit:** Implement multi-condition bucketed trial generation and NIST-compliant verification metrics (EER, minDCF, $C_{\text{llr}}$, DET curves).
3. **Adaptive Score Normalization & Calibration:** Engineer Top-$K$ AS-Norm ($K=300$) and an empirical log-likelihood ratio logistic calibrator.
4. **Production Inference Adapter:** Wrap the verifier in a strict `SHARED.md §2` error-handling envelope with adaptive voiceprint gating.

All four modules were developed, unit-tested, and fully validated locally.

---

## 2. Work Completed

### 2.1 Stability Head $q(x)$ (`aaa_engine/training/stability_head.py`) [CUSTOM]
- **Architecture:** 
  $$\text{Linear}(6144, 256) \to \text{ReLU} \to \text{Linear}(256, 1) \to \text{Sigmoid}$$
  Takes the 6,144-dimensional Attentive Statistics Pooling (ASP) output ($3,072 \text{ mean} + 3,072 \text{ std}$) and outputs a scalar $q(x) \in [0, 1]$.
- **Gradient Isolation:** Both the calling harness and the module explicitly enforce `pooled_stats.detach()`. Backpropagating $\mathcal{L}_q$ yields zero gradient into the backbone.
- **Stop-Gradient Target:** 
  $$y_q = \text{clamp}(\cos(e(x_{\text{deg}}), \text{stop\_grad}(e(x_{\text{ref}}))), 0, 1)$$
  Calculated between the augmented short training view and a clean $\le 6.0$s reference view.
- **Joint Loss Function:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{AAM}} + 0.1 \cdot \mathcal{L}_q$.
- **Training Harness (`train_aaa_sv.py`):** Extended the training pipeline with `DualViewDataset` to generate synchronized (degraded, clean reference, speaker ID) triples.

### 2.2 Evaluation Toolkit (`aaa_engine/evaluation/eval_toolkit.py`) [CUSTOM]
- **Bucketed Trial Generation:** `BucketedTrialGenerator` automatically partitions trials across:
  - Duration: $1$s, $2$s, $3$s, $5$s.
  - Signal-to-Noise Ratio (SNR): $0$, $5$, $10$, $20$ dB.
  - Codecs: MP3, AAC, Opus, AMR, and 8 kHz upsampled telephone speech.
- **Metrics Computed:**
  - Equal Error Rate (EER) via dense threshold search.
  - False Rejection Rate at 1% False Acceptance Rate ($\text{FRR@FAR1\%}$).
  - minDCF ($p_{\text{target}}=0.05, C_{\text{fa}}=1, C_{\text{miss}}=10$) per NIST SRE standard.
  - Log-likelihood ratio cost ($C_{\text{llr}}$).
  - Detection Error Tradeoff (DET) curve coordinates.
  - Non-parametric bootstrap confidence intervals (1,000 iterations, 95% CI).
- **Safety Freeze Guard:** Enforces `RuntimeError` if an evaluation script targets a manifest containing `"test"` without the explicit `--freeze` flag.

### 2.3 Score Calibration & Normalization (`aaa_engine/calibration/calibrator.py`) [CUSTOM]
- **Top-$K$ Adaptive S-Norm:** Normalizes raw cosine similarities against a fixed cohort of training speakers:
  $$s_{\text{norm}} = \frac{1}{2} \left( \frac{s - \mu_E}{\sigma_E} + \frac{s - \mu_T}{\sigma_T} \right)$$
  using $K=300$ nearest cohort embeddings.
- **Multi-Condition Logistic Calibrator:** Fits log-odds mapping using L-BFGS-B optimization to minimize $C_{\text{llr}}$ across features:
  $$[s_{\text{norm}}, q, \log(\text{duration}), \text{snr}/20, \text{channel\_class}]$$
- **Calibration Manifest Guard:** Prevents contamination by rejecting manifests containing `"train"`, `"val"`, or `"test"`.

### 2.4 Inference Wrapper (`aaa_engine/inference/aaasv.py`) [CUSTOM]
- **Shared Error Envelope:** All public methods (`embed`, `enroll`, `verify`, `maybe_update`) return an `AAASVResponse` envelope with explicit error codes (`E_FORMAT`, `E_DECODE`, `E_TOO_SHORT`, `E_TOO_LONG`, `E_SILENT`, `E_NONFINITE`).
- **Duration Handling:** Clips $< 1.0$s are rejected (`E_TOO_SHORT`); clips between $1.0$s and $2.0$s pass with a `W_SHORT` warning and `"low"` confidence. No repeat-padding is applied.
- **Adaptive Voiceprint Updates:** The `maybe_update` function gates model adaptation strictly on match score ($s > 0.85$) and quality ($q > 0.70$), returning updated voiceprint objects without directly touching the persistence layer.

---

## 3. Developer Verification & Test Results

All new functionality was accompanied by isolated unit and integration test suites:

```bash
python -m pytest tests/test_stability_head.py tests/test_eval_toolkit.py tests/test_calibrator.py tests/test_aaasv.py -v
```

**Results:**
- `test_stability_head.py`: 10 passed
- `test_eval_toolkit.py`: 18 passed
- `test_calibrator.py`: 27 passed
- `test_aaasv.py`: 19 passed
- **Day 2 Test Subtotal: 74 passed, 0 failed in 6.2s.**

---

## 4. Repository State & Commits

- `c235d33` feat(phase3): stability head q(x), joint loss, AAA-SV training script, 10 tests (49/49 pass)
- `5b04c79` feat(phase3): evaluation toolkit (3.B), calibration & s-norm (3.C), aaasv inference wrapper (3.D) [113/113 tests pass]
- `5e40684` fix(validate): indent row processing inside csv.DictReader loop in validate()
- `cda5065` fix(hpc): pin aaa_sv training to free node01 and use batch_size 16 with grad_accum 2

---

## 5. Next Steps for Day 3
1. Monitor HPC V100 GPU fine-tuning to completion.
2. Harvest and document official model checkpoint hashes (`checkpoint_best.pt`).
3. Package final model card, calibration files, and developer test evidence for peer review by Karthik.
