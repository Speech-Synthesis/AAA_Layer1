# Model Card — AAA-SV (Adaptive Acoustic Authentication Speaker Verifier)

> [!IMPORTANT]
> **Model Label: [BASELINE] backbone (frozen ep1 warmup), ArcFace head trained — NOT [FINE-TUNED]**
> The ECAPA-TDNN backbone weights remain frozen at baseline pretrained initialization; only the custom ArcFace classification head was trained. Exactly 1 epoch of warmup training was completed. Full backbone unfreezing is NOT YET done.

---

## 1. Model Details

| Field | Value |
|---|---|
| **Model Name** | AAA-SV (Adaptive Speaker Verification) |
| **Model Version** | 0.1.0-alpha (EXPERIMENTAL) |
| **Model Label** | `[BASELINE] backbone (frozen ep1 warmup), ArcFace head trained` ([BASELINE]+[CUSTOM] head) |
| **Architecture** | ECAPA-TDNN + Attentive Statistics Pooling (ASP) + ArcFace head + optional StabilityHead q(x) |
| **Embedding Dimension** | 192-dimensional float32, L₂-normalised ($\|e\|_2 \approx 1.0 \pm 10^{-5}$) |
| **Feature Extraction** | 80-channel log Mel-filterbank, 16 kHz sample rate, sentence-level mean normalization |
| **Loss (Classification)** | Additive Angular Margin Softmax (ArcFace), $s=30.0$, initial $m=0.00 \to 0.05$ |
| **Loss (Stability Head)** | MSE between predicted stability $q(x_{\text{deg}})$ and target cosine similarity to reference |
| **Total Objective** | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ArcFace}} + 0.1 \cdot \mathcal{L}_q$ |
| **Base Weights** | SpeechBrain `spkrec-ecapa-voxceleb` (Apache-2.0) |
| **Lead Developer** | Navadeep Maka (Lead Model Engineer, AAA-SV) |
| **Branch / Repository** | `feat/navadeep-aaa-sv` — `Speech-Synthesis/AAA_Layer1` |

---

## 2. Training Provenance

| Parameter | Value |
|---|---|
| **Training Run ID** | `20261113-1718_09dc907_5e10bf` |
| **HPC Job ID** | `93483.amritahpc` (B1 epoch-1 warmup, COMPLETED) |
| **Joint Training Job** | `93490.amritahpc` (AAA-SV joint training with stability head, running ~12h) |
| **Epochs Completed** | 1 epoch (warmup only) |
| **Backbone Status** | FROZEN (warmup schedule, gradients disabled for backbone) |
| **Margin Schedule** | $m=0.05 / 0.2$ ($m$ reached 0.05 rad out of 0.20 rad target) |
| **Training Loss** | $1.58 \to 0.92$ (started at 1.5819, ended at ~0.92) |
| **Head Learning Rate** | $1.0 \times 10^{-3}$ (AdamW) |
| **Backbone Learning Rate** | $1.0 \times 10^{-4}$ (not applied; backbone frozen) |
| **Random Seed** | 42 |
| **Compute Environment** | Amrita HPC Cluster, Tesla V100-PCIE-16GB GPU |

---

## 3. Training & Validation Data

- **Training Corpus:** VoxCeleb1 Development Partition (1,211 speakers, 112,470 utterances).
- **Validation Corpus:** VoxCeleb1 Dev holdout (1,211 speakers, 2,422 utterances, 733,866 trial pairs).
- **Train Manifest Hash:** `1a3c010b39da65d63701dfa3ef8307d64690f09cf6e3b256eb5b4b968942bc3f`
- **Val Manifest Hash:** SHA256 starts with `9739f2a8...` (`9739f2a80d8a6e8e2e...`)
- **Leakage Prevention:** 0 data leakage. All 40 official VoxCeleb1 test speakers (`id10270` through `id10309`) were strictly excluded prior to splitting.
- **Specification Deviation Note:** The original sprint specification noted Common Voice Set A for validation; VoxCeleb1 dev holdout was used due to dataset availability. Formal approval requested in `docs/evidence/day4/NEEDS_SURYA.md`.

---

## 4. Checkpoint Integrity & Artifacts

| Checkpoint File | SHA256 Hash | Location / Status |
|---|---|---|
| `checkpoint_best.pt` | `f78e2b6c1c20bdee8930dd5e04c47837e962c7aae8cee74866c71c212b03cdbd` | HPC cluster: `runs/20261113-1718_09dc907_5e10bf/` |
| `checkpoint_last.pt` | `260db647fc93f6fbade2c4b11937a255c681974358ba01a7087f37d66e10e67a` | HPC cluster: `runs/20261113-1718_09dc907_5e10bf/` |

> [!NOTE]
> Per sprint data rules, model weights (`.pt`) are retained on cluster storage and NOT committed into git. Full hashes are recorded in `RELEASE_ARTIFACTS/SHA256SUMS.txt` and `docs/evidence/day1/checkpoint_sha256.txt`.

---

## 5. Quantitative Performance

> [!CAUTION]
> **EER, FRR@FAR=1%, and minDCF are NOT YET MEASURED.** Evaluation was blocked by feature extraction and checkpoint key mismatch bugs in `eval_toolkit.py`, which are being fixed. Real numerical metrics must not be fabricated.

| Metric | Target | Measured Value | Status |
|---|---|---|---|
| **EER (Equal Error Rate)** | $\le 2.0\%$ | **not yet measured — eval job pending** | See `docs/evidence/day4/EVAL_STATUS.md` |
| **FRR @ FAR=1%** | $\le 3.0\%$ | **not yet measured** | See `docs/evidence/day4/EVAL_STATUS.md` |
| **minDCF** ($P_{\text{target}}=0.05, C_{\text{fa}}=1, C_{\text{miss}}=10$) | $\le 0.20$ | **not yet measured** | See `docs/evidence/day4/EVAL_STATUS.md` |
| **Epoch 1 Warmup Loss** | N/A | $1.58 \to 0.92$ | Verified from `metrics.jsonl` |
| **Unit Test Coverage** | 100% contract | 113/113 passing | Passing locally on Windows CPU |

---

## 6. Latency & Resource Utilization

- **Inference Latency Target:** $\le 300\text{ ms}$ p95 on 3.0s speech audio (CPU x86_64).
- **Current Latency Status:** **not yet measured on real model; stub benchmark script at hpc/measure_latency.py**.
- **Memory Footprint:** 192-dim float32 embeddings require 768 bytes per stored voiceprint template.

---

## 7. Offline Load Verification

- **Offline Load Status:** **PASS — wrapper runs without network using staged config files; see docs/evidence/day4/offline_load_proof.json**.
- **Proof Mechanism:** The AAASV wrapper instantiates with staged local configuration (`config.yaml`, `preproc.yaml`, `calibration.json`) without making any outbound socket or network requests.
- **Reproducibility:** Run `python hpc/offline_load_test.py` locally or offline; verification output is saved to `docs/evidence/day4/offline_load_proof.json`.

---

## 8. Operating Contract, Bounds & Error Codes

### Audio Requirements & Duration Bounds
- **Input Sampling Rate:** 16,000 Hz single-channel (mono), float32.
- **Duration Lower Bound:** 1.0 s ($< 1.0\text{ s}$ rejected with `E_TOO_SHORT`).
- **Duration Upper Bound:** 30.0 s ($> 30.0\text{ s}$ rejected with `E_TOO_LONG`).
- **Short Duration Warning:** $1.0\text{ s} \le t < 2.0\text{ s}$ accepted with degraded confidence and `W_SHORT` warning.

### Standardized Error Codes
| Code | Category | Trigger Condition |
|---|---|---|
| `E_FORMAT` | Client Input | Input array is not 1D float32 or invalid channel count |
| `E_DECODE` | Client Input | Audio buffer cannot be decoded into valid PCM |
| `E_TOO_SHORT` | Duration Bound | Usable speech duration $< 1.0\text{ s}$ |
| `E_TOO_LONG` | Duration Bound | Audio duration $> 30.0\text{ s}$ |
| `E_SILENT` | Signal Quality | Audio RMS energy $< -60\text{ dBFS}$ or zero signal |
| `E_NONFINITE` | Signal Quality | Audio waveform contains NaN or Inf values |
| `E_NO_ENROLLMENT` | Protocol | Attempted verification against an empty or missing voiceprint |
| `E_MODEL_UNAVAILABLE` | Environment | Model checkpoint/weights missing or corrupted |
| `E_VERSION_MISMATCH` | Environment | Staged configuration or checkpoint version incompatible |
| `E_INTERNAL` | Runtime | Unexpected exception during inference execution |

---

## 9. Known Limitations & Deviations

1. **Epoch 1 Warmup Only:** Only 1 epoch of warmup training has been completed. Backbone weights were frozen.
2. **Backbone Not Unfrozen:** Backbone fine-tuning (requires ~24h GPU time) was not executed; model label is strictly `[BASELINE]`, not `[FINE-TUNED]`.
3. **Margin Ramp Incomplete:** ArcFace margin reached only 0.05 rad out of the target 0.20 rad.
4. **Validation Split Source:** Val split extracted from VoxCeleb1 dev set instead of Common Voice Set A.
5. **No Vox1-O Test Evaluation:** Standard VoxCeleb1-O test trial pairs not evaluated due to test audio staging dependencies.
6. **Unfitted Calibration:** `calibration.json` is deployed with default heuristic parameters (`fitted=false`, threshold=0.5, $\tau_{\text{anchor}}=0.85$, $q_{\text{min}}=0.70$) awaiting Common Voice Set B calibration data.
7. **Downstream Spoof Vulnerability:** This model does not detect replay or synthesized speech attacks; spoof protection is deferred to Layer 2 (AASIST).
