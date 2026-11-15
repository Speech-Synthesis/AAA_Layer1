# AAA-SV Model Registry Manifest

## Model Information

| Field | Value |
|---|---|
| **Model Identifier** | `aaa-sv-b1-ep1` |
| **Model Name** | Adaptive Acoustic Speaker Verification (AAA-SV) |
| **Version** | `0.1.0-alpha (EXPERIMENTAL)` |
| **Architecture** | ECAPA-TDNN + Attentive Statistics Pooling + ArcFace Head (+ optional StabilityHead $q(x)$) |
| **Classification Label** | `[BASELINE] backbone frozen (epoch 1 warmup); [CUSTOM] ArcFace head` |
| **Status** | **EXPERIMENTAL — not [VERIFIED], EER not measured** |
| **Lead Engineer** | Navadeep Maka |

---

## Artifact Integrity & Checkpoints

| Field | Value |
|---|---|
| **Primary Checkpoint** | `checkpoint_best.pt` |
| **SHA256 Checksum** | `f78e2b6c1c20bdee8930dd5e04c47837e962c7aae8cee74866c71c212b03cdbd` |
| **Last Checkpoint** | `checkpoint_last.pt` |
| **Last Checkpoint SHA256** | `260db647fc93f6fbade2c4b11937a255c681974358ba01a7087f37d66e10e67a` |
| **Training Run ID** | `20261113-1718_09dc907_5e10bf` |
| **HPC Job ID** | `93483.amritahpc` |
| **Git Repository** | `https://github.com/Speech-Synthesis/AAA_Layer1.git` |
| **Branch / Commit** | `feat/navadeep-aaa-sv` / `460b2f0` |

---

## Technical Dependencies

- **Runtime:** Python 3.10 / 3.12
- **Deep Learning Framework:** PyTorch 1.13.1 (CUDA 11.6 on HPC) / PyTorch 2.x (CPU local)
- **Audio & Speech Toolkits:** `torchaudio == 0.13.1` (or matching PyTorch version), `speechbrain == 0.5.16`
- **Scientific Computing:** `numpy >= 1.23.0`, `scipy >= 1.9.0`, `soundfile >= 0.12.0`

---

## Known Limitations

1. **Epoch 1 Warmup Only:** Exactly 1 training epoch completed. The ECAPA-TDNN backbone remained frozen during training.
2. **Backbone Status:** Frozen at pretrained baseline initialization; model must not be cited as `[FINE-TUNED]`.
3. **Margin Ramp:** ArcFace angular margin reached only $m=0.05$ rad (out of target $0.20$ rad).
4. **Validation Metrics:** EER, FRR@FAR=1%, and minDCF are **not yet measured** due to HPC evaluation queue and pending feature extraction script fixes.
5. **Calibration Status:** `calibration.json` deployed with `fitted=false` awaiting Common Voice Set B calibration data.
