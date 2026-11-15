# B1 Training Pipeline Smoke Run — Evidence

> **Document Scope:** This document records the **B1 training pipeline smoke run** on synthetic audio data, verifying that the fine-tuning script (`train_b1.py`), data loader, loss computation, and checkpointing run end-to-end on a clean working tree. This is **NOT** a B0 baseline evaluation (zero-shot evaluation on real benchmark data is documented in the dedicated section below).

**Run ID:** `20261112-0758_ca0744b_6596ea`  
**Date:** 2026-11-12 13:32 IST  
**Branch:** `feat/navadeep-aaa-sv`  
**Git SHA:** `ca0744b` (clean HEAD, verified with 0-byte `git_diff.txt`)  
**Config hash:** `6596ea`  
**Status:** SMOKE OK (Clean Working Tree)  

---

## Command

```powershell
cd d:\AAA_Layer1
python -m aaa_engine.training.train_b1 --config configs/train_local_smoke.yaml
```

---

## Environment

| Item | Value |
|---|---|
| Python | 3.12.4 |
| PyTorch | 2.10.0+cpu |
| SpeechBrain | 1.1.1 |
| librosa | 1.0.0 |
| soundfile | 0.12.1 |
| GPU | None (CPU only) |
| Seed | 1986 |

---

## Provenance Hashes

| Artifact | SHA-256 |
|---|---|
| `train.csv` (manifest) | `27189a59bc12659472c9ce9ac3fbcbc612991608269c1ee9724ff43de00f63e5` |
| `checkpoint_best.pt` (epoch 1) | `de1e6523e4f1a4c923f9c0ee7774780d3183db5248133c4da923310b79d77cb0` |
| `checkpoint_last.pt` (epoch 2) | `d03262b4af13d7ec6b422344515a9a57e0c8507ac6818cb097756a72af0e37cc` |

*Working tree provenance: `git_diff.txt` in the run directory is 0 bytes (empty). The run was executed strictly on clean commit `ca0744b`.*

---

## Training Log

```
[13:28:40] INFO AAA Engine B1 Training — Run ID: 20261112-0758_ca0744b_6596ea
[13:28:40] INFO [seed] All RNGs seeded to 1986
[13:28:40] INFO [device] Using: cpu
[13:28:44] INFO [run] Run directory: runs\20261112-0758_ca0744b_6596ea
[13:28:44] INFO [data] Training speakers: 20
[13:28:44] WARNING [augment] ffmpeg not found on PATH — codec augmentation disabled.
[13:28:45] INFO [BASELINE] Loading pretrained ECAPA-TDNN from: speechbrain/spkrec-ecapa-voxceleb
[13:28:45] INFO [BASELINE] Direct ckpt load (Windows-safe): embedding_model.ckpt
[13:28:45] INFO [BASELINE] Pretrained ECAPA-TDNN weights loaded (192-dim). [direct ckpt]
[13:28:45] INFO [EXPERIMENT] Initialising class centres from reference embeddings …
[13:30:41] INFO [EXPERIMENT] Class centre initialisation complete.
[13:30:41] INFO [train] Backbone frozen for epoch 1.
[13:30:42] INFO [epoch 1] crop_len=1.0 s, 105 batches
[13:31:12] INFO [1/2] step=10 loss=9.2086 margin=0.200 lr_bb=2.00e-05 lr_hd=2.00e-04 gn=6.12
[13:31:22] INFO [epoch 1] Running validation …
[13:31:22] WARNING [val] No trial pairs could be built — skipping EER computation.
[13:31:22] INFO [epoch 1] val_EER=1.0000  val_FRR@FAR1%=1.0000
[13:31:22] INFO [ckpt] Saved checkpoint_last.pt (epoch 1)
[13:31:22] INFO [ckpt] New best EER=1.0000 — saved checkpoint_best.pt
[13:31:22] INFO [train] Backbone unfrozen at epoch 2.
[13:31:22] INFO [epoch 2] crop_len=1.0 s, 105 batches
[13:31:58] INFO [2/2] step=20 loss=7.7624 margin=0.200 lr_bb=4.00e-05 lr_hd=4.00e-04 gn=9.31
[13:32:28] INFO [epoch 2] Running validation …
[13:32:29] WARNING [val] No trial pairs could be built — skipping EER computation.
[13:32:29] INFO [epoch 2] val_EER=1.0000  val_FRR@FAR1%=1.0000
[13:32:29] INFO [ckpt] Saved checkpoint_last.pt (epoch 2)
[13:32:29] INFO [early-stop] No improvement. Patience 1/3
[13:32:29] INFO Training complete.  Best val EER: 1.0000
[13:32:29] INFO Run ID: 20261112-0758_ca0744b_6596ea
```

---

## Metrics (from metrics.jsonl)

| epoch | global_step | loss | margin | lr_backbone | lr_head | grad_norm |
|---|---|---|---|---|---|---|
| 1 | 10 | 9.2086 | 0.200 | 2.00e-05 | 2.00e-04 | 6.119 |
| 2 | 20 | 7.7624 | 0.200 | 4.00e-05 | 4.00e-04 | 9.314 |

> **Step Count Verification:** With 420 training clips, `batch_size = 4` yields 105 micro-batches per epoch. With `grad_accum_steps = 8`, the optimizer updates every 8 micro-batches: 105 ÷ 8 = 13 optimizer steps per epoch (26 total optimizer steps across the 2 epochs, with 1 remainder micro-batch per epoch). Because `log_every_n_steps = 10`, metrics are emitted only at step 10 and step 20. The 2 lines in `metrics.jsonl` represent these two 10-step logging events, not the complete optimizer step count.

---

## Validation Result

**val_EER = 1.0000, val_FRR@FAR1% = 1.0000**

> [!IMPORTANT]
> **These numbers are NOT meaningful measurements.** The synthetic dev set (`dev.csv`, 80 utterances across 20 speakers = 4 clips/speaker) does not contain enough cross-speaker pairs to build a valid EER trial list. The validator logs: *"No trial pairs could be built — skipping EER computation"* and falls back to worst-case (1.0).
>
> This is expected behaviour for a pipeline smoke test on synthetic data. Real validation requires real speech datasets (VoxCeleb1 / Common Voice) with adequate trial pairs.

---

## Pipeline Checks ✅

| Check | Result | Notes |
|---|---|---|
| Config loaded without error | ✅ | `configs/train_local_smoke.yaml` parsed cleanly |
| Seeds set (torch/numpy/python/augmenter) | ✅ | Seed 1986 |
| Pretrained weights loaded (direct ckpt, Windows-safe) | ✅ | Bypass symlink WinError 1314 via direct `.ckpt` |
| Class centres initialised from pretrained embeddings | ✅ | Mean normalised embeddings per speaker class |
| Backbone frozen for epoch 1 | ✅ | Gradients disabled for backbone in epoch 1 |
| Backbone unfrozen at epoch 2 | ✅ | Gradients enabled for backbone in epoch 2 |
| AAM margin ramped 0→0.200 | ✅ | Margin reached target 0.200 |
| Loss recorded at steps 10 and 20 | ✅ | Logged at configured 10-step frequency |
| `checkpoint_last.pt` written (atomic) | ✅ | Temporary file write followed by atomic rename |
| `checkpoint_best.pt` written | ✅ | Written on validation step |
| `metrics.jsonl` written | ✅ | Valid JSON lines output |
| `run_config.yaml` saved | ✅ | Saved in run directory |
| `manifest_hash.txt` saved | ✅ | SHA-256 recorded |
| `pip_freeze.txt` saved | ✅ | Environment freeze captured |
| `git_diff.txt` saved | ✅ | Verified 0 bytes (clean tree) |

---

## Historical Runs Archive

| Run ID | Commit | Checkpoint Best SHA-256 | Status | Notes |
|---|---|---|---|---|
| `20261111-1918_9eef8dc_6596ea` | `9eef8dc` | `f479378f16472e1a7a6589ac6537eb2a617d5954c5a52f0f2c8d00a17831ee6c` | SUPERSEDED | Run had uncommitted squeeze fix; superseded by `20261112-0758_ca0744b_6596ea` |

---

## B0 Baseline Evaluation (Pretrained Zero-Shot on VoxCeleb1-O)

**Status:** NOT YET RUN

The B0 baseline evaluation evaluates the published, unmodified `speechbrain/spkrec-ecapa-voxceleb` model via `aaa_engine.training.sv_eval`.

- **Target trial list:** VoxCeleb1-O cleaned trial list (`veri_test2.txt`; exact trial count to be taken directly from file, distinct from the 37,720 trials in the uncleaned `veri_test.txt`).
- **Expected reference performance:** ~0.87% EER on VoxCeleb1-O cleaned.
- **Status in Day 1:** The evaluation script `sv_eval.py` has been implemented, verified, and committed with L2 normalisation check and trial verification. The zero-shot run on VoxCeleb1-O is scheduled for Phase 3 on real benchmark data. No benchmark evaluation was conducted on synthetic data because random synthetic audio does not yield valid speaker verification metrics.
