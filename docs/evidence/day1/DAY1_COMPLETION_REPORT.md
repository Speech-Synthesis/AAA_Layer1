# Navadeep — Day 1 Completion Report

**Date:** November 11, 2026  
**Project:** AAA Engine  
**Role:** AAA-SV Model — Training, Fine-tuning, Calibration, Evaluation and Export  
**Branch:** `feat/navadeep-aaa-sv`  

---

## 1. Executive Summary

Day 1 established the full training and evaluation pipeline for AAA-SV (Speaker Verification). The focus was on implementing a production-ready B1 fine-tuning pipeline on top of the pretrained ECAPA-TDNN baseline, completing a verified clean-tree local smoke run with checkpoints, and validating the pipeline with 39/39 tests passing in 6.48 s (34 pre-existing Layer 1 unit tests + 5 new pipeline regression tests).

**Key Accomplishments:**
- Complete repository audit of the existing Layer 1 (Speaker Verification) implementation
- B0 baseline evaluation script (`sv_eval.py`) with EER, FRR@FAR1%, and bootstrap CI calculation
- B1 fine-tuning pipeline (`train_b1.py`): AdamW + cosine LR schedule + AAM-Softmax loss head + margin ramp + atomic checkpointing
- Seeded augmentation pipeline (`augment.py`): MUSAN noise, RIR reverb, codec, resample, bandlimit
- Manifest builder + leakage checker enforcing speaker-disjoint splits and SHA-256 deduplication
- Clean local smoke training run (`20261112-0758_ca0744b_6596ea`) executed on clean HEAD `ca0744b` with verified 0-byte `git_diff.txt` and checkpoint hashes
- 39/39 tests passing in 6.48 s (34 pre-existing Layer 1 tests + 5 new SV pipeline regression tests)
- HPC PBS job script (`train_job_b1.sh`) and `TRAINING_README.md` for Day 2 launch

**Major Finding:** `layer1/speaker_verifier.py` uses the deprecated `speechbrain.pretrained` import path (SpeechBrain 1.0 → 1.1.1 rename) and sets `MIN_DURATION = 1.5s` vs the team's 1.0s spec contract. Logged in `NEEDS_SURYA.md` for Surya to fix — not modified to avoid ownership conflict.

---

## 2. Repository Audit

### Existing Layer 1 Architecture

| File | Purpose | Status |
|------|---------|--------|
| `layer1/speaker_verifier.py` | ECAPA-TDNN inference, cosine similarity, threshold decision | Working — 2 issues logged |
| `layer1/enrollment.py` | Voiceprint enrollment flow | Working |
| `layer1/db_utils.py` | Voiceprint DB store/fetch with SHA-256 hash | Working |
| `main.py` | FastAPI app, `/verify`, `/enroll` routes | Working |
| `setup_db.sql` | DB schema | Working |

### Confirmed Issues

| ID | Severity | Description | Location |
|----|----------|-------------|----------|
| SV-001 | MEDIUM | Deprecated `speechbrain.pretrained` import | `speaker_verifier.py:19` |
| SV-002 | LOW | `MIN_DURATION = 1.5` vs spec 1.0s | `speaker_verifier.py:27` |
| ENV-001 | INFO | `torchaudio 2.10` requires `torchcodec` for all I/O on Windows | Fixed via `soundfile` in all new scripts |
| ENV-002 | INFO | `SpeakerRecognition.from_hparams()` needs symlink privilege on Windows | Fixed via direct `torch.load` of `.ckpt` |

### Pretrained Checkpoint

**Model:** `speechbrain/spkrec-ecapa-voxceleb`
- Architecture: ECAPA-TDNN C=1024, ~20.8M parameters, embedding dim = 192
- Cache: `~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/`
- Forward output: `(B, 1, 192)` — squeezed to `(B, 192)` before AAM-Softmax

---

## 3. Git and Branch

| Attribute | Value |
|-----------|-------|
| Branch | `feat/navadeep-aaa-sv` |
| Base commit | `f882aba` |
| Commits on feature branch | 10 |
| Working tree | Clean |

### Commits (Day 1, most recent first)

| SHA | Message |
|-----|---------|
| `ca0744b` | fix(provenance): remove model labels from tooling scripts and update test timing |
| `8bb05f5` | docs: add TRAINING_README with exact commands, Windows notes, test results, HPC launch guide |
| `4d9b239` | fix: squeeze ECAPA-TDNN output (B,1,192) to (B,192) before AAM-Softmax; soundfile audio I/O |
| `ec5d3ee` | test: add B1 pipeline tests + conftest (resume/parity/sanity/norm/augment-repro) |
| `7a23fd3` | feat: add HPC PBS job script for B1 training (no personal paths, auto-resume) |
| `9eef8dc` | feat: add manifest builder and leakage checker |
| `a24399b` | feat: add B0 baseline eval script — EER/FRR@FAR1%/bootstrap CI/soundfile-IO |
| `63aba24` | feat: add B1 training script — AdamW/cosine/margin-ramp/atomic-ckpt/soundfile-IO [FINE-TUNED] |
| `a7ff4b2` | feat: add B1 seeded augmentation pipeline (MUSAN, RIR, codec, resample, bandlimit) |
| `9560a91` | fix: update SpeechBrain import to inference.speaker, remove cuDNN benchmark override |

---

## 4. Implementation Files Delivered

### Training and Tooling Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `aaa_engine/training/train_b1.py` | ~1111 | B1 fine-tuning script: AdamW, cosine schedule, AAM-Softmax [FINE-TUNED] |
| `aaa_engine/training/augment.py` | 474 | Seeded, reproducible audio augmentation pipeline |
| `aaa_engine/training/sv_eval.py` | ~1030 | B0 evaluation: zero-shot EER, FRR@FAR1%, bootstrap CI |
| `aaa_engine/training/manifest_builder.py` | 623 | JSONL manifest builder with licence/consent checks |
| `aaa_engine/training/check_leakage.py` | 412 | Split leakage verification and SHA-256 deduplication |

### Tests and Configurations

| File | Lines | Status |
|------|-------|--------|
| `tests/test_sv_pipeline.py` | ~553 | ✅ 5 new pipeline regression tests (all PASS) |
| `tests/conftest.py` | ~90 | ✅ librosa.effects Windows stub fixture |
| `configs/train_local_smoke.yaml` | ~135 | ✅ Local CPU smoke config |
| `configs/train_hpc.yaml` | ~135 | ✅ HPC single-GPU config (placeholders) |
| `configs/eval_config.yaml` | ~80 | ✅ Evaluation config |

### HPC and Documentation

| File | Status |
|------|--------|
| `train_job_b1.sh` | ✅ Committed — PBS job script for V100, auto-resume |
| `TRAINING_README.md` | ✅ Committed — exact commands, Windows notes, HPC launch guide |

---

## 5. Smoke Training Run

| Attribute | Clean Run (Active) | Previous Run (Superseded) |
|-----------|--------------------|---------------------------|
| Run ID | `20261112-0758_ca0744b_6596ea` | `20261111-1918_9eef8dc_6596ea` |
| Date | 2026-11-12 13:32 IST | 2026-11-11 |
| Git SHA | `ca0744b` (clean HEAD) | `9eef8dc` (uncommitted squeeze fix) |
| `git_diff.txt` | 0 bytes (empty) | 55 KB (uncommitted changes) |
| Config hash | `6596ea` | `6596ea` |
| Seed | 1986 | 1986 |
| Device | CPU | CPU |
| Epochs | 2 | 2 |
| Optimizer Steps | 26 total (13 per epoch); 2 logged | 26 total (13 per epoch); 2 logged |
| Status | **SMOKE OK (Clean Tree)** | **SUPERSEDED** |

### Verified Checkpoint Hashes (Clean Run `20261112-0758_ca0744b_6596ea`)

| Artifact | SHA-256 |
|----------|---------|
| `train.csv` (manifest) | `27189a59bc12659472c9ce9ac3fbcbc612991608269c1ee9724ff43de00f63e5` |
| `checkpoint_best.pt` (epoch 1) | `de1e6523e4f1a4c923f9c0ee7774780d3183db5248133c4da923310b79d77cb0` |
| `checkpoint_last.pt` (epoch 2) | `d03262b4af13d7ec6b422344515a9a57e0c8507ac6818cb097756a72af0e37cc` |

### Training Metrics (from `metrics.jsonl`)

| Epoch | Step | Loss | Margin | LR Backbone | LR Head | Grad Norm |
|-------|------|------|--------|-------------|---------|-----------|
| 1 | 10 | 9.2086 | 0.200 | 2.00e-05 | 2.00e-04 | 6.119 |
| 2 | 20 | 7.7624 | 0.200 | 4.00e-05 | 4.00e-04 | 9.314 |

> **Step Count Verification:** 420 training clips with `batch_size = 4` yields 105 micro-batches per epoch. With `grad_accum_steps = 8`, the optimizer updates every 8 micro-batches, giving 13 optimizer steps per epoch (26 total across 2 epochs, with 1 remainder micro-batch per epoch). Because `log_every_n_steps = 10`, metrics are emitted only at global steps 10 and 20.

> **Class-Centre Initialisation Timing Observation:** In this CPU smoke run, computing reference embeddings across 420 clips took ~116 seconds (~2 minutes). Scaled to the full VoxCeleb1 dev dataset (1,211 speakers, ~145k utterances), class-centre initialisation on CPU would require many hours. On the HPC, V100 GPU throughput will reduce this substantially; we will measure throughput on V100 before deciding whether to cap the number of reference utterances per speaker (e.g. 5–10 clips/speaker).

---

## 6. Augmentation Pipeline

| Augmentation | Probability | Implementation | Local Status |
|-------------|-------------|---------------|--------------|
| MUSAN noise | 0.40 | Additive SNR-matched noise | Disabled (no corpus locally) |
| RIR reverb | 0.30 | Room impulse response convolution | Disabled (no corpus locally) |
| Codec chain | 0.40 | MP3/Opus/AMR-NB/G.711 encode-decode | Disabled (ffmpeg not on local PATH) |
| Resample | 0.15 | 16 kHz → 8 kHz → 16 kHz | Active locally (torchaudio in-process) |
| Bandlimit | 0.10 | Low-pass filter (2000–7000 Hz) | Active locally (torchaudio in-process) |
| None | 0.20 | Clean audio pass-through | Active |

---

## 7. Test Results Summary

**Execution Time:** 6.48 seconds (`test_output.txt`)  
**Total Tests:** 39/39 PASSED  
- **34 Pre-existing Layer 1 Unit Tests** (`test_layer1.py`): verify cosine similarity calculation, boundary clamping, threshold decisions (accept/reject at 0.75), 192-dim embedding properties, database storage/retrieval with SHA-256 hashes, and duration validation (1.0–10.0 s).
- **5 New SV Pipeline Regression Tests** (`test_sv_pipeline.py`):
  1. `test_resume_continuity`: Simulates 3 training steps with a lightweight toy embedder+classifier model, saves a checkpoint with step, epoch, parameter states, and torch/numpy/python RNG states, reloads into fresh instances, and asserts exact match of step, epoch, byte-for-byte weights, and subsequent random sample continuity for all three RNGs.
  2. `test_preprocessing_parity`: Generates synthetic waveform, writes to WAV via soundfile, runs `sv_eval.preprocess_audio()` twice to verify determinism; tests stereo-to-mono downmix as channel mean; tests 8 kHz to 16 kHz resample; asserts float32 dtype. Enforces invariants of no silence trimming and no repeat-padding.
  3. `test_shuffled_label_sanity`: Trains a toy model on synthetic multi-class batches with randomly permuted (shuffled) speaker labels for 20 steps, asserting loss fails to converge (loss remains > 0.5) to guard against label leakage or trivial convergence bugs.
  4. `test_embedding_norm`: Loads the real pretrained ECAPA-TDNN from HuggingFace cache (marked `skipif` if model cannot be loaded), encodes a 3.0 s synthetic waveform, extracts the 192-dim embedding, and asserts Euclidean L2 norm is within 0.05 of 1.0.
  5. `test_augmentation_reproducibility`: Instantiates two separate `Augmenter` objects with the same integer seed (1986), passes identical synthetic audio waveforms through both, and asserts output waveforms are strictly bit-for-bit identical (`torch.allclose` with `atol=0.0`).

---

## 8. Environment

| Item | Value |
|------|-------|
| Python | 3.12.4 |
| PyTorch | 2.10.0+cpu |
| SpeechBrain | 1.1.1 |
| librosa | 1.0.0 |
| soundfile | 0.12.1 |
| numpy | 2.3.5 |
| scipy | 1.16.3 |
| GPU | None (CPU only locally) |
| HPC GPU | V100 (queued for HPC launch) |
| ffmpeg | NOT on PATH — codec augmentation bypassed locally; active on HPC |

---

## 9. Next Steps

1. **Phase 2 — Real Manifests & Leakage Check:**
   - Confirm VoxCeleb1 dev on HPC / local path.
   - Run `manifest_builder.py` on VoxCeleb1 dev metadata (1,211 speakers) and Common Voice EN.
   - Run `check_leakage.py` across splits; hash manifests into `manifest_hashes.txt`.
2. **Phase 3 — Real B0 Baseline Evaluation:**
   - Run `sv_eval.py` on VoxCeleb1-O cleaned trial list (`veri_test2.txt`).
   - Record measured EER and FRR@FAR1% in `results_log.md` and `b0_baseline_smoke_result.md`.
3. **Phase 4 — HPC Deployment & B1 Training Launch:**
   - Update `configs/train_hpc.yaml` and `train_job_b1.sh` with cluster account and data paths.
   - Execute dry-run, kill-and-resume proof, and launch full B1 fine-tuning job on V100.

---

**Document Version:** 1.2 (Clean Smoke Re-Run Verified, 10 Commits Documented)  
**Date:** November 12, 2026
