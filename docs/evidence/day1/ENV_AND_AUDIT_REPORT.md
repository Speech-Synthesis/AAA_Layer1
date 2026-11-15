# ENV_AND_AUDIT_REPORT.md
## AAA-SV Preflight & Audit Report
**Date:** 2026-11-11 | **Auditor:** Antigravity (AI pair) | **Branch target:** `feat/navadeep-aaa-sv`

---

> **ACTION REQUIRED before any writes:** Current branch is `master`; `feat/navadeep-aaa-sv` does not yet exist.
> Create and switch to that branch before any sprint code changes.

---

## 1. Git State

| Field | Value |
|---|---|
| **Current branch** | `master` |
| **Default branch** | `master` (origin/HEAD → origin/master) |
| **Latest commits** | `2a416bc` Configure training pipeline for VoxCeleb1 on HPC V100 |
| | `f882aba` feat: AAA Engine Layer 1 - Speaker Verification with HPC training setup |
| **Remote** | `origin https://github.com/Speech-Synthesis/AAA_Layer1.git` |
| **Working tree** | Clean — nothing to commit |
| **Open branches** | `master` only (no feature branches exist) |
| **Stash** | Empty |

**Gap:** Feature branch `feat/navadeep-aaa-sv` does not exist. Must be created before any sprint work.

---

## 2. Repository Map

### 2a. Training & Data-Prep Scripts

| Script | Purpose | Inputs | Outputs | Runnable? |
|---|---|---|---|---|
| `aaa_engine/training/train_ecapa.py` | Fine-tune ECAPA-TDNN with AAM-Softmax; inline `prepare_csv_from_folder()` | YAML hparams, audio folder | Checkpoints under `results/`, `train_log.txt` | ⚠ Partial — needs `data_folder`, missing augmentation, AdamW, cosine schedule, duration hardcoded at 3 s only |
| `aaa_engine/training/prepare_data.py` | Scan speaker dirs → CSV manifests (train/dev/test) + test-pair CSV | `sample_dataset/` | `manifests/{train,dev,test,test_pairs}.csv` | ✅ Runnable locally with sample data |
| `aaa_engine/training/generate_sample_dataset.py` | Synthesise 20 speakers × 30 utterances as 16 kHz WAV | — | `sample_dataset/speaker_NNNN/utterance_NN.wav` | ✅ Runnable; for smoke tests only |
| `aaa_engine/training/evaluate.py` | Compute EER from test pairs using trained checkpoint | `manifests/test_pairs.csv`, checkpoint | Console EER report | ⚠ Hard-coded result path; no manifest-hash locking; no VoxCeleb-O support |
| `aaa_engine/training/export_model.py` | Copy best checkpoint + generate `hyperparams.yaml` + `custom.py` for inference | Training checkpoint | `pretrained_models/ecapa-tdnn-custom/` | ✅ Runnable after training |

### 2b. Config Files

| File | Purpose | Notes |
|---|---|---|
| `aaa_engine/training/train_hparams.yaml` | Local training config (batch=8, 5 epochs, no augmentation) | `data_folder: !PLACEHOLDER` correct; seed 1986; uses CyclicLR, Adam (not AdamW) |
| `aaa_engine/training/train_hparams_hpc.yaml` | HPC config (batch=16, 10 epochs, random_chunk=True) | Same gaps as above |
| `configs/config.json` | Supplementary JSON config — mirrors YAML | Inconsistent: `out_n_neurons: 20` vs `out_n_neurons: 1211` in YAML; not read by training script |
| `train_job.sh` | PBS job script | Contains personal absolute path (see Section 3) |

### 2c. Inference & API Scripts

| Script | Purpose |
|---|---|
| `aaa_engine/layer1/speaker_verifier.py` | Load model (singleton), preprocess audio, extract 192-d embedding, cosine verify |
| `aaa_engine/layer1/enrollment.py` | Enroll / re-enroll via DB |
| `aaa_engine/layer1/db_utils.py` | PostgreSQL voiceprint store / fetch / auth-log |
| `aaa_engine/main.py` | FastAPI app: `/auth/enroll`, `/auth/verify` |

### 2d. Tests

`tests/test_layer1.py` — 20 unit tests: cosine math, threshold boundary, verify pipeline, embedding shape/norm, enrollment flow, DB utils (all mocked).

---

## 3. Hard-Coded HPC Assumptions

| File | Line | Assumption |
|---|---|---|
| `train_job.sh` | L4–L8 | `#PBS` directives: queue `gpu`, `select=1:ncpus=8:ngpus=1:mem=32gb`, `walltime=24:00:00` |
| `train_job.sh` | L9 | `#PBS -M your.email@university.edu` — placeholder but must be personalised |
| `train_job.sh` | L44 | `module load cuda/11.6` — specific CUDA module name |
| `train_job.sh` | L48–L49 | `micromamba shell hook` + `micromamba activate speaker_sv` |
| **`train_job.sh`** | **L76** | **`--data_folder=/home/n_harini/datasets/voxceleb1_sv/vox1_dev_wav/wav`** — absolute personal home path; already in Git history (commit `2a416bc`) |
| `aaa_engine/training/evaluate.py` | L32–L34 | Hard-coded relative result path `../results/ecapa_sample/1986/save` |
| `aaa_engine/training/export_model.py` | L24–L28 | Hard-coded relative paths for checkpoint and export dirs |
| `aaa_engine/training/prepare_data.py` | L36–L37 | `DATASET_DIR` and `OUTPUT_DIR` relative to `__file__` — only works from `aaa_engine/` cwd |

---

## 4. Dependencies

### 4a. Environment (confirmed by import test)

| Library | Installed Version | Notes |
|---|---|---|
| Python | **3.12.4** | ✅ |
| PyTorch | **2.10.0+cpu** | CPU-only build — no CUDA on this machine |
| torchaudio | **2.10.0+cpu** | CPU-only |
| speechbrain | **1.1.1** | `speechbrain.pretrained` deprecated → now `speechbrain.inference` |
| numpy | **2.3.5** | ✅ |
| scipy | **1.16.3** | ✅ |
| librosa | **1.0.0** | ✅ |
| soundfile | **0.12.1** | ✅ |
| hyperpyyaml | installed | ✅ |
| fastapi | **0.110.0** | ✅ |
| psycopg2-binary | installed | ✅ (gracefully disabled when DB absent) |

### 4b. Requirements File

`requirements.txt` has **no pinned versions** — only bare package names. This is a reproducibility risk.

### 4c. Missing Tools

| Tool | Status | Impact |
|---|---|---|
| **ffmpeg** | **NOT FOUND** on PATH | Codec-chain augmentation (B1 requirement) cannot run. |
| CUDA / GPU | Not present locally | Training must go to HPC; local runs are CPU-only. |

---

## 5. Hardware (Local Machine)

| Resource | Value |
|---|---|
| GPU | **None** (CPU-only PyTorch build) |
| CPU cores | **20** |
| RAM total | **25.5 GB** |
| RAM available | ~12.4 GB (at time of audit) |
| CUDA | Not available |

**Implication:** All real training runs go to HPC. Local runs limited to smoke tests on synthetic dataset.

---

## 6. Model Checkpoint Status

### Official Pretrained: `speechbrain/spkrec-ecapa-voxceleb` [BASELINE]

**Status: ✅ Already staged locally in HF cache — no download required.**

| File | Size | Location |
|---|---|---|
| `embedding_model.ckpt` | **83.3 MB** | `~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d…/` |
| `classifier.ckpt` | 5.5 MB | Same |
| `mean_var_norm_emb.ckpt` | 1.9 KB | Same |
| `label_encoder.txt` | 125.6 KB (1,211 speakers) | Same |

Total parameters in embedding model: ~20.8 M (matches published ECAPA-TDNN C=1024).

### Custom Export: `ecapa-tdnn-custom/` (do NOT use for evaluation)

| File | Size | Notes |
|---|---|---|
| `classifier.ckpt` | 16 KB | Only 3,840 params — 20-class head from synthetic data |
| `mean_var_norm_emb.ckpt` | 1.6 KB | Normaliser state |
| **`embedding_model.ckpt`** | **MISSING** | **Backbone weights were not exported — inference from this directory will fail** |

> **CAUTION:** The `ecapa-tdnn-custom/` directory is missing `embedding_model.ckpt`. If `ECAPA_MODEL_PATH` points here, inference will crash. This export is from a 20-speaker synthetic run only.

---

## 7. Code Behaviour Checklist

### 7a. Preprocessing

| Aspect | Current behaviour | Spec requirement | Status |
|---|---|---|---|
| Sample rate | 16 kHz (resample if needed) | 16 kHz | ✅ |
| Channels | Stereo → mono (mean) | Mono | ✅ |
| Data type | float32 (implicit) | float32 | ✅ |
| Duration range (inference) | 1.5 s – 10.0 s (`MIN_DURATION=1.5`) | 1.0 s – 10.0 s | ⚠ GAP: inference rejects < 1.5 s |
| **Silence trimming** | **`librosa.effects.trim(top_db=25)` in `preprocess_audio()`** | No VAD/trimming (spec says no VAD in training) | ⚠ GAP: trim in inference only |
| **Repeat-padding** | **Not present** | Not permitted | ✅ GOOD |
| Training crop | Fixed `sentence_len=3.0 s`, or single-length `random_chunk=True` | `{1,2,3,4}` s uniformly per batch | ⚠ GAP: no multi-duration batch sampling |
| Train/inference parity | Training: `torchaudio.load` + no trim; Inference: same + `librosa.trim` | Same preprocessing in both | ⚠ GAP: mismatch |

### 7b. Augmentation

| Status | Detail |
|---|---|
| **MISSING** | Both configs explicitly disable augmentation |
| Required for B1 | MUSAN noise, RIRS_NOISES reverb, codec chain, resample chain, band-limit |
| Probabilities | From AAA-SV.md section 13 — file not present in repo |
| Codec chain | Blocked by missing `ffmpeg` |

### 7c. Loss (AAM-Softmax)

| Aspect | Current | Spec (B1) | Status |
|---|---|---|---|
| Margin | 0.2 (fixed) | Ramp 0 → 0.2 over first 20% of steps | ⚠ GAP |
| Scale | 30 | 30 | ✅ |

### 7d. Optimizer & Scheduler

| Aspect | Current | Spec (B1) | Status |
|---|---|---|---|
| Optimizer | **Adam** | **AdamW** | ⚠ GAP |
| Param groups | Single LR for all params | Backbone 1e-4, head 1e-3 | ⚠ GAP |
| Scheduler | CyclicLR | Cosine + 500-step warmup | ⚠ GAP |
| Gradient clip | Not present | clip 5.0 | ⚠ GAP |
| Mixed precision | Not implemented | Required | ⚠ GAP |
| Gradient accumulation | Not implemented | Required for local smoke | ⚠ GAP |
| Backbone freeze | Not implemented | First epoch frozen (head warm-up) | ⚠ GAP |

### 7e. Checkpointing

| Aspect | Current | Spec (B1) | Status |
|---|---|---|---|
| Atomic writes | Not implemented; SB writes directly | Temp-then-rename | ⚠ GAP |
| Keep last + best | Only best | Last AND best | ⚠ GAP |
| Saved state | model + classifier + normaliser + epoch + LR scheduler | + optimizer, grad scaler, RNG states, sampler state, config hash, manifest hash, git sha | ⚠ GAP |
| Run ID format | None | `YYYYMMDD-HHMM_<gitsha7>_<cfghash6>` | ⚠ GAP |
| Auto-resume | SpeechBrain implicit | Explicit with validation | ✅ Partial |

### 7f. Logging

| Aspect | Current | Spec | Status |
|---|---|---|---|
| Train log | `train_log.txt` only | Also `metrics.jsonl`, pip freeze, git diff, GPU info, seed, manifest hashes | ⚠ GAP |
| Early stopping metric | Classification error rate | **Validation EER and FRR@FAR1%** | ⚠ GAP (wrong metric for SV) |

### 7g. Seeds — Contradiction Found

`set_seed()` sets `torch.backends.cudnn.deterministic = True`, but then `__main__` (line 364) immediately sets `torch.backends.cudnn.benchmark = True`, which overrides determinism. **This breaks GPU reproducibility.** One line must be removed.

### 7h. Tests — What Exists vs. What is Required

| Required test | Current status |
|---|---|
| Cosine similarity math | ✅ 6 tests |
| Threshold boundary | ✅ 4 tests |
| Verify pipeline | ✅ 5 tests |
| Embedding shape/norm | ✅ 2 tests (mocked) |
| Enrollment flow | ✅ 5 tests |
| Audio preprocessing | ✅ 5 tests |
| DB utils (mocked) | ✅ 4 tests |
| **Resume test** | ❌ Missing |
| **Preprocessing parity (train vs inference)** | ❌ Missing |
| **Shuffled-label sanity** | ❌ Missing |
| **Embedding norm check (live model)** | ❌ Missing |
| **Augmentation reproducibility** | ❌ Missing |

---

## 8. Gap List vs. Spec (Prioritised by Training Blocker)

Spec files `AAA-SV.md` and `SHARED.md` are **not present** in the repository. Gaps assessed against the brief.

| # | Gap | Blocks | Effort |
|---|---|---|---|
| **B1** | Feature branch `feat/navadeep-aaa-sv` does not exist | Everything | 1 min |
| **B2** | Spec docs (`AAA-SV.md`, `SHARED.md`) absent | Gap verification | 30 min stubs |
| **B3** | JSONL manifest infrastructure absent (sha256, licence, consent_id, duration, session_id, device, split) | Dataset governance, leakage checker | High |
| **B4** | Current split is utterance-level per speaker, not speaker-disjoint across train/val/test | Valid B1 training | Medium |
| **B5** | Augmentation pipeline absent (MUSAN, RIRS, codec, resample, band-limit) | B1 training | High |
| **B6** | `ffmpeg` not on PATH | B1 codec augmentation | Medium (install) |
| **B7** | Optimizer is Adam; spec requires AdamW | B1 | Low code |
| **B8** | No separate param groups (backbone vs head LR) | B1 fine-tune | Low code |
| **B9** | LR scheduler is CyclicLR; spec requires cosine+warmup | B1 schedule | Low code |
| **B10** | Backbone freeze for first epoch not implemented | B1 head warm-up | Low code |
| **B11** | Margin ramp 0→0.2 not implemented | B1 loss | Low code |
| **B12** | Mixed precision not in training script | B1 speed | Low code |
| **B13** | Gradient clipping not implemented | B1 stability | Low code |
| **B14** | Gradient accumulation not implemented | Local smoke | Low code |
| **B15** | cuDNN `benchmark=True` contradicts `deterministic=True` | Reproducibility | 1-line fix |
| **B16** | Checkpoint missing: optimizer, grad scaler, RNG, config hash, manifest hash, git sha | Resume | Medium |
| **B17** | No atomic checkpoint writes | Safety | Low code |
| **B18** | Early stopping / best-ckpt on classification error, not EER/FRR@FAR1% | Valid selection | Medium |
| **B19** | `metrics.jsonl` / structured log missing | Evidence | Low |
| **B20** | Run ID format missing | Traceability | Low |
| **B21** | `speechbrain.pretrained` deprecated; should be `speechbrain.inference.speaker` | Forward compat | 1-line fix |
| **B22** | `test_pairs.csv` has absolute paths from `E:\AAA\...` (another machine) | Evaluation runnable | Regenerate |
| **B23** | `MIN_DURATION=1.5 s` vs spec 1.0 s | Inference contract | 1-line fix |
| **B24** | `librosa.trim` in inference but not training — preprocessing mismatch | Pipeline parity | Decision needed |
| **B25** | Missing tests: resume, preprocessing parity, shuffled-label, embedding norm, augmentation repro | Spec compliance | Medium |
| **B26** | Trial generator for bucketed evaluation absent | B0 evaluation | High |
| **B27** | VoxCeleb1-O cleaned trial list not available locally | B0 full eval | HPC data |
| **B28** | Calibration infrastructure (cohort/AS-norm, PLDA) absent | Day 2 | High |
| **B29** | HPC job script: personal absolute path (`/home/n_harini/...`) in Git history | HPC launch safety | 1-line fix |
| **B30** | `requirements.txt` has no pinned versions | Reproducibility | Low |

---

## 9. Recommended Minimal-Change Plan

### Phase 0 — Foundation (Day 1 morning, ~1 h)
1. `git checkout -b feat/navadeep-aaa-sv` from master
2. Create `docs/` with stub `AAA-SV.md` and `SHARED.md` (or confirm Surya has them)
3. Fix cuDNN contradiction (remove `benchmark=True` from `__main__`, keep deterministic) — `train_ecapa.py:364`
4. Fix deprecated import (`speechbrain.pretrained` → `speechbrain.inference.speaker`) — `speaker_verifier.py:19`, `train_ecapa.py:420`
5. Fix `MIN_DURATION` to `1.0` — `speaker_verifier.py:27`
6. Run existing tests → save output as `docs/evidence/day1/test_layer1_output.txt`

### Phase 1 — B0 Baseline Evaluation (Day 1, ~2 h)
7. Write `sv_eval.py`: load pretrained checkpoint from HF cache, extract embeddings, cosine scoring, EER + FRR@FAR1%, bootstrap CI — locked to manifest hash
8. Confirm embedding dim=192, L2-norm=1.0 at import time
9. Record result in `baseline_smoke_result.md` (synthetic data only until HPC data available)
10. Build trial generator scaffold with duration/SNR buckets

### Phase 2 — B1 Training Pipeline (Day 1–2)
11. New configs: `configs/train_local_smoke.yaml` (batch 4, grad_accum 8, 2 epochs, synthetic subset) + `configs/train_hpc.yaml` (all paths as labelled placeholders)
12. Rewrite training script (`train_b1.py` on feature branch) with: AdamW, two param groups, cosine schedule + warmup, backbone freeze epoch 1, margin ramp, mixed precision, grad clip, grad accumulation, full checkpoint state, atomic writes, `metrics.jsonl`, run ID
13. Add `augment.py` — conditioned on `ffmpeg` available; skip codec chain if absent and log the skip
14. Speaker-disjoint split logic in manifest builder
15. Run local smoke test → save artefacts under `runs/`
16. Run 5 required tests; save outputs under `docs/evidence/day1/`

### Phase 3 — HPC Launch Prep (Day 2)
17. HPC job script with no personal paths (all `${PLACEHOLDER}`)
18. `LAUNCH_CHECKLIST.md`, `PROFILING_GUIDE.md`
19. `NEEDS_SURYA.md` with open questions

### Phase 4 — Dataset Governance & Recording Kit (Day 1–2 parallel)
20. `DATASET_PLAN.md`
21. `CONSENT_FORM.md`, `RECORDING_PROTOCOL.md`, manifest-builder script design

---

## Appendix: Files Frozen — Do Not Modify

Per scope rules: `aaa_engine/main.py`, `aaa_engine/layer1/db_utils.py`, `aaa_engine/layer1/enrollment.py`, `aaa_engine/layer1/speaker_verifier.py`, `setup_db.sql`, `aaa_engine/__init__.py`, `aaa_engine/layer1/__init__.py`.

Changes needed in those modules (deprecated import fix, preprocessing alignment) → logged in `NEEDS_SURYA.md` for joint review.

---

*Report generated by read-only inspection of workspace `d:\AAA_Layer1` on 2026-11-11T22:18 IST.*
*No other files were modified in producing this report.*
