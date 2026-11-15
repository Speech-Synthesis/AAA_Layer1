# AAA-SV — Day 1 Individual Report
**Member:** Navadeep  
**Workstream:** AAA-SV (Speaker Verification — Model, Training, Calibration, Evaluation)  
**Sprint Day:** Day 1 — Wednesday 11 Nov 2026  
**Branch:** `feat/navadeep-aaa-sv`  
**Report Version:** 1.2 (Clean Smoke Re-Run Verified, 10 Commits Documented)  

---

## 1. Work Package Completed Today

Per SHARED.md, Navadeep owns: *"AAA-SV model, training/fine-tuning, calibration, evaluation and export"*

Day 1 scope (per AAA-SV.md Stage 0 + SHARED.md Day 1 exit criteria):

| Stage | Activity | Outcome |
|---|---|---|
| Stage 0 | Audit repo, environment, pretrained checkpoint | Complete — `ENV_AND_AUDIT_REPORT.md` |
| Stage 0 | Dataset governance plan + licence list | Complete — `DATASET_PLAN.md` |
| Stage 0 | B0 baseline evaluation script | Complete — `sv_eval.py` committed |
| Stage 1 | B1 training pipeline runnable locally | Complete — clean smoke run verified, 2 checkpoints saved (`20261112-0758_ca0744b_6596ea`) |
| Stage 1 | Augmentation pipeline (seeded, reproducible) | Complete — `augment.py` committed |
| Stage 1 | HPC job script (auto-resume, no personal paths) | Complete — `train_job_b1.sh` committed |
| Stage 1 | All developer tests green | Complete — 39/39 PASSED in 6.48 s |

---

## 2. Environment

| Item | Value |
|---|---|
| OS | Windows 11 (dev machine) |
| Python | 3.12.4 |
| PyTorch | 2.10.0+cpu |
| SpeechBrain | 1.1.1 |
| librosa | 1.0.0 |
| soundfile | 0.12.1 |
| numpy | 2.3.5 |
| scipy | 1.16.3 |
| GPU (local) | None — CPU only |
| GPU (HPC, planned) | V100 |
| RAM | 25.5 GB |
| ffmpeg | Not on PATH — codec augmentation bypassed locally |

---

## 3. Repository State

**Branch:** `feat/navadeep-aaa-sv`  
**Base commit (master):** `f882aba`  
**Commits added today:** 10

| Commit | Message |
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

**Protected files — NOT touched:** `main.py`, `layer1/db_utils.py`, `layer1/enrollment.py`, `layer1/speaker_verifier.py`, `setup_db.sql`

---

## 4. Files Delivered

### New implementation files (all committed)

| File | Purpose | Provenance label |
|---|---|---|
| `aaa_engine/training/train_b1.py` | B1 fine-tune: AdamW/cosine/AAM-Softmax/atomic-ckpt/resume | [FINE-TUNED] |
| `aaa_engine/training/augment.py` | Seeded aug: MUSAN/RIR/codec/resample/bandlimit | Tooling (no model label) |
| `aaa_engine/training/sv_eval.py` | B0 eval: EER/FRR@FAR1%/bootstrap CI/trial buckets | Tooling (no model label) |
| `aaa_engine/training/manifest_builder.py` | JSONL manifest builder with licence/consent checks | Tooling (no model label) |
| `aaa_engine/training/check_leakage.py` | Speaker overlap + SHA256 dedup across splits | Tooling (no model label) |
| `configs/train_local_smoke.yaml` | Local CPU smoke config (no personal paths) | — |
| `configs/train_hpc.yaml` | HPC single-GPU config (placeholders) | — |
| `configs/eval_config.yaml` | Evaluation config | — |
| `train_job_b1.sh` | PBS job script, auto-resume | — |
| `tests/test_sv_pipeline.py` | 5 pipeline regression tests | — |
| `tests/conftest.py` | pytest fixtures, librosa Windows stub | — |
| `TRAINING_README.md` | Exact commands, Windows notes, HPC launch guide | — |

### Planning and Evidence Documents (in `docs/evidence/day1/`)

| File | Purpose |
|---|---|
| `ENV_AND_AUDIT_REPORT.md` | Full repo audit, preflight check |
| `DATASET_PLAN.md` | Dataset governance, licence VERIFY list, split design |
| `NEEDS_SURYA.md` | Review requests for shared contracts and modules |
| `CONSENT_FORM.md` | In-house recording biometric consent form |
| `RECORDING_PROTOCOL.md` | In-house recording session protocol |
| `baseline_smoke_result.md` | B1 pipeline smoke run record + B0 status |
| `results_log.md` | Training and evaluation run ledger |
| `test_output.txt` | Raw pytest output (39/39 passing in 6.48 s) |

---

## 5. Pretrained Checkpoint (B0 Baseline Model)

**Model:** `speechbrain/spkrec-ecapa-voxceleb` (C=1024 ECAPA-TDNN)  
**Parameters:** ~20.8M  
**Embedding dim:** 192  
**Cache path:** `~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/`  
**Status:** Fully staged locally — no download needed  
**Loading:** Direct `torch.load(embedding_model.ckpt)` (Windows-safe, avoids symlink error WinError 1314)  

---

## 6. B1 Smoke Training Runs

### Active Clean Run: `20261112-0758_ca0744b_6596ea`
- **Date/Time:** 2026-11-12 13:32 IST
- **Status:** **SMOKE OK (Clean Working Tree)**
- **Git SHA:** `ca0744b` (clean HEAD, verified with 0-byte `git_diff.txt`)
- **Config hash:** `6596ea`
- **Seed:** 1986
- **Data:** Synthetic 20-speaker dataset, 420 train / 80 val utterances (3s each, 16kHz)
- **Manifest SHA-256:** `27189a59bc12659472c9ce9ac3fbcbc612991608269c1ee9724ff43de00f63e5`

**Verified Checkpoint Hashes:**
| File | SHA-256 |
|---|---|
| `checkpoint_best.pt` (epoch 1) | `de1e6523e4f1a4c923f9c0ee7774780d3183db5248133c4da923310b79d77cb0` |
| `checkpoint_last.pt` (epoch 2) | `d03262b4af13d7ec6b422344515a9a57e0c8507ac6818cb097756a72af0e37cc` |

**Training metrics (logged in `metrics.jsonl`):**
| epoch | global_step | loss | margin | lr_backbone | lr_head | grad_norm |
|---|---|---|---|---|---|---|
| 1 | 10 | 9.2086 | 0.200 | 2.00e-05 | 2.00e-04 | 6.119 |
| 2 | 20 | 7.7624 | 0.200 | 4.00e-05 | 4.00e-04 | 9.314 |

> **Step Count Analysis:** With 420 training clips and `batch_size = 4`, there are 105 micro-batches per epoch. With `grad_accum_steps = 8`, an optimizer update occurs every 8 micro-batches, resulting in 13 optimizer steps per epoch (26 total optimizer steps across the 2 epochs, with 1 unaccumulated micro-batch per epoch). With `log_every_n_steps = 10`, metrics were emitted only at step 10 and step 20. The two entries in `metrics.jsonl` represent these 10-step logging events.

> **Class-Centre Initialisation Timing:** Initialising reference class centres across 420 clips took ~116 seconds (~2 minutes) on CPU. On the full VoxCeleb1 dev dataset (1,211 speakers, ~145k utterances), CPU initialisation would take many hours. On the HPC, V100 GPU speed will reduce this significantly; throughput will be profiled on V100 before deciding whether to cap reference utterances per speaker (e.g. 5–10 clips/speaker).

### Superseded Run: `20261111-1918_9eef8dc_6596ea`
- Executed on commit `9eef8dc` with working-tree squeeze fix. Preserved for audit trail and marked `SUPERSEDED BY 20261112-0758_ca0744b_6596ea`.

---

## 7. Test Results Summary

**Total:** 39/39 PASSED in 6.48 seconds (verified from `test_output.txt`)  
**Command:** `cd d:\AAA_Layer1\aaa_engine && python -m pytest ../tests/test_layer1.py ../tests/test_sv_pipeline.py -v`  

- **34 Pre-existing Layer 1 Unit Tests** (`tests/test_layer1.py`):  
  Validates baseline speaker verifier functionality: cosine similarity, decision threshold logic (accept/reject at 0.75), custom threshold support, 192-dim embedding properties, database storage and retrieval with SHA-256 hashes, and duration/channel audio validation. All 34 PASSED.

- **5 New SV Pipeline Regression Tests** (`tests/test_sv_pipeline.py`):  
  1. `test_resume_continuity`: Simulates 3 training steps with a lightweight toy embedder+classifier model (no speechbrain/network dependency), saves checkpoint containing step, epoch, model/optimizer states, and torch/numpy/python RNG states, reloads into fresh instances, and asserts exact match of step, epoch, byte-for-byte parameter weights, and subsequent random sample continuity for torch, numpy, and python RNG.
  2. `test_preprocessing_parity`: Generates synthetic waveform, writes to WAV via soundfile, runs `sv_eval.preprocess_audio()` twice to verify determinism; tests stereo-to-mono downmix as channel mean; tests 8 kHz to 16 kHz resample; asserts float32 dtype. Invariants tested: no silence trimming and no repeat-padding.
  3. `test_shuffled_label_sanity`: Generates synthetic multi-class batch with randomly permuted (shuffled) speaker labels, trains a toy model for 20 steps, and asserts loss fails to converge (loss remains > 0.5), verifying that random labels do not lead to learning.
  4. `test_embedding_norm`: Loads the real pretrained ECAPA-TDNN from HuggingFace cache (marked `skipif` if model cannot be loaded), encodes a 3.0 s synthetic waveform, extracts the 192-dim embedding, and asserts Euclidean L2 norm is within 0.05 of 1.0.
  5. `test_augmentation_reproducibility`: Instantiates two separate `Augmenter` objects with the same integer seed (1986), passes identical synthetic audio waveforms through both, and asserts output waveforms are strictly bit-for-bit identical (`torch.allclose` with `atol=0.0`).

---

## 8. Bugs Found and Fixed

| Bug | Root Cause | Fix |
|---|---|---|
| `librosa.effects` deadlocks on Windows | Lazy submodule import hangs in sandboxed env with no audio backend | `conftest.py` stubs module at session start |
| `torchaudio.load/save` requires torchcodec | torchaudio 2.10 on Windows needs optional `torchcodec` package | `soundfile` as primary I/O backend in all scripts |
| `torch.load` default changed to `weights_only=True` in PyTorch 2.6+ | NumPy arrays in checkpoint fail safe unpickling | Added `weights_only=False` for ckpt with numpy RNG state |
| ECAPA-TDNN output `(B, 1, 192)` → AAM-Softmax crash | SpeechBrain returns 3D tensor; `one_hot.scatter_` expects 2D | Squeeze to `(B, 192)` before classifier |
| Windows symlink privilege error in `SpeakerRecognition.from_hparams()` | Windows requires elevated rights for symlinks | Direct `.ckpt` load via `torch.load()` |
| `datetime.utcnow()` deprecation warning | Python 3.12 deprecation | Switched to `datetime.now(timezone.utc)` |

---

## 9. Next Steps (Closing Day 1)

1. **Phase 2 — Real Manifests & Leakage Check:**
   - Confirm VoxCeleb1 dev path.
   - Run `manifest_builder.py` on VoxCeleb1 dev metadata (1,211 dev speakers, asserting zero overlap with Vox1-O test speakers).
   - Run `check_leakage.py` across splits; hash manifests into `manifest_hashes.txt`.
2. **Phase 3 — Real B0 Baseline Evaluation:**
   - Run `sv_eval.py` on VoxCeleb1-O cleaned trial list (`veri_test2.txt`).
   - Record measured EER and FRR@FAR1% in `results_log.md` and `b0_baseline_smoke_result.md`.
3. **Phase 4 — HPC Deployment & B1 Training Launch:**
   - Dry run, kill-and-resume test, and V100 throughput profiling.
   - Launch full B1 fine-tuning job on HPC cluster.

---

*Report author: Navadeep | Date: 2026-11-12*  
*All metrics in this report reflect actual measurements. No values are estimated or invented.*
