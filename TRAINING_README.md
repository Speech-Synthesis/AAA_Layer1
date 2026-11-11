# B1 Training — README

> [!NOTE]
> [FINE-TUNED] ECAPA-TDNN fine-tune with AAM-Softmax head, robust augmentation, duration crops.

## Quick Start

```powershell
# From project root (d:\AAA_Layer1)
cd d:\AAA_Layer1

# Local CPU smoke run (synthetic dataset, 2 epochs)
python -m aaa_engine.training.train_b1 --config configs/train_local_smoke.yaml

# Resume (automatic — just re-run the same command)
python -m aaa_engine.training.train_b1 --config configs/train_local_smoke.yaml

# HPC launch (fill placeholders in configs/train_hpc.yaml first)
qsub train_job_b1.sh
```

## Files

| File | Purpose |
|---|---|
| `aaa_engine/training/train_b1.py` | Main training script |
| `aaa_engine/training/augment.py` | Seeded augmentation (MUSAN, RIR, codec, resample, bandlimit) |
| `aaa_engine/training/sv_eval.py` | B0 baseline evaluation (EER, FRR@FAR1%, bootstrap CI) |
| `configs/train_local_smoke.yaml` | Local CPU smoke config (tiny dataset, 2 epochs) |
| `configs/train_hpc.yaml` | HPC single-GPU config (fill placeholders) |
| `configs/eval_config.yaml` | Evaluation config |
| `train_job_b1.sh` | PBS job script for HPC |
| `tests/test_sv_pipeline.py` | Pipeline tests (resume, parity, sanity, norm, augment-repro) |
| `tests/conftest.py` | pytest fixtures (librosa.effects prewarm, markers) |

## Environment

| Item | Value |
|---|---|
| Python | 3.12.4 |
| PyTorch | 2.10.0+cpu |
| SpeechBrain | 1.1.1 |
| librosa | 1.0.0 |
| numpy | 2.3.5 |
| soundfile | installed |
| ffmpeg | NOT on PATH (codec augmentation disabled locally) |
| GPU | None (CPU-only dev machine) |
| HPC GPU | V100 (planned) |

## Windows-Specific Notes

- **`torchaudio` audio I/O**: torchaudio 2.10 requires `torchcodec` for `.load()` and `.save()`.  
  All audio I/O uses **soundfile** as primary backend (`_load_audio()` helper, `sv_eval.preprocess_audio()`).
- **SpeechBrain symlinks**: `SpeakerRecognition.from_hparams()` creates symlinks which require  
  elevated privilege on Windows. `load_pretrained_ecapa()` uses direct `.ckpt` load via `torch.load()`  
  when `pretrained_ckpt_direct` is set in config.
- **librosa.effects hang**: `librosa.effects` lazy-imports hang in sandboxed Windows environments.  
  Fixed via `conftest.py` which injects a stub module at session start.

## Training Config

```yaml
# Local smoke (CPU, synthetic 20-speaker dataset)
batch_size: 4           # micro-batch
grad_accum_steps: 8     # effective batch = 32
number_of_epochs: 2
crop_lengths: [1, 2]    # random crop length in seconds
backbone_lr: 0.0001     # frozen epoch 1, unfrozen epoch 2+
head_lr: 0.001
warmup_steps: 50
aam_margin: 0.2         # ramped 0 → 0.2 over first 20% of steps
aam_scale: 30
seed: 1986
```

## Run ID Format

```
YYYYMMDD-HHMM_<gitsha7>_<cfghash6>
```

Each run creates: `runs/<run_id>/`
- `checkpoint_last.pt` — latest checkpoint (atomic write)
- `checkpoint_best.pt` — best validation EER checkpoint
- `metrics.jsonl` — per-step metrics
- `metrics.csv` — same in CSV format
- `run_config.yaml` — exact config used

## Resume

Resume is automatic — just re-run the same command. The script detects
`runs/<run_id>/checkpoint_last.pt` and continues from the last step.

> [!IMPORTANT]
> Run ID is derived from the git SHA and config hash, so resume works only if
> you use the same config file. If you change the config, a new run ID is created.

## Tests

```powershell
cd d:\AAA_Layer1\aaa_engine
python -m pytest ../tests/test_layer1.py ../tests/test_sv_pipeline.py -v
```

**Result (2026-11-12):** 39/39 PASSED in 6.25s — see `docs/evidence/day1/test_output.txt`

| Test | Description | Status |
|---|---|---|
| `test_resume_continuity` | Checkpoint save/load/RNG continuity | ✅ PASS |
| `test_preprocessing_parity` | Training vs inference path parity | ✅ PASS |
| `test_shuffled_label_sanity` | Shuffled labels must not learn | ✅ PASS |
| `test_embedding_norm` | Embedding L2 norm ≈ 1.0 | ✅ PASS |
| `test_augmentation_reproducibility` | Seeded augmentation reproducibility | ✅ PASS |

## Augmentation

Seeded, reproducible augmentation via `Augmenter` class:

| Type | Requires | Status (local) |
|---|---|---|
| Noise injection | MUSAN corpus | Disabled (no corpus) |
| Reverb | RIRS_NOISES corpus | Disabled (no corpus) |
| Codec chain | ffmpeg | Disabled (ffmpeg not on PATH) |
| Resample chain | None | ✅ Active |
| Band-limit | None | ✅ Active |

On HPC: set `musan_dir`, `rirs_dir` in `configs/train_hpc.yaml` and ensure ffmpeg is installed.

## HPC Launch

1. Fill placeholders in `configs/train_hpc.yaml` and `train_job_b1.sh`:
   - `PLACEHOLDER_HPC_ACCOUNT` → your HPC account
   - `PLACEHOLDER_HPC_PARTITION` → GPU partition name
   - `PLACEHOLDER_DATA_ROOT` → path to dataset on HPC
   - etc.
2. Record the manifest hash: `python -c "import hashlib; print(hashlib.sha256(open('aaa_engine/training/manifests/train.csv','rb').read()).hexdigest())"`
3. Run leakage check: `python -m aaa_engine.training.check_leakage`
4. Submit: `qsub train_job_b1.sh`

## Metrics

> [!IMPORTANT]
> No metrics are reported here. Results are TBD until a full HPC run completes
> with recorded run_id, config_hash, manifest_hash, and checkpoint_hash.
> See `docs/evidence/day1/results_log.md` for the results tracking table.

---

*Last updated: 2026-11-12 | Branch: feat/navadeep-aaa-sv*
