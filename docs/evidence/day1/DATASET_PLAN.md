# DATASET_PLAN.md
## AAA-SV Dataset Governance Plan
**Date:** 2026-11-11 | **Sprint:** AAA Engine Day 1 | **Status:** DRAFT — licence fields marked VERIFY

---

## 1. Dataset Registry

### 1.1 VoxCeleb1 Dev Set

| Field | Value |
|---|---|
| **Role** | Train (B1 fine-tune) |
| **Exact release** | VoxCeleb1 v1 (2017) — dev portion |
| **Source URL** | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html |
| **Licence** | CC BY 4.0 (for research) — **VERIFY** at https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html (check the data usage agreement; originally scraped from YouTube, some jurisdictions may require additional clearance) |
| **Approximate size** | ~40 GB (145,265 utterances, 1,251 speakers, avg ~8 s each) |
| **Storage location** | HPC only — do not copy to dev laptop |
| **Speaker count** | 1,211 dev speakers (official dev split of the 1,251 total VoxCeleb1 speakers; 40 test speakers are held out in the Vox1-O test partition) |
| **SHA256 of manifest** | TBD — record after split generation |
| **Notes** | Speaker IDs are de-identified VGGFace IDs. Audio is from "in the wild" YouTube videos. No consent from speakers. Use only for research per licence. **Do not use for cal or test-AAA.** |

### 1.2 VoxCeleb1-O Cleaned Trial List

| Field | Value |
|---|---|
| **Role** | test-std (standard benchmark) |
| **Exact release** | VoxCeleb1 "cleaned" trial list (`veri_test2.txt`, Chung et al. 2020 — corrected labels) |
| **Source URL** | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt |
| **Licence** | Same as VoxCeleb1 — **VERIFY** |
| **Approximate size** | Cleaned trial list (trial count taken directly from file; distinct from the 37,720 pairs of the original uncleaned veri_test.txt); audio from VoxCeleb1 test set (40 speakers) |
| **Storage location** | HPC only |
| **Notes** | Published EER on this list: **~0.87%** (pretrained ECAPA-TDNN per reference doc). Test data is frozen after Day 3 freeze. Dev portion of VoxCeleb (1,211 speakers) must never appear in this list. |

### 1.3 MUSAN

| Field | Value |
|---|---|
| **Role** | Augmentation (noise injection) |
| **Exact release** | MUSAN v1.0 (2015, Snyder et al.) |
| **Source URL** | https://www.openslr.org/17/ |
| **Licence** | CC BY 4.0 — **VERIFY** at https://www.openslr.org/17/ (check individual noise source licences within MUSAN) |
| **Approximate size** | ~11 GB (60 h noise, 42 h music, 60 h speech) |
| **Storage location** | HPC; copy of noise-only subset to local for smoke test (small) |
| **Notes** | Only noise and music subsets used for augmentation; speech subset excluded. |

### 1.4 RIRS_NOISES (Room Impulse Responses)

| Field | Value |
|---|---|
| **Role** | Augmentation (reverb simulation) |
| **Exact release** | RIRS_NOISES (Ko et al. 2017) |
| **Source URL** | https://www.openslr.org/28/ |
| **Licence** | Apache 2.0 — **VERIFY** at https://www.openslr.org/28/ |
| **Approximate size** | ~36 GB (simulated + real RIRs) |
| **Storage location** | HPC only |
| **Notes** | Simulated RIRs used for reverb augmentation during training. |

### 1.5 Common Voice English

| Field | Value |
|---|---|
| **Role** | val (set A), cal (set B), test-AAA (set C) — see split design |
| **Exact release** | Mozilla Common Voice English — **specify exact corpus version before use** (e.g., CV 15.0, 2024-06-23) — **VERIFY version** |
| **Source URL** | https://commonvoice.mozilla.org/en/datasets |
| **Licence** | CC0 (audio) + CC BY 4.0 (metadata) — **VERIFY** per version; contributor agreement requires attribution; not all clips have explicit consent for biometric use |
| **Approximate size** | Varies by version; CV 15.0 English ~90 GB validated |
| **Storage location** | HPC; filtered subset to local for smoke test |
| **Notes** | Speaker label is `client_id` — this is a **noisy label** (same person may have multiple client_ids across sessions). Apply minimum 6-clips-per-client filter. Must confirm that use for speaker verification evaluation is covered by the licence and contributor consent. **VERIFY with legal/Surya.** |

### 1.6 In-House Recordings

| Field | Value |
|---|---|
| **Role** | cal (half 1), test-AAA (half 2) |
| **Exact release** | Internal — AAA Engine sprint recordings (Nov 2026) |
| **Source URL** | N/A (internal) |
| **Licence** | Internal — governed by signed consent forms (see `CONSENT_FORM.md`) |
| **Approximate size** | TBD — target ≥20 speakers × 2 sessions × 2 devices |
| **Storage location** | Restricted network storage (NOT Git) + HPC if needed |
| **Notes** | Consent forms must be collected before any audio is used. Consent IDs must map to pseudonymous speaker IDs. Key table stored separately from audio. |

---

## 2. DATASETS.md Rows (Draft)

The following rows should be inserted into the shared `DATASETS.md` file (pending Surya's confirmation of schema):

```markdown
| voxceleb1-dev | VoxCeleb1 Dev | CC BY 4.0 [VERIFY] | 40 GB | HPC | train |
| voxceleb1-o-clean | VoxCeleb1-O Cleaned Trial List | CC BY 4.0 [VERIFY] | <1 GB trial file | HPC | test-std |
| musan | MUSAN v1.0 | CC BY 4.0 [VERIFY] | 11 GB | HPC | augmentation |
| rirs-noises | RIRS_NOISES | Apache 2.0 [VERIFY] | 36 GB | HPC | augmentation |
| common-voice-en | Mozilla Common Voice English vXX.X [VERIFY VERSION] | CC0+CC BY 4.0 [VERIFY biometric consent] | ~90 GB | HPC | val/cal/test-AAA |
| inhouse-2026-11 | In-House Recordings Nov 2026 | Internal consent (see CONSENT_FORM.md) | TBD | Restricted storage | cal/test-AAA |
```

### Licence Verification Checklist

The following pages must be checked and confirmed before any data is used:

1. **VoxCeleb1 usage agreement** — https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
   - Confirm: CC BY 4.0 for research; no commercial use; attribution required
   - Check: whether scraped YouTube audio requires additional clearance in India/EU
2. **MUSAN individual licence** — https://www.openslr.org/17/ → check README for per-track source licences
3. **RIRS_NOISES licence** — https://www.openslr.org/28/ → check for Apache 2.0 confirmation
4. **Common Voice version** — https://commonvoice.mozilla.org/en/datasets
   - Confirm exact corpus version and datestamp being used
   - Check: whether CC0 audio licence covers biometric processing (speaker verification)
   - Check: whether contributor agreement explicitly permits speaker ID extraction

---

## 3. Speaker-Disjoint Split Design

### 3.1 Principles

- Speakers are assigned to exactly one primary partition. No speaker appears in two partitions.
- Augmented copies follow their source file's partition (never cross boundaries).
- De-duplication by SHA256 is applied across all partitions before training.
- VoxCeleb dev speakers must never appear in cal or test-AAA.
- Common Voice `client_id` is a noisy speaker label: apply the ≥6 clips filter, then split by `client_id` (not by clip).
- In-house recordings are split by physical speaker ID (from consent form), not by file or session.

### 3.2 Partition Definitions

| Partition | Source | Purpose | Speaker count |
|---|---|---|---|
| **train** | VoxCeleb1 dev (1,211 dev speakers) | B1 fine-tune | 1,211 |
| **val** | Common Voice English — set A (speaker subset) | Hyperparameter selection (EER, FRR@FAR1%) | ≥50 CV speakers [VERIFY] |
| **cal** | Common Voice English — set B + In-house half 1 | Calibrator fitting (AS-norm cohort + logistic regression calibrator; per AAA-SV.md §7, no PLDA is used) | TBD |
| **test-std** | VoxCeleb1-O cleaned trial list | Standard benchmark (B0, B1) | Test set from VoxCeleb1 test partition |
| **test-AAA** | In-house half 2 + Common Voice set C | AAA-specific evaluation | TBD |

### 3.3 Common Voice Split Procedure

```
1. Download Common Voice English (exact version TBD — confirm with Surya).
2. Filter to clips with client_id having ≥6 validated clips.
3. Remove any client_id that appears in VoxCeleb1 (by cross-matching speaker name 
   metadata if available — otherwise treat as separate by convention).
4. Randomly assign client_ids to sets A, B, C using seeded split 
   (seed=42, ratio A:B:C = 60:20:20) with no client_id overlap.
5. All clips of a client_id go to its assigned set.
6. Augmented copies of any clip go to the same set.
7. SHA256 deduplicate across all sets.
```

### 3.4 In-House Recording Split Procedure

```
1. Collect signed consent forms → assign pseudonymous speaker IDs.
2. Split speakers randomly into half-1 (cal) and half-2 (test-AAA) 
   BEFORE recording sessions start (pre-registration).
3. All files for a speaker go to their assigned half, regardless of session or device.
4. Session IDs track recording sessions; device field tracks the hardware used.
5. SHA256 deduplicate within each half.
```

### 3.5 SHA256 Deduplication

```python
# Pseudocode — real implementation in manifest_builder.py
seen_hashes = set()
for record in all_records:
    h = sha256(open(record.path, 'rb').read()).hexdigest()
    if h in seen_hashes:
        log_duplicate(record)
        continue
    seen_hashes.add(h)
    emit_record(record, sha256=h)
```

---

## 4. Manifest Builder Design

### 4.1 Schema (SHARED.md section 4 — interpreted from brief)

One JSONL file; one JSON object per line. Each record:

```json
{
  "id": "<unique string: source_spk_utterance>",
  "split": "train|val|cal|test-std|test-AAA",
  "audio_path": "<path relative to manifest root>",
  "sha256": "<hex string>",
  "duration_s": 3.14,
  "sample_rate": 16000,
  "channels": 1,
  "speaker_id": "<pseudonymous ID>",
  "source_dataset": "voxceleb1|musan|rirs|commonvoice|inhouse",
  "licence": "CC BY 4.0",
  "consent_id": "<UUID or null for non-consent datasets>",
  "session_id": "<recording session UUID or null>",
  "device": "<device descriptor or null>",
  "is_augmented": false,
  "augmentation": null
}
```

### 4.2 Exclusion Rules

Records are excluded (and counted in an exclusion report) if:
- `licence` is null or empty
- `consent_id` is null AND `source_dataset == "inhouse"`
- `duration_s < 1.0` or `duration_s > 10.0` (after any trimming)
- `sha256` is a duplicate of a previously seen record (first occurrence kept)
- `sample_rate != 16000` (after resampling; raw SR is recorded before resample)

### 4.3 Exclusion Report Format

```
manifest_exclusions_<run_id>.txt
  TOTAL_RECORDS_SCANNED: N
  EXCLUDED_NO_LICENCE: N
  EXCLUDED_NO_CONSENT (inhouse only): N
  EXCLUDED_DURATION_OOB: N
  EXCLUDED_DUPLICATE_SHA256: N
  TOTAL_INCLUDED: N
```

---

## 5. Leakage Checker Design

The leakage checker is a standalone script (`check_leakage.py`) that:

### 5.1 Speaker Overlap Check

```
For each pair of partitions (train, val, cal, test-std, test-AAA):
  overlap = set(speakers_in_A) ∩ set(speakers_in_B)
  if overlap:
    FAIL: list overlapping speaker IDs
  else:
    PASS
```

### 5.2 SHA256 Overlap Check

```
Collect all sha256 values across all partitions with their partition label.
For each sha256 that appears in >1 partition:
  FAIL: report sha256, file path, conflicting partitions
```

### 5.3 VoxCeleb Dev Isolation Check

```
speakers_in_voxceleb_dev = set of VoxCeleb1 dev speaker IDs (1,251 IDs)
for partition in [cal, test-AAA]:
  overlap = speakers_in_voxceleb_dev ∩ set(speakers_in_partition)
  if overlap:
    FAIL: "VoxCeleb dev speaker found in {partition}"
  else:
    PASS
```

### 5.4 Test Freeze Check

```
After Day 3 freeze:
  record sha256 of test-std manifest and test-AAA manifest.
  Any subsequent run that modifies these manifests must fail CI.
```

### 5.5 Output

```
leakage_report_<run_id>.json
{
  "run_id": "...",
  "timestamp": "...",
  "speaker_overlap": { "train_val": "PASS", "train_cal": "PASS", ... },
  "sha256_overlap": { "result": "PASS|FAIL", "conflicts": [] },
  "voxceleb_dev_in_cal": "PASS",
  "voxceleb_dev_in_test_aaa": "PASS",
  "overall": "PASS|FAIL"
}
```

---

## 6. Licence Pages to Verify

The following pages must be visited and their terms confirmed by Navadeep before the dataset governance plan is finalised:

| # | Dataset | URL to check | What to confirm |
|---|---|---|---|
| 1 | VoxCeleb1 usage agreement | https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html | Research-only CC BY 4.0; attribution; YouTube source implications |
| 2 | MUSAN per-track sources | https://www.openslr.org/17/ → README.txt | Per-track source licences for noise clips |
| 3 | RIRS_NOISES | https://www.openslr.org/28/ → README.txt | Confirm Apache 2.0 |
| 4 | Common Voice English licence | https://commonvoice.mozilla.org/en/datasets | Confirm CC0 audio; check biometric processing clause in contributor agreement |
| 5 | Common Voice exact version | https://commonvoice.mozilla.org/en/datasets | Record version number, datestamp, and validate checksum |
| 6 | In-house consent | `CONSENT_FORM.md` (this repo) | Confirm scope fields match; get Surya's sign-off on legal review |

---

*Status: DRAFT. No datasets have been downloaded. No data processing has occurred.*
*All licence fields marked VERIFY must be confirmed before data use.*
