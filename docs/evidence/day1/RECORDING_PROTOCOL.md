# RECORDING_PROTOCOL.md
## In-House Voice Recording Protocol
### AAA Engine — Speaker Verification & ASR Evaluation

**Version:** 1.0 | **Date:** 2026-11-11 | **Target:** ≥20 speakers

---

## 1. Overview

This protocol governs all in-house voice recordings made for the AAA Engine sprint.
Recordings serve two purposes:
- **cal (calibration):** half of speakers → AS-norm cohort / calibrator fitting
- **test-AAA:** other half → AAA-specific evaluation

The partition (cal / test-AAA) is assigned per speaker before recording begins. This prevents any calibration-influenced selection of test speakers.

---

## 2. Speaker Targets

| Requirement | Target |
|---|---|
| Total speakers | **≥ 20** (aim for 24 to allow for dropouts) |
| Sessions per speaker | **≥ 2** (at least 1 day apart if possible; minimum 1 session) |
| Devices per speaker | **≥ 2**: at least one phone + one laptop/desktop per speaker |
| Minimum clips per speaker | **6** (across all sessions and devices) |
| Speaker pre-assignment | Randomly assigned to cal (half 1) or test-AAA (half 2) **before** recording |

---

## 3. Equipment Checklist

Before each session, verify:

- [ ] Recording app / script ready on device
- [ ] Device clock synced (for session_id timestamp)
- [ ] Background noise level acceptable (< 40 dB SPL for "quiet" sessions)
- [ ] Device model and OS version noted in metadata CSV
- [ ] Consent form collected and consent_id written in CSV

**Device descriptors to use in metadata:**

| Device type | Descriptor format example |
|---|---|
| Smartphone | `phone_samsung_galaxys23` |
| Laptop internal mic | `laptop_dell_xps15_internal` |
| Laptop external USB mic | `laptop_external_usb_mic` |
| Desktop mic | `desktop_external_mic` |

---

## 4. Recording Content

### 4.1 Challenge Phrases (read aloud)

Participants read the following fixed phrases in a random order. The operator shows each phrase on a screen or printed card.

```
1. "The quick brown fox jumps over the lazy dog."
2. "My voice is my passport, verify me."
3. "Open sesame and authenticate this session."
4. "Authentication requires both presence and identity."
5. "Zero one one zero one — speaker verification active."
6. "The session begins now. Please record this phrase."
7. "Security is not a product but a process."
8. "Please verify that this is my genuine voice."
9. "The cat sat on the mat by the door."
10. "Five seconds of clear speech is all that is needed."
```

### 4.2 Free Speech

After the challenge phrases, the participant speaks freely about a neutral topic (the weather, their commute, a hobby) for **60–120 seconds**.

### 4.3 Noisy Condition

One session (or device in a session) should be recorded in a **lightly noisy** environment (e.g., a room with background conversation or traffic audible). Record the condition as `quiet` or `noisy` in the metadata CSV.

### 4.4 Duration Targets

| Clip type | Target duration |
|---|---|
| Each challenge phrase | 3–8 seconds |
| Free speech | 60–120 seconds (split into ≤10 s segments at silence) |
| Minimum per speaker per session | 60 s usable audio |

---

## 5. Session ID and Filename Convention

### 5.1 Session ID

Format: `SESSION-{YYYYMMDD}-{SPEAKERID}-{SEQ}`

Example: `SESSION-20261112-SPK-20261112-0001-01`

The sequence suffix (`-01`, `-02`) distinguishes multiple sessions of the same speaker on the same day.

### 5.2 Filename Convention

```
{session_id}_{device}_{phrase_index}.wav
```

Examples:
```
SESSION-20261112-SPK-20261112-0001-01_phone_samsung_galaxys23_phrase01.wav
SESSION-20261112-SPK-20261112-0001-01_phone_samsung_galaxys23_free01.wav
```

**Rules:**
- No real names, emails, or personal identifiers in filenames.
- Use only the pseudonymous `SPK-YYYYMMDD-NNNN` identifier.
- All filenames must be lowercase with underscores (no spaces).

---

## 6. Metadata CSV Format

The operator fills in a CSV (`sessions_metadata.csv`) with one row per audio file:

```csv
consent_id,speaker_id,session_id,device,condition,filename,original_sample_rate,duration_s,scope,notes
CONSENT-20261112-0001,SPK-20261112-0001,SESSION-20261112-SPK-20261112-0001-01,phone_samsung_galaxys23,quiet,SESSION-20261112-..._phrase01.wav,44100,4.2,"sv_enroll,sv_eval",
```

| Column | Required | Notes |
|---|---|---|
| `consent_id` | Yes | From signed consent form |
| `speaker_id` | Yes | Pseudonymous `SPK-YYYYMMDD-NNNN` |
| `session_id` | Yes | As defined above |
| `device` | Yes | As defined above |
| `condition` | Yes | `quiet` or `noisy` |
| `filename` | Yes | Relative to recording root folder |
| `original_sample_rate` | Yes | As recorded (before any resampling) |
| `duration_s` | Yes | In seconds |
| `scope` | Yes | Comma-separated scope codes from consent form |
| `notes` | No | Any relevant observation (e.g., participant coughed during recording) |

---

## 7. Operator Checklist (per session)

**Before recording:**
- [ ] Consent form signed and consent_id assigned
- [ ] Speaker pre-partition (cal / test-AAA) confirmed in partition log
- [ ] Device metadata noted in CSV
- [ ] Recording script/app started and tested

**During recording:**
- [ ] Participant reads all challenge phrases without rushing
- [ ] Free speech recorded (quiet condition)
- [ ] At least one noisy-condition recording made (if second device available)
- [ ] No real names spoken aloud during recording

**After recording:**
- [ ] All files transferred to secure storage (encrypted drive or access-restricted network share)
- [ ] Metadata CSV updated with all files
- [ ] SHA256 checksums computed for all files and noted in CSV or separate checksum file
- [ ] Consent form stored in locked folder (not co-located with audio)

---

## 8. Manifest Builder Design

A script (`build_inhouse_manifest.py`) produces the JSONL manifest from the metadata CSV and audio folder.

### Input

```
recordings/
  SESSION-20261112-SPK-20261112-0001-01_phone_..._phrase01.wav
  ...
sessions_metadata.csv
```

### Processing Steps

```
1. Read sessions_metadata.csv.
2. For each row:
   a. Check consent_id is not null/empty → reject and count if missing.
   b. Check scope is not empty → reject if no scope selected.
   c. Load audio file → measure actual duration_s and sample_rate.
   d. Compute sha256 of raw file bytes.
   e. Check duration_s in [1.0, 10.0] → exclude OOB clips (count separately).
   f. Check sha256 not already seen (deduplicate).
   g. Emit JSONL record.
3. Write manifest to output_manifest.jsonl.
4. Write exclusion_report.txt with counts.
```

### Output JSONL Record

```json
{
  "id": "SPK-20261112-0001_SESSION-20261112-..._phrase01",
  "split": "cal",
  "audio_path": "recordings/SESSION-20261112-..._phrase01.wav",
  "sha256": "<hex>",
  "duration_s": 4.2,
  "sample_rate": 44100,
  "channels": 1,
  "speaker_id": "SPK-20261112-0001",
  "source_dataset": "inhouse",
  "licence": "internal_consent",
  "consent_id": "CONSENT-20261112-0001",
  "session_id": "SESSION-20261112-SPK-20261112-0001-01",
  "device": "phone_samsung_galaxys23",
  "condition": "quiet",
  "scope": ["sv_enroll", "sv_eval"],
  "is_augmented": false,
  "augmentation": null
}
```

### Exclusion Report

```
TOTAL_ROWS_IN_CSV: N
EXCLUDED_NO_CONSENT_ID: N         ← rows where consent_id is null
EXCLUDED_NO_SCOPE: N              ← rows where scope is empty
EXCLUDED_DURATION_OOB: N          ← clips outside [1.0, 10.0] s
EXCLUDED_DUPLICATE_SHA256: N      ← exact duplicate files
TOTAL_INCLUDED: N
```

---

## 9. Storage and Security

| Item | Storage | Access |
|---|---|---|
| Audio files | Encrypted network share OR encrypted external drive | AAA Engine team only — not Git |
| Metadata CSV (`sessions_metadata.csv`) | Same encrypted location | AAA Engine team only — not Git |
| Speaker key table (`speaker_key_table.csv`: consent_id ↔ real name) | **Separate** encrypted drive/share | Principal investigators only — not accessible to other team members |
| Signed consent forms | Locked folder (physical or encrypted PDF) | Principal investigators only |
| JSONL manifest (pseudonymous only) | Can be committed to Git on feature branch | No real names; consent_ids are pseudonymous |

**The `speaker_key_table.csv` must NEVER be committed to Git or stored with the audio files.**

If a participant withdraws:
1. Delete all their audio files from storage.
2. Remove their rows from `sessions_metadata.csv` and re-run the manifest builder.
3. Mark their consent_id as WITHDRAWN in the key table (do not delete the key table row — needed for audit).
4. If their data has already been included in a released evaluation result, note the withdrawal in a data-usage log.
