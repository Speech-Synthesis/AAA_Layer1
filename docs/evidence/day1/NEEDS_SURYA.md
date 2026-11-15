# NEEDS_SURYA.md
## Items Requiring Surya's Review / Decision
### AAA-SV Sprint — Nov 2026

**Author:** Navadeep | **Created:** 2026-11-11 | **Updated:** 2026-11-12

---

This file lists items that touch shared contracts, other members' code, or require Surya's
input before Navadeep can proceed. Per scope rules, no changes are made to shared code
without joint review first.

---

## 1. Contract Confirmation (BLOCKING)

**Status:** BLOCKING — cannot finalise inference adapter or deployment without this.

The inference adapter for AAA-SV exposes these functions (draft in `docs/day1/inference_adapter_draft.md`):

```
embed(audio_bytes) → np.ndarray[192, float32], L2-normalised
enroll(user_id, audio_bytes) → candidate_embedding: np.ndarray[192] (never touches DB)
verify(user_id, audio_bytes, stored_embedding) → score: float, decision: str
maybe_update(user_id, score, live_emb) → updated_embedding | None
```

**Questions for Surya:**
- Do these match the shared contract defined in SHARED.md sections 8, 21, 22?
- Is the error code set from SHARED.md section 2 finalised? (Current codes in `speaker_verifier.py`: `REJECT` with `layer_blocked=1`, no numeric error codes.)
- Should `embed()` accept raw bytes or a file path? The brief says bytes; confirm.

---

## 2. Deprecated SpeechBrain Import (SHARED MODULE)

**File:** `aaa_engine/layer1/speaker_verifier.py` line 19
**Current:** `from speechbrain.pretrained import SpeakerRecognition`
**Required:** `from speechbrain.inference.speaker import SpeakerRecognition`

This module is shared (used by the API). Navadeep will not change it without Surya's agreement.
The fix is a one-line change. SpeechBrain 1.1.1 issues a deprecation warning but still works.

**Decision needed:** Surya confirms the change is safe → Navadeep makes it on the feature branch.

---

## 3. Preprocessing Mismatch (SHARED CONTRACT QUESTION)

**Current state:**
- Inference path (`preprocess_audio()` in `speaker_verifier.py`): applies `librosa.effects.trim(top_db=25)` after resampling.
- Training path (`audio_pipeline()` in `train_ecapa.py`): no trimming — just `torchaudio.load` with start/stop offsets.

**Impact:** The model is fine-tuned on untrimmed audio but evaluated on trimmed audio. This introduces a preprocessing mismatch that may inflate or deflate EER.

**Options:**
1. Remove `librosa.trim` from inference (simpler — matches training).
2. Add `librosa.trim` to training audio pipeline.
3. Keep mismatch and document it as a known limitation.

**Decision needed:** Surya or Navadeep decides which option to adopt. If option 1 or 2 touches shared code (option 1 → `speaker_verifier.py`), Surya must approve.

---

## 4. `MIN_DURATION` Mismatch

**File:** `aaa_engine/layer1/speaker_verifier.py` line 27
**Current:** `MIN_DURATION = 1.5`
**Spec (brief):** Shared audio contract: 1.0 s minimum

The spec says the shared contract is 1.0 s minimum. The inference module rejects clips < 1.5 s.
This is in a shared module — change requires Surya's sign-off.

---

## 5. In-House Recording Consent

**Question for Surya:**
- Do we have legal/ethics clearance to collect voice recordings for biometric purposes?
- Does a template consent form already exist from another project, or should we use the new `CONSENT_FORM.md`?
- Who is the principal investigator for consent form storage purposes?
- When can recording sessions start? (Target: at least by Day 2 so cal data exists before Day 3 freeze.)

---

## 6. Common Voice Biometric Consent

**Question for Surya:**
- Has someone confirmed that the Common Voice CC0 licence covers biometric processing (speaker verification)?
- If not, can we use Common Voice only for ASR (not SV evaluation)?
- Fallback: use only VoxCeleb1-O for test-std and in-house for test-AAA (which we control).

---

## 7. Shared Augmentation Library Overlap (SHARED.md Section 6)

**Observation:** Navadeep implemented `aaa_engine/training/augment.py` with seeded, reproducible transforms (MUSAN noise injection, RIR convolution, bandlimiting low-pass, 16→8→16 kHz resampling, and ffmpeg-based codec roundtripping). This overlaps the shared augmentation library requirement specified in SHARED.md section 6.

**Local Augmentation State (Truthful Audit):**
- **Active locally:** Resample chain (16→8→16 kHz) and bandlimit filter (2000–7000 Hz cutoff) run in-process via PyTorch/torchaudio with no external dependencies or datasets.
- **Disabled locally:** MUSAN additive noise (disabled: no MUSAN corpus on local dev machine; config `musan_dir: null`), RIRS_NOISES reverb convolution (disabled: no RIR corpus locally; config `rirs_dir: null`), and codec roundtripping (disabled: `ffmpeg` is not on local PATH).
- **Requires ffmpeg:** Codec roundtripping (`opus`, `aac`, `amr_nb`, `g711_ulaw`) strictly requires `ffmpeg` binary on system PATH. When absent, `check_ffmpeg_codecs()` returns empty list and codec augmentation is cleanly bypassed. On HPC, ffmpeg and corpus directories must be configured.

**Questions for Surya:**
- Does `aaa_engine/training/augment.py` become the shared team augmentation library across workstreams (e.g., for Aditya's AAA-AD and Nandan's AAA-ASR)?
- Or does Surya plan a different shared augmentation module that `augment.py` should integrate with?
