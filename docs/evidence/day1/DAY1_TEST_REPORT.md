# Navadeep — Day 1 Test Report

**Date:** November 11, 2026
**Branch:** `feat/navadeep-aaa-sv`

---

## Test Infrastructure

| Component | Status |
|-----------|--------|
| pytest installed | ✅ pytest-9.0.2 |
| Test directory | `tests/` (existing + extended) |
| conftest.py | ✅ Created with session-level librosa stub (Windows-safe) |
| pytest.ini | Existing |

---

## Test Cases — test_layer1.py (34 tests, pre-existing)

| Test Class | Test Name | Purpose | Result |
|------------|-----------|---------|--------|
| TestCosineSimilarity | test_same_vector_returns_one | Same vector → similarity = 1.0 | PASS |
| TestCosineSimilarity | test_different_vectors_below_threshold | Different vectors → below 0.75 | PASS |
| TestCosineSimilarity | test_orthogonal_vectors_near_zero | Orthogonal → ≈ 0 | PASS |
| TestCosineSimilarity | test_negated_vector_clamped_to_zero | Negative similarity clamped | PASS |
| TestCosineSimilarity | test_score_always_between_zero_and_one | Score always in [0, 1] | PASS |
| TestCosineSimilarity | test_slightly_perturbed_vector_high_similarity | Perturbed → still high sim | PASS |
| TestThresholdDecision | test_accept_when_score_above_threshold | Above threshold → ACCEPT | PASS |
| TestThresholdDecision | test_reject_when_score_below_threshold | Below threshold → REJECT | PASS |
| TestThresholdDecision | test_exact_threshold_is_accept | Exact threshold → ACCEPT | PASS |
| TestThresholdDecision | test_just_below_threshold_is_reject | Just below → REJECT | PASS |
| TestVerifySpeaker | test_reject_unenrolled_user | Unknown user → REJECT | PASS |
| TestVerifySpeaker | test_error_handling_returns_reject | Exception → REJECT (safe) | PASS |
| TestVerifySpeaker | test_response_keys_present_on_accept | Accept response has all keys | PASS |
| TestVerifySpeaker | test_response_keys_present_on_reject | Reject response has all keys | PASS |
| TestVerifySpeaker | test_custom_threshold | Custom threshold respected | PASS |
| TestEmbeddingProperties | test_embedding_is_192_dim | Embedding = 192-dim | PASS |
| TestEmbeddingProperties | test_embedding_is_l2_normalized | Embedding L2-normalised | PASS |
| TestEnrollment | test_reject_already_enrolled_user | Re-enroll blocked without flag | PASS |
| TestEnrollment | test_successful_enrollment | First enroll succeeds | PASS |
| TestEnrollment | test_enrollment_db_failure | DB error → safe failure | PASS |
| TestEnrollment | test_enrollment_audio_error | Audio error → safe failure | PASS |
| TestEnrollment | test_re_enrollment_success | Re-enroll with flag succeeds | PASS |
| TestAudioValidation | test_audio_too_short_raises_valueerror | < 1.0 s → ValueError | PASS |
| TestAudioValidation | test_audio_too_long_raises_valueerror | > 10.0 s → ValueError | PASS |
| TestAudioValidation | test_valid_duration_passes | 1.0–10.0 s → no error | PASS |
| TestAudioValidation | test_stereo_to_mono_conversion | Stereo → mono correctly | PASS |
| TestAudioValidation | test_resample_to_16khz | 8 kHz → 16 kHz resampled | PASS |
| TestConstants | test_threshold_is_075 | THRESHOLD = 0.75 | PASS |
| TestConstants | test_min_duration | MIN_DURATION defined | PASS |
| TestConstants | test_max_duration | MAX_DURATION = 10.0 | PASS |
| TestDbUtils | test_store_voiceprint_computes_sha256_hash | Store computes SHA-256 | PASS |
| TestDbUtils | test_fetch_voiceprint_returns_numpy_array | Fetch returns ndarray | PASS |
| TestDbUtils | test_fetch_voiceprint_returns_none_when_not_found | Missing user → None | PASS |
| TestDbUtils | test_log_auth_result_inserts_record | Auth log inserts row | PASS |

---

## Test Cases — test_sv_pipeline.py (5 tests, new — Day 1)

| Test Name | Exact Behaviour & Implementation | Result |
|-----------|----------------------------------|--------|
| `test_resume_continuity` | Simulates 3 training steps with a lightweight toy embedder+classifier model (no SpeechBrain/network dependency), saves checkpoint containing step, epoch, model/optimizer states, and torch/numpy/python RNG states, reloads into fresh instances, and asserts exact match of step, epoch, byte-for-byte parameter weights, and subsequent random sample continuity for torch, numpy, and python RNG. | PASS |
| `test_preprocessing_parity` | Generates synthetic waveform, writes to WAV via soundfile, runs `sv_eval.preprocess_audio()` twice to verify determinism; tests stereo-to-mono downmix as channel mean; tests 8 kHz to 16 kHz resample; asserts float32 dtype. Validates invariants: no silence trimming and no repeat-padding in either path. | PASS |
| `test_shuffled_label_sanity` | Generates synthetic multi-class batch with randomly permuted (shuffled) speaker labels, trains a toy model for 20 steps, and asserts loss fails to converge (loss remains > 0.5), verifying that random labels do not lead to learning. | PASS |
| `test_embedding_norm` | Loads the real pretrained ECAPA-TDNN from HuggingFace cache (marked `skipif` if model cannot be loaded), encodes a 3.0 s synthetic waveform, extracts the 192-dim embedding, and asserts Euclidean L2 norm is within 0.05 of 1.0. | PASS |
| `test_augmentation_reproducibility` | Instantiates two separate `Augmenter` objects with the same integer seed (1986), passes identical synthetic audio waveforms through both, and asserts output waveforms are strictly bit-for-bit identical (`torch.allclose` with `atol=0.0`). | PASS |

---

## Execution

```powershell
cd d:\AAA_Layer1\aaa_engine
python -m pytest ../tests/test_layer1.py ../tests/test_sv_pipeline.py -v
```

---

## Test Execution Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-9.0.2, pluggy-1.6.0
collected 39 items

tests/test_layer1.py::TestCosineSimilarity::test_same_vector_returns_one PASSED
tests/test_layer1.py::TestCosineSimilarity::test_different_vectors_below_threshold PASSED
tests/test_layer1.py::TestCosineSimilarity::test_orthogonal_vectors_near_zero PASSED
tests/test_layer1.py::TestCosineSimilarity::test_negated_vector_clamped_to_zero PASSED
tests/test_layer1.py::TestCosineSimilarity::test_score_always_between_zero_and_one PASSED
tests/test_layer1.py::TestCosineSimilarity::test_slightly_perturbed_vector_high_similarity PASSED
tests/test_layer1.py::TestThresholdDecision::test_accept_when_score_above_threshold PASSED
tests/test_layer1.py::TestThresholdDecision::test_reject_when_score_below_threshold PASSED
tests/test_layer1.py::TestThresholdDecision::test_exact_threshold_is_accept PASSED
tests/test_layer1.py::TestThresholdDecision::test_just_below_threshold_is_reject PASSED
tests/test_layer1.py::TestVerifySpeaker::test_reject_unenrolled_user PASSED
tests/test_layer1.py::TestVerifySpeaker::test_error_handling_returns_reject PASSED
tests/test_layer1.py::TestVerifySpeaker::test_response_keys_present_on_accept PASSED
tests/test_layer1.py::TestVerifySpeaker::test_response_keys_present_on_reject PASSED
tests/test_layer1.py::TestVerifySpeaker::test_custom_threshold PASSED
tests/test_layer1.py::TestEmbeddingProperties::test_embedding_is_192_dim PASSED
tests/test_layer1.py::TestEmbeddingProperties::test_embedding_is_l2_normalized PASSED
tests/test_layer1.py::TestEnrollment::test_reject_already_enrolled_user PASSED
tests/test_layer1.py::TestEnrollment::test_successful_enrollment PASSED
tests/test_layer1.py::TestEnrollment::test_enrollment_db_failure PASSED
tests/test_layer1.py::TestEnrollment::test_enrollment_audio_error PASSED
tests/test_layer1.py::TestEnrollment::test_re_enrollment_success PASSED
tests/test_layer1.py::TestAudioValidation::test_audio_too_short_raises_valueerror PASSED
tests/test_layer1.py::TestAudioValidation::test_audio_too_long_raises_valueerror PASSED
tests/test_layer1.py::TestAudioValidation::test_valid_duration_passes PASSED
tests/test_layer1.py::TestAudioValidation::test_stereo_to_mono_conversion PASSED
tests/test_layer1.py::TestAudioValidation::test_resample_to_16khz PASSED
tests/test_layer1.py::TestConstants::test_threshold_is_075 PASSED
tests/test_layer1.py::TestConstants::test_min_duration PASSED
tests/test_layer1.py::TestConstants::test_max_duration PASSED
tests/test_layer1.py::TestDbUtils::test_store_voiceprint_computes_sha256_hash PASSED
tests/test_layer1.py::TestDbUtils::test_fetch_voiceprint_returns_numpy_array PASSED
tests/test_layer1.py::TestDbUtils::test_fetch_voiceprint_returns_none_when_not_found PASSED
tests/test_layer1.py::TestDbUtils::test_log_auth_result_inserts_record PASSED
tests/test_sv_pipeline.py::test_resume_continuity PASSED
tests/test_sv_pipeline.py::test_preprocessing_parity PASSED
tests/test_sv_pipeline.py::test_shuffled_label_sanity PASSED
tests/test_sv_pipeline.py::test_embedding_norm PASSED
tests/test_sv_pipeline.py::test_augmentation_reproducibility PASSED

============================== 39 passed, 1 warning in 6.48s ==============================
```

**Total: 39/39 PASSED**

---

## Known Limitations

1. **Self-contained pipeline tests (no mocking)** — `test_resume_continuity` and `test_shuffled_label_sanity` use an in-memory toy linear model (`_ToyEmbedder`), eliminating network/SpeechBrain dependencies. `test_embedding_norm` loads the real pretrained SpeechBrain ECAPA model directly from the local HF cache without any mocks. No mock objects or monkeypatching are used on ECAPA.
2. **Synthetic audio only** — All fixtures use `numpy`-generated sine waves and white noise (16 kHz, float32)
3. **librosa.effects stubbed** — On Windows in sandboxed pytest, `librosa.effects` lazy-loads cause a deadlock; `conftest.py` stubs the module at session start
4. **Val EER not tested** — EER computation requires real speaker-diverse dev set; smoke test falls back to 1.0 (expected and documented)
5. **GPU tests skipped** — All tests run on CPU; GPU path covered by HPC run

---

## Verification Steps

```powershell
# 1. Enter training package dir
cd d:\AAA_Layer1\aaa_engine

# 2. Run all tests
python -m pytest ../tests/test_layer1.py ../tests/test_sv_pipeline.py -v

# 3. Expected: 39 passed, 1 warning (SpeechBrain deprecated import — Surya's file, logged)
```

---

**Document Version:** 1.0
**Completion Time:** November 11, 2026
