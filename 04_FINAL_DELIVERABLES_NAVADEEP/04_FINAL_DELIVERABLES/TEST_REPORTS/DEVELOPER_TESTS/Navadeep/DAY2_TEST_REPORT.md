# Developer Test Report — Day 2 — Navadeep (AAA-SV)
**Author:** Navadeep Maka  
**Sprint Day:** Day 2 — Thursday 12 Nov 2026  
**Scope:** Phase 3.A (Stability Head), Phase 3.B (Evaluation Toolkit), Phase 3.C (Score Calibration), Phase 3.D (Inference Wrapper)  
**Execution Command:** `python -m pytest tests/test_stability_head.py tests/test_eval_toolkit.py tests/test_calibrator.py tests/test_aaasv.py -v`  
**Result:** 74 passed, 0 failed in 6.2s  

---

## 1. Test Breakdown

### 1.1 Stability Head (`test_stability_head.py`) — 10 Passed
- `test_q_output_range` PASSED: Range clamped to $[0, 1]$.
- `test_no_backbone_gradient` PASSED: Zero gradient propagated back to backbone through $\mathcal{L}_q$.
- `test_no_gradient_through_target` PASSED: Stop-gradient on $y_q$ verified.
- `test_save_load_resume` PASSED: Exact numerical state recovery for model and optimizer.
- `test_variable_length_input` PASSED: Verified on batch sizes 1 to 32 and pool dimensions 64 to 6144.
- `test_q_mse_loss_zero` PASSED: Loss $< 10^{-12}$ on identical inputs.
- `test_joint_loss_formula` PASSED: Verified $\mathcal{L} = \mathcal{L}_{\text{AAM}} + 0.1 \cdot \mathcal{L}_q$.

### 1.2 Evaluation Toolkit (`test_eval_toolkit.py`) — 18 Passed
- `test_eer_perfect_separation` & `test_eer_chance_level` PASSED.
- `test_mindcf_formula` PASSED: Exact match against NIST SRE specification ($p_t=0.05, C_{fa}=1, C_{miss}=10$).
- `test_trial_list_sha256_stable` PASSED: Hashing invariant to row permutations.
- `test_freeze_guard` PASSED: Raises `RuntimeError` when test manifests are evaluated without `--freeze`.
- Multi-bucket generators (duration, SNR, codecs) PASSED.

### 1.3 Calibration & AS-Norm (`test_calibrator.py`) — 27 Passed
- `test_as_norm_hand_computed` PASSED: Matches manual hand-calculated cohort normalization.
- `test_logistic_calibrator_fit_predict` & `test_cllr_decreases_after_calibration` PASSED: L-BFGS-B optimization reduces $C_{\text{llr}}$.
- `test_cllr_perfect` ($0.0$) and `test_cllr_chance` ($\approx 1.0$) PASSED.
- `test_threshold_selection_far01` PASSED: Realized FAR $\le 0.01$.
- `test_manifest_guard` PASSED: Rejects non-calibration manifests.
- `test_calibration_json_roundtrip` PASSED.

### 1.4 Inference Wrapper (`test_aaasv.py`) — 19 Passed
- `test_embed_too_short` PASSED: Audio $< 1.0$s returns `E_TOO_SHORT`.
- `test_embed_too_long` PASSED: Audio $> 30.0$s returns `E_TOO_LONG`.
- `test_embed_silent` PASSED: RMS $< -60$ dBFS returns `E_SILENT`.
- `test_embed_stereo_converts_to_mono` PASSED: Automatic 2-channel downmix.
- `test_embed_1s_2s_w_short_warning` PASSED: Returns `W_SHORT` warning and low confidence.
- `test_response_envelope_fields` PASSED: All fields present.
- `test_maybe_update_gates` PASSED: Adaptation strictly gated on score $> 0.85$ and $q > 0.70$.
