# Items Requiring Action / Approval from Surya

**To:** Surya (Layer 1 Integration Lead / Project Architect)  
**From:** Navadeep Maka (Lead Model Engineer, AAA-SV)  
**Date:** Saturday 15 November 2026  
**Component:** AAA-SV (Adaptive Speaker Verification)

---

## 1. Request for Approval: Validation Split Dataset Deviation
- **Specification Expectation:** Specification originally outlined evaluation against Common Voice Set A.
- **Implemented Reality:** Validation split was extracted directly from the VoxCeleb1 Development partition (1,211 speakers, 2,422 utterances, 733,866 cross/same speaker trial pairs; SHA256 manifest starts with `9739f2a8...`).
- **Reason:** VoxCeleb1 dev was pre-staged and accessible on the HPC environment; Common Voice Set A was not pre-staged or partitioned in time for the training kickoff. All 40 VoxCeleb1 test speakers (`id10270`–`id10309`) were strictly excluded to ensure 0 data leakage.
- **Action Needed from Surya:** Formal review and sign-off on accepting the VoxCeleb1 dev holdout as the baseline validation partition.

---

## 2. Calibration Dataset: Common Voice Set B Required
- **Current State:** The adaptive voiceprint update gates in `maybe_update()` have been refactored away from hardcoded numbers and now read dynamically from `calibration.json` (`tau_anchor = 0.85`, `q_min = 0.70`, `threshold = 0.5`).
- **Limitation:** `calibration.json` is currently deployed with `fitted: false` because no clean, dedicated out-of-domain calibration partition was available.
- **Action Needed from Surya:** Provide / stage Common Voice Set B on cluster storage so the offline calibration script (`aaa_engine/calibration/calibrator.py`) can be executed to fit calibrated log-odds score transforms and optimize the empirical gating thresholds.

---

## 3. Adapter Integration & Handoff
- **Wrapper Status:** Production `AAASV` wrapper is fully implemented in `aaa_engine/inference/aaasv.py`.
- **Architectural Isolation:** The class contains zero database access, zero network calls, and adheres strictly to dataclass return types (`AAASVResponse`, `Voiceprint`, `VerifyResult`, `EmbedResult`).
- **Action Needed from Surya:** Integrate the `AAASV` class into the Layer 1 service pipeline. Detailed method signatures, duration boundaries (1.0s–30.0s), and error code taxonomies are documented in `docs/evidence/day4/INTEGRATION_ACCEPTANCE.md`.

---

## 4. Staging of VoxCeleb1-O Test Audio on HPC
- **Current State:** The official VoxCeleb1-O (cleaned) test trials protocol requires audio from the 40 test speakers (`id10270`–`id10309`). On the cluster, only the VoxCeleb1 dev directory was uncompressed and staged.
- **Action Needed from Surya:** Confirm location or stage the Vox1-O test speaker wav files on cluster scratch/storage so standard benchmark evaluation (VoxCeleb1-O EER) can be officially executed and published.
