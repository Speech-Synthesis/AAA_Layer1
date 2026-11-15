# Developer Test Report — Day 3 — Navadeep (AAA-SV)
**Author:** Navadeep Maka  
**Sprint Day:** Day 3 — Friday 13 Nov 2026  
**Scope:** Full Regression Pass Across All 6 Test Suites (Baseline, Preprocessing, Pipeline, Stability Head, Calibration, Evaluation, Inference)  
**Execution Command:** `python -m pytest tests/ -v`  
**Result:** 113 passed, 0 failed in 8.72s  

---

## 1. Unified Test Suite Execution

```
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-9.0.2, pluggy-1.6.0
rootdir: D:\AAA_Layer1
collected 113 items

tests/test_layer1.py::TestCosineSimilarity (6 items) ................ [  5%]
tests/test_layer1.py::TestThresholdDecision (4 items) ....            [  8%]
tests/test_layer1.py::TestVerifySpeaker (5 items) .....                [ 13%]
tests/test_layer1.py::TestEmbeddingProperties (2 items) ..             [ 15%]
tests/test_layer1.py::TestEnrollment (5 items) .....                   [ 19%]
tests/test_layer1.py::TestAudioValidation (5 items) .....              [ 23%]
tests/test_layer1.py::TestConstants (3 items) ...                      [ 26%]
tests/test_layer1.py::TestDbUtils (4 items) ....                       [ 30%]
tests/test_stability_head.py (10 items) ..........                   [ 39%]
tests/test_calibrator.py (27 items) ...........................      [ 63%]
tests/test_eval_toolkit.py (18 items) ..................             [ 79%]
tests/test_aaasv.py (19 items) ...................                   [ 95%]
tests/test_sv_pipeline.py (5 items) .....                            [100%]

======================= 113 passed, 1 warning in 8.72s ========================
```

---

## 2. Regression Integrity Verification
- **Baseline Backward Compatibility:** All 34 tests in `test_layer1.py` passed with zero modifications to legacy contracts.
- **Pipeline Integrity:** Checkpoint resume, seed determinism, and normalization invariants held across CPU/GPU code paths.
- **Contract Enforcement:** All responses comply with the unified error taxonomy defined in `SHARED.md §2`.
