# Developer Test Report — Day 4 — Navadeep (AAA-SV)

**Author:** Navadeep Maka (Lead Model Engineer, AAA-SV)  
**Sprint Day:** Day 4 — Saturday 15 Nov 2026  
**Scope:** Full Regression Pass Across All 6 Test Suites + Offline Load Verification  
**Execution Command:** `python -m pytest tests/ -v`  
**Result:** 113 passed, 0 failed in 10.67s  

---

## 1. Unified Test Suite Execution Log

```text
============================= test session starts =============================
platform win32 -- Python 3.12.4, pytest-9.0.2, pluggy-1.6.0
rootdir: D:\AAA_Layer1
collected 113 items

tests/test_aaasv.py (19 items) ...................                   [ 16%]
tests/test_calibrator.py (27 items) ...........................      [ 40%]
tests/test_eval_toolkit.py (18 items) ..................             [ 56%]
tests/test_layer1.py (34 items) ..................................   [ 86%]
tests/test_stability_head.py (10 items) ..........                   [ 95%]
tests/test_sv_pipeline.py (5 items) .....                            [100%]

======================= 113 passed, 1 warning in 10.67s =======================
```

---

## 2. Test Suite Breakdown

| Suite | Tests | Status | Scope |
|---|---|---|---|
| `test_aaasv.py` | 19 | PASS | Duration bounds (1.0s–30.0s), silence detection, NaN/Inf rejection, enrollment, verification, and `maybe_update` calibration gating. |
| `test_calibrator.py` | 27 | PASS | Logistic calibrator, Platt scaling, Adaptive S-Norm cohort indexing, and JSON serialization. |
| `test_eval_toolkit.py` | 18 | PASS | Trial pair generation, EER bisection, ROC convex hull, minDCF calculation, and CLI interface. |
| `test_layer1.py` | 34 | PASS | Baseline speaker verification, cosine scoring, threshold decision, audio preprocessing, and DB isolation. |
| `test_stability_head.py` | 10 | PASS | Quality estimation head architecture, MSE objective, stop-gradient target, and parameter gradient checks. |
| `test_sv_pipeline.py` | 5 | PASS | Model loader, offline staged execution, fallback stubbing, and end-to-end enrollment/verification consistency. |

---

## 3. Offline Load Verification Test

- **Script:** `python hpc/offline_load_test.py`
- **Result:** PASS (`docs/evidence/day4/offline_load_proof.json`)
- **Verified:** Instantiation without network, stub inference fallback, 192-dim L₂-normalized embedding output.
