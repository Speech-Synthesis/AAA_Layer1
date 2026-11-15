# AAA-SV Peer Review Package — Karthik's Verification Checklist

**Target Reviewer:** Karthik (Layer 2 Lead / Independent Peer Reviewer)  
**Author:** Navadeep Maka (Lead Model Engineer, AAA-SV)  
**Date:** Saturday 15 November 2026  
**Branch:** `feat/navadeep-aaa-sv`  
**Repository:** `https://github.com/Speech-Synthesis/AAA_Layer1.git`

> [!IMPORTANT]
> **Independent Review Rule:** Per sprint rules, Navadeep CANNOT execute Karthik's verification tests or fill in his evaluation checklist on his behalf. Karthik must clone the repository independently, execute the verification suite, and populate the review findings below.

---

## 1. Setup & Environment Instructions

Execute the following commands in a clean Python 3.10 / 3.12 virtual environment:

```bash
git clone https://github.com/Speech-Synthesis/AAA_Layer1.git
cd AAA_Layer1
git checkout feat/navadeep-aaa-sv
pip install -r requirements.txt
python -m pytest tests/ -v
```

**Expected Outcome:** 113 tests collected, 113 passed.

---

## 2. Frozen Peer Review Verification Checklist

Please verify each item and check the box `[x]` upon confirmation:

- [ ] **113/113 Unit Tests Passing:** All tests pass cleanly (`python -m pytest tests/ -v`).
- [ ] **Valid Embedding Generation:** `AAASV.embed()` returns `AAASVResponse` with `ok=True` for valid 3.0s 16 kHz audio.
- [ ] **Lower Duration Bound Enforced:** `AAASV.embed()` returns error code `E_TOO_SHORT` for audio duration $< 1.0\text{ s}$.
- [ ] **Upper Duration Bound Enforced:** `AAASV.embed()` returns error code `E_TOO_LONG` for audio duration $> 30.0\text{ s}$.
- [ ] **Silence Detection Enforced:** `AAASV.embed()` returns error code `E_SILENT` for silent audio ($\text{RMS} < -60\text{ dBFS}$).
- [ ] **Enrollment Contract:** `AAASV.enroll()` returns a valid `Voiceprint` with the specified `speaker_id` and average embedding.
- [ ] **Verification Contract:** `AAASV.verify()` returns `VerifyResult` with cosine similarity score in $[-1.0, 1.0]$.
- [ ] **Adaptive Update Gating:** `AAASV.maybe_update()` respects $\tau_{\text{anchor}}$ ($0.85$) and $q_{\text{min}}$ ($0.70$) thresholds loaded from `calibration.json`.
- [ ] **Embedding Invariants:** Embedding vector shape is strictly `(192,)`, dtype `float32`, and L₂ norm $\|e\|_2 \approx 1.0 \pm 10^{-5}$.
- [ ] **Duration Warning:** Audio between $1.0\text{ s}$ and $2.0\text{ s}$ emits the non-fatal warning `W_SHORT`.
- [ ] **Model Card Integrity:** Complete and accurate model card present at `MODEL_CARD.md` (and `04_FINAL_DELIVERABLES_NAVADEEP/04_FINAL_DELIVERABLES/MODEL_EVALUATION/MODEL_CARD.md`).
- [ ] **Calibration File Presence:** `calibration.json` present with `fitted: false` and default thresholds.
- [ ] **Integrity Checksums:** `RELEASE_ARTIFACTS/SHA256SUMS.txt` present with correct SHA256 hashes for `checkpoint_best.pt` and `checkpoint_last.pt`.
- [ ] **Code Hygiene & Privacy:** Zero PII, zero credentials/API keys, and no hardcoded absolute user directory paths in committed code.

---

## 3. Defects, Deviations & Observations (To be completed by Karthik)

*Karthik to document any discrepancies, failed assertions, or notes below:*

```text
[Defects Log - Karthik to fill in]

1.
2.
3.

Overall Review Verdict: [ ACCEPT / CONDITIONAL ACCEPT / REJECT ]
Reviewer Signature: Karthik
Review Date: ____________________
```
