# AAA-SV Integration Acceptance Specification

**Component:** Layer 1 Adaptive Acoustic Speaker Verifier (`AAASV`)  
**Implementation:** `aaa_engine/inference/aaasv.py`  
**Engineer:** Navadeep Maka (Lead Model Engineer)  
**Integration Lead / Consumer:** Surya (Layer 1 Integration / Architecture)  
**Date:** Saturday 15 November 2026

---

## 1. Architectural Overview & Adapter Handoff

The `AAASV` engine provides the core speaker verification interface for the AAA pipeline. It is architected for strict isolation:
- **No Database Dependencies:** The wrapper performs pure numerical computation and memory operations; it possesses zero database or network access.
- **No Public API Breakage:** Implements clean factory and dataclass patterns (`AAASVResponse`, `Voiceprint`, `VerifyResult`, `EmbedResult`).
- **Adapter Location:** The primary production class `AAASV` resides directly in `aaa_engine/inference/aaasv.py`. Surya can import and wrap this class directly into the Layer 1 verification service without schema translations.

---

## 2. Public API Wrapper Contract

All public methods return an `AAASVResponse` wrapper containing:
```python
@dataclass
class AAASVResponse:
    ok: bool
    error_code: Optional[str]
    result: Any
    warnings: List[str]
    latency_ms: float
```

### 2.1. Method Signatures & Details

#### `embed(wav: np.ndarray, sr: int) -> AAASVResponse`
- **Purpose:** Extracts a 192-dimensional speaker embedding and auxiliary stability score $q(x)$.
- **Inputs:**
  - `wav`: 1D `np.ndarray` of type `float32`.
  - `sr`: Sampling rate in Hz (must be 16,000 Hz or resampled).
- **Returns:** `AAASVResponse.result` contains `EmbedResult(embedding, q_score, duration_s, sample_rate)`.
- **Potential Errors:** `E_FORMAT`, `E_TOO_SHORT`, `E_TOO_LONG`, `E_SILENT`, `E_NONFINITE`, `E_MODEL_UNAVAILABLE`, `E_INTERNAL`.
- **Potential Warnings:** `W_SHORT` (for $1.0\text{ s} \le t < 2.0\text{ s}$).

#### `enroll(wavs: List[np.ndarray], srs: List[int], speaker_id: str) -> AAASVResponse`
- **Purpose:** Enrolls a speaker from one or more audio samples, producing an aggregated `Voiceprint`.
- **Inputs:**
  - `wavs`: List of 1D float32 audio arrays.
  - `srs`: List of corresponding sample rates.
  - `speaker_id`: Unique non-empty string identifier.
- **Returns:** `AAASVResponse.result` contains `Voiceprint(speaker_id, embedding, n_utterances, created_at, updated_at, mean_q)`.
- **Potential Errors:** All `embed` errors, plus validation errors on input lists.

#### `verify(wav: np.ndarray, sr: int, voiceprint: Voiceprint) -> AAASVResponse`
- **Purpose:** Verifies whether a trial utterance matches a previously enrolled `Voiceprint`.
- **Inputs:**
  - `wav`: 1D float32 audio array.
  - `sr`: Sample rate in Hz.
  - `voiceprint`: Valid `Voiceprint` dataclass instance.
- **Returns:** `AAASVResponse.result` contains `VerifyResult(score, decision, threshold, q_score, confidence)`.
- **Scoring:** Cosine similarity $s = \frac{e_{\text{trial}} \cdot e_{\text{enroll}}}{\|e_{\text{trial}}\|_2 \|e_{\text{enroll}}\|_2} \in [-1.0, 1.0]$.
- **Decision:** $s \ge \text{threshold}$. Default threshold is $0.5$ (loaded from `calibration.json`).
- **Potential Errors:** `E_NO_ENROLLMENT` (if voiceprint missing/invalid) or `embed` errors.

#### `maybe_update(voiceprint: Voiceprint, emb: np.ndarray, q: float, gates: dict) -> Voiceprint`
- **Purpose:** Conditionally updates an enrolled voiceprint via online running average if quality and similarity pass strict confidence gates.
- **Update Criteria:** Fires if and only if:
  $$\text{gates}["\text{score}"] > \tau_{\text{anchor}} \quad \text{AND} \quad q > q_{\text{min}}$$
  (Where $\tau_{\text{anchor}}=0.85$ and $q_{\text{min}}=0.70$ are loaded from `calibration.json`).
- **Update Formulation:**
  $$e_{\text{new}} = \text{L2\_norm}\left(\frac{e_{\text{old}} \cdot n + e_{\text{trial}}}{n + 1}\right)$$
- **Returns:** Updated `Voiceprint` (or original instance if gates are not met).

---

## 3. Invariants & Duration Bounds

- **Duration Bounds:**
  - Minimum allowed duration: $1.0\text{ s}$ ($< 1.0\text{ s}$ rejected with `E_TOO_SHORT`).
  - Maximum allowed duration: $30.0\text{ s}$ ($> 30.0\text{ s}$ rejected with `E_TOO_LONG`).
  - Warning zone: $1.0\text{ s} \le t < 2.0\text{ s}$ emits `W_SHORT`.
- **Embedding Invariants:**
  - Dimensionality: exactly 192 elements.
  - Type: `np.float32`.
  - Normalization: L₂-norm strictly equal to $1.0 \pm 10^{-5}$.
- **Latency Target:** $\le 300\text{ ms}$ p95 on 3.0s speech audio (CPU x86_64). *(Note: real model latency not yet measured on production hardware; benchmark harness available at `hpc/measure_latency.py`)*.

---

## 4. Standard Error Codes Specification

| Error Code | Trigger Condition | Handler Action |
|---|---|---|
| `E_FORMAT` | Audio array is not 1D, wrong data type, or unparseable buffer | Reject with 400 Bad Request |
| `E_DECODE` | Compressed audio payload fails decoding | Reject with 400 Bad Request |
| `E_TOO_SHORT` | Duration $< 1.0\text{ s}$ | Prompt user for longer speech sample |
| `E_TOO_LONG` | Duration $> 30.0\text{ s}$ | Prompt user to provide shorter sample |
| `E_SILENT` | RMS energy $< -60\text{ dBFS}$ or all zeros | Prompt user to check microphone |
| `E_NONFINITE` | Audio buffer contains NaN or Infinite values | Reject corrupt sample |
| `E_NO_ENROLLMENT` | No voiceprint template found for the given speaker ID | Direct user to enrollment flow |
| `E_MODEL_UNAVAILABLE` | Checkpoint or config file missing from staged model path | Trigger system alert / 503 Service Unavailable |
| `E_VERSION_MISMATCH` | Incompatible model architecture or config version | Reject load / deploy compatible weights |
| `E_INTERNAL` | Uncaught Python exception during execution | Log stack trace, return 500 Internal Error |

---

## 5. Unit Test Coverage Mapping

All 113 unit tests pass on local CPU (`pytest tests/ -v`).

| Test File | Test Count | Scope of Coverage |
|---|---|---|
| `tests/test_aaasv.py` | 19 tests | Complete `AAASV` API contract, duration boundaries, silence, non-finite values, enrollment, verification, and `maybe_update` gating. |
| `tests/test_calibrator.py` | 27 tests | Multi-condition calibrator, logistic regression, AS-Norm cohort indexing, score normalization, calibration serialization. |
| `tests/test_eval_toolkit.py` | 18 tests | Trial pair generation, EER calculation, ROC convex hull, minDCF computation, confusion matrices, CLI arguments. |
| `tests/test_layer1.py` | 34 tests | End-to-end Layer 1 pipeline, audio preprocessing, voiceprint serialization, audio sanity guards, format parsing. |
| `tests/test_stability_head.py` | 10 tests | Auxiliary Stability Head $q(x)$ architecture, forward pass, gradient propagation, MSE loss formulation. |
| `tests/test_sv_pipeline.py` | 5 tests | Model loader, staged offline execution, stub fallback, end-to-end enrollment-verification consistency. |
| **Total** | **113 tests** | **100% of defined Layer 1 contract tested and passing.** |
