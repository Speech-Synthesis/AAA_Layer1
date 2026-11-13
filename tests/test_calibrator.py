# =============================================================
# tests/test_calibrator.py
# AAA Engine — Unit tests for aaa_engine/calibration/calibrator.py
# =============================================================

from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup – ensure aaa_engine package is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from aaa_engine.calibration.calibrator import (
    ASNorm,
    LogisticCalibrator,
    compute_cllr,
    load_calibration,
    save_calibration,
    select_threshold,
    _guard_manifest,
    _cosine_similarity,
    _l2_norm,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. AS-Norm hand-computed test
# ══════════════════════════════════════════════════════════════════════════════

class TestASNormHandComputed:
    """
    Verify AS-Norm output against a fully manual calculation using a 3-vector
    cohort (K=3, so ALL cohort vectors are used → standard score normalisation).
    """

    def _make_cohort_and_pair(self):
        """
        Cohort: 3 unit vectors in R^2.
        Enrol:  [1, 0] (unit x-axis)
        Test:   [0, 1] (unit y-axis)

        raw_score = cos([1,0],[0,1]) = 0.0

        cosines of enrol vs cohort:
            c1=[1,0]  → 1.0
            c2=[0,1]  → 0.0
            c3=[1/√2, 1/√2] → 1/√2 ≈ 0.7071

        mean_e = (1.0 + 0.0 + 0.7071) / 3 ≈ 0.5690
        std_e  = std([1.0, 0.0, 0.7071])

        cosines of test vs cohort (test=[0,1]):
            c1=[1,0]  → 0.0
            c2=[0,1]  → 1.0
            c3=[1/√2, 1/√2] → 1/√2 ≈ 0.7071

        mean_t = (0.0 + 1.0 + 0.7071) / 3 ≈ 0.5690
        std_t  = same as std_e (symmetric)

        norm_score = 0.5 * ((0 - mean_e)/std_e + (0 - mean_t)/std_t)
                   = 0.5 * 2 * (-mean_e/std_e)
                   = -mean_e/std_e
        """
        cohort = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)],
        ], dtype=np.float64)
        enroll = np.array([1.0, 0.0])
        test = np.array([0.0, 1.0])
        return cohort, enroll, test

    def _manual_norm_score(self):
        vals_e = np.array([1.0, 0.0, 1.0 / math.sqrt(2)])
        mean_e = np.mean(vals_e)
        std_e = np.std(vals_e, ddof=0)
        # By symmetry vals_t has the same stats
        raw = 0.0
        return 0.5 * ((raw - mean_e) / std_e + (raw - mean_e) / std_e)

    def test_as_norm_hand_computed(self):
        cohort, enroll, test = self._make_cohort_and_pair()
        asnorm = ASNorm(cohort, K=3)
        result = asnorm.normalize(enroll, test)
        expected = self._manual_norm_score()
        assert math.isclose(result, expected, rel_tol=1e-6), (
            f"AS-Norm mismatch: got {result:.8f}, expected {expected:.8f}"
        )

    def test_as_norm_k_less_than_cohort(self):
        """K < N should use only the top-K similarities."""
        cohort, enroll, test = self._make_cohort_and_pair()
        # K=1: only the best match for each side
        asnorm = ASNorm(cohort, K=1)
        result = asnorm.normalize(enroll, test)
        # With K=1: mean_e=1.0, std_e=1e-12, mean_t=1.0, std_t=1e-12
        # raw=0 → both sides give (0 - 1.0)/1e-12 → very large negative
        assert result < -1e6, f"Expected very large negative with K=1, got {result}"

    def test_as_norm_identical_embeddings(self):
        """Identical enrol and test should give a positive normalised score."""
        cohort = np.array([
            [0.5, 0.5],
            [-1.0, 0.0],
            [0.0, -1.0],
        ], dtype=np.float64)
        v = np.array([1.0, 0.0])
        asnorm = ASNorm(cohort, K=3)
        result = asnorm.normalize(v, v)
        # raw_score = 1.0 (identical unit vectors); cohort sims are mixed
        assert result > 0.0, f"Expected positive score for identical embeddings, got {result}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Logistic calibrator fit & predict
# ══════════════════════════════════════════════════════════════════════════════

class TestLogisticCalibratorFitPredict:

    def _make_separable_data(self, n: int = 200, seed: int = 42):
        """
        Linearly separable dataset: genuine scores ~ N(0.8, 0.05²),
        impostor scores ~ N(-0.3, 0.1²).
        Feature vector: [score, q, log_dur, snr_norm, channel_class].
        """
        rng = np.random.default_rng(seed)
        n_half = n // 2
        gen_scores = rng.normal(0.8, 0.05, size=n_half)
        imp_scores = rng.normal(-0.3, 0.10, size=n_half)
        all_scores = np.concatenate([gen_scores, imp_scores])

        # q: genuines tend to have higher stability
        gen_q = rng.uniform(0.7, 1.0, size=n_half)
        imp_q = rng.uniform(0.0, 0.5, size=n_half)
        all_q = np.concatenate([gen_q, imp_q])

        # Other features: random but uninformative
        all_log_dur = rng.uniform(0.0, 2.0, size=n)
        all_snr_norm = rng.uniform(0.5, 2.0, size=n)
        all_chan = rng.integers(0, 3, size=n).astype(float)

        X = np.column_stack([all_scores, all_q, all_log_dur, all_snr_norm, all_chan])
        y = np.array([1] * n_half + [0] * n_half, dtype=np.int32)
        return X, y

    def test_logistic_calibrator_fit_predict(self):
        X, y = self._make_separable_data()
        cal = LogisticCalibrator(use_q=True)
        cal.fit(X, y)

        log_odds = X @ cal.coef_ + cal.intercept_

        # Genuine should have positive log-odds, impostors negative
        n_half = len(y) // 2
        gen_log_odds = log_odds[:n_half]
        imp_log_odds = log_odds[n_half:]

        assert np.mean(gen_log_odds) > np.mean(imp_log_odds), (
            "Calibrated genuine log-odds should exceed impostor log-odds on average"
        )

        # Separation: at least 80 % of genuines should have log_odds > 0
        pct_gen_positive = np.mean(gen_log_odds > 0)
        assert pct_gen_positive >= 0.80, (
            f"Expected ≥80% genuines with log_odds>0, got {pct_gen_positive:.2%}"
        )

    def test_cllr_decreases_after_calibration(self):
        """Calibrated Cllr should be ≤ Cllr of raw scores (nearly always true)."""
        X, y = self._make_separable_data()
        cal = LogisticCalibrator(use_q=True)
        cal.fit(X, y)

        # Raw score Cllr (using only the first feature as "log-odds")
        raw_log_odds = X[:, 0]  # raw normalised score
        raw_cllr = compute_cllr(raw_log_odds, y)
        # Calibrated Cllr
        assert cal.cllr is not None
        assert cal.cllr <= raw_cllr + 1e-6, (
            f"Calibrated Cllr {cal.cllr:.4f} should be ≤ raw Cllr {raw_cllr:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Cllr perfect predictions
# ══════════════════════════════════════════════════════════════════════════════

class TestCllrPerfect:

    def test_cllr_perfect(self):
        """
        Cllr → 0 when log-odds → ±∞ in the correct direction
        (genuine log-odds very large, impostor log-odds very negative).
        """
        rng = np.random.default_rng(0)
        labels = np.array([1] * 500 + [0] * 500, dtype=np.int32)
        # Very confident correct predictions
        log_odds = np.where(labels == 1, 20.0, -20.0).astype(np.float64)
        cllr = compute_cllr(log_odds, labels)
        assert cllr < 0.001, f"Expected Cllr ≈ 0 for perfect predictions, got {cllr:.6f}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. Cllr ≈ 1 for chance (random) predictions
# ══════════════════════════════════════════════════════════════════════════════

class TestCllrChance:

    def test_cllr_chance(self):
        """
        A log-odds of 0.0 for every trial corresponds to a system that says
        'prior probability only'.  For equal priors this gives Cllr = 1.0.
        """
        n = 1000
        labels = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int32)
        log_odds = np.zeros(n, dtype=np.float64)  # always 0 → prior only
        cllr = compute_cllr(log_odds, labels)
        # log2(1 + exp(0)) = log2(2) = 1.0, so Cllr = 0.5*(1 + 1) = 1.0
        assert math.isclose(cllr, 1.0, rel_tol=1e-6), (
            f"Expected Cllr ≈ 1.0 for uninformative system, got {cllr:.6f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Threshold selection FAR ≤ 0.01
# ══════════════════════════════════════════════════════════════════════════════

class TestThresholdSelectionFAR01:

    def test_threshold_selection_far01(self):
        """
        select_threshold should return realized_far ≤ 0.01 on the given data.
        """
        rng = np.random.default_rng(7)
        n_gen = 500
        n_imp = 5000
        gen_scores = rng.normal(2.0, 0.5, size=n_gen)
        imp_scores = rng.normal(-1.0, 0.5, size=n_imp)
        scores = np.concatenate([gen_scores, imp_scores])
        labels = np.array([1] * n_gen + [0] * n_imp, dtype=np.int32)

        result = select_threshold(scores, labels, target_far=0.01)

        assert result["realized_far"] <= 0.01 + 1e-9, (
            f"realized_far {result['realized_far']} exceeds target 0.01"
        )
        assert result["n_impostor"] == n_imp
        assert result["n_genuine"] == n_gen
        assert "threshold" in result
        assert "realized_frr" in result

    def test_threshold_selection_far_001_low_impostor_count(self):
        """
        When target_far=0.001 and n_impostor < 3000, result should include
        'validated': False and a warning message.
        """
        rng = np.random.default_rng(99)
        n_gen = 200
        n_imp = 100  # deliberately < 3000
        gen_scores = rng.normal(2.0, 0.3, size=n_gen)
        imp_scores = rng.normal(-1.0, 0.3, size=n_imp)
        scores = np.concatenate([gen_scores, imp_scores])
        labels = np.array([1] * n_gen + [0] * n_imp, dtype=np.int32)

        result = select_threshold(scores, labels, target_far=0.001)

        assert result.get("validated") is False
        assert "warning" in result
        assert "3000" in result["warning"]

    def test_threshold_selection_far_001_high_impostor_count(self):
        """
        When target_far=0.001 and n_impostor >= 3000, 'validated' should be True.
        """
        rng = np.random.default_rng(42)
        n_gen = 500
        n_imp = 5000  # >= 3000
        gen_scores = rng.normal(3.0, 0.3, size=n_gen)
        imp_scores = rng.normal(-2.0, 0.3, size=n_imp)
        scores = np.concatenate([gen_scores, imp_scores])
        labels = np.array([1] * n_gen + [0] * n_imp, dtype=np.int32)

        result = select_threshold(scores, labels, target_far=0.001)

        assert result.get("validated") is True
        assert "warning" not in result


# ══════════════════════════════════════════════════════════════════════════════
# 6. Manifest guard
# ══════════════════════════════════════════════════════════════════════════════

class TestManifestGuard:

    @pytest.mark.parametrize("bad_path", [
        "data/manifests/train.csv",
        "data/manifests/val.csv",
        "data/manifests/test.csv",
        "/some/absolute/path/train_speakers.csv",
        "runs/exp1/val_set.csv",
        "mytest.csv",
    ])
    def test_manifest_guard(self, bad_path: str):
        with pytest.raises(ValueError, match=r"train|val|test"):
            _guard_manifest(bad_path)

    @pytest.mark.parametrize("good_path", [
        "data/manifests/cal.csv",
        "data/manifests/cal_2024.csv",
        "runs/exp1/cal_set.csv",
        "calibration_manifest.csv",
    ])
    def test_manifest_guard_valid(self, good_path: str):
        # Should not raise
        _guard_manifest(good_path)


# ══════════════════════════════════════════════════════════════════════════════
# 7. Calibration JSON round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrationJsonRoundtrip:

    def _fit_simple_calibrator(self) -> LogisticCalibrator:
        rng = np.random.default_rng(1)
        n = 100
        gen_scores = rng.normal(1.0, 0.1, size=n // 2)
        imp_scores = rng.normal(-1.0, 0.1, size=n // 2)
        scores = np.concatenate([gen_scores, imp_scores])
        labels = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int32)

        X = np.column_stack([
            scores,
            np.zeros(n),   # q
            np.zeros(n),   # log_duration
            np.zeros(n),   # snr_norm
            np.zeros(n),   # channel_class
        ])
        cal = LogisticCalibrator(use_q=True)
        cal.fit(X, labels)
        return cal

    def test_calibration_json_roundtrip(self):
        """save_calibration then load_calibration produces identical dict."""
        cal = self._fit_simple_calibrator()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "calibration.json")

            save_calibration(
                cal,
                out_path,
                model_label="aaa_sv_test",
                as_norm_K=300,
                threshold=0.5,
                realized_far=0.009,
                cohort_size=1000,
                cal_manifest_hash="abc123",
                timestamp="2024-01-01T00:00:00Z",
            )

            loaded = load_calibration(out_path)

        # Verify required fields are present
        required_fields = [
            "model_label", "as_norm_K", "threshold", "realized_far",
            "cllr", "feature_names", "logistic_coef", "logistic_intercept",
            "cohort_size", "cal_manifest_hash", "timestamp",
        ]
        for field in required_fields:
            assert field in loaded, f"Missing field '{field}' in loaded dict"

        # Verify scalar values
        assert loaded["model_label"] == "aaa_sv_test"
        assert loaded["as_norm_K"] == 300
        assert math.isclose(loaded["threshold"], 0.5)
        assert math.isclose(loaded["realized_far"], 0.009)
        assert loaded["cohort_size"] == 1000
        assert loaded["cal_manifest_hash"] == "abc123"
        assert loaded["timestamp"] == "2024-01-01T00:00:00Z"

        # Verify coef round-trip (via JSON → list)
        assert isinstance(loaded["logistic_coef"], list)
        assert len(loaded["logistic_coef"]) == len(cal.coef_)
        for a, b in zip(loaded["logistic_coef"], cal.coef_.tolist()):
            assert math.isclose(a, b, rel_tol=1e-10), (
                f"Coefficient mismatch: {a} vs {b}"
            )

        assert math.isclose(loaded["logistic_intercept"], cal.intercept_, rel_tol=1e-10)
        assert loaded["feature_names"] == cal.feature_names

        # Verify Cllr is stored
        assert cal.cllr is not None
        assert math.isclose(loaded["cllr"], cal.cllr, rel_tol=1e-10)

    def test_calibration_json_is_valid_json(self):
        """The saved file must be parseable by json.loads independently."""
        cal = self._fit_simple_calibrator()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "cal.json")
            save_calibration(cal, out_path, model_label="x", as_norm_K=50,
                             threshold=0.0, realized_far=0.0, cohort_size=10,
                             cal_manifest_hash="deadbeef")

            with open(out_path, "r") as fh:
                raw = fh.read()

        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


# ══════════════════════════════════════════════════════════════════════════════
# Additional edge-case tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_logistic_calibrator_no_q(self):
        """LogisticCalibrator with use_q=False should use 4 features."""
        cal = LogisticCalibrator(use_q=False)
        assert len(cal.feature_names) == 4
        assert "q" not in cal.feature_names

    def test_logistic_calibrator_raises_before_fit(self):
        cal = LogisticCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.calibrate(0.5)

    def test_calibrate_single_score(self):
        """calibrate() should return a float after fitting."""
        rng = np.random.default_rng(0)
        n = 100
        gen = rng.normal(1.0, 0.1, size=n // 2)
        imp = rng.normal(-1.0, 0.1, size=n // 2)
        X = np.column_stack([
            np.concatenate([gen, imp]),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
        ])
        y = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int32)
        cal = LogisticCalibrator(use_q=True)
        cal.fit(X, y)

        result = cal.calibrate(0.8, q=0.9, log_duration=1.0, snr_norm=1.5, channel_class=0.0)
        assert isinstance(result, float)
        # A clearly genuine score should give positive log-odds
        assert result > 0.0

    def test_asnorm_bad_cohort_shape(self):
        """ASNorm should reject 1-D cohort arrays."""
        with pytest.raises(ValueError):
            ASNorm(np.array([1.0, 0.0, 1.0]))  # 1-D

    def test_compute_cllr_requires_both_classes(self):
        with pytest.raises(ValueError, match="both positive"):
            compute_cllr(np.array([1.0, 2.0]), np.array([1, 1]))
