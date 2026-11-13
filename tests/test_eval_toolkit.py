# =============================================================
# tests/test_eval_toolkit.py
# AAA Engine — Phase 3.B  Unit tests for eval_toolkit
# =============================================================
# Tests:
#   1. test_eer_perfect_separation   — EER=0 when scores perfectly separated
#   2. test_eer_chance_level         — EER≈0.5 when labels are random
#   3. test_mindcf_formula           — verify minDCF matches hand-computed value
#   4. test_trial_list_sha256_stable — same inputs → same hash
#   5. test_freeze_guard             — raises RuntimeError for test manifest
#                                      without freeze
# =============================================================

from __future__ import annotations

import csv
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aaa_engine.evaluation.eval_toolkit as toolkit
from aaa_engine.evaluation.eval_toolkit import (
    BucketedTrialGenerator,
    EvalResult,
    TrialList,
    _bootstrap_ci,
    _compute_eer,
    _compute_frr_at_far,
    _compute_min_dcf,
    _sha256_trial_list,
    evaluate_model,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_simple_trial_list(
    pairs: List[Tuple[str, str, int]],
    bucket_name: str = "test_bucket",
    source_manifest: str = "data/manifests/val.csv",
) -> TrialList:
    """Build a minimal TrialList for testing (no real files needed)."""
    sha = _sha256_trial_list(pairs)
    n_pos = sum(1 for _, _, l in pairs if l == 1)
    n_neg = sum(1 for _, _, l in pairs if l == 0)
    return TrialList(
        pairs=pairs,
        sha256=sha,
        source_manifest=source_manifest,
        bucket_name=bucket_name,
        n_positive=n_pos,
        n_negative=n_neg,
    )


def _dummy_embedding_fn(scores_map: dict):
    """Return an embedding_fn that produces vectors whose cosine similarity
    equals the pre-computed value in scores_map[(enroll, test)]."""

    # We'll assign random unit vectors but patch cosine scoring instead.
    # For tests we construct embeddings so that dot(e1, e2) ≈ desired_score.
    dim = 192
    rng = np.random.default_rng(0)

    cache: dict = {}

    def _get(path: str) -> np.ndarray:
        if path not in cache:
            v = rng.standard_normal(dim).astype(np.float64)
            v /= np.linalg.norm(v)
            cache[path] = v
        return cache[path]

    def embedding_fn(path: str) -> np.ndarray:
        return _get(path)

    return embedding_fn, cache


# ===========================================================================
# Test 1: EER = 0 when scores are perfectly separated
# ===========================================================================

class TestEerPerfectSeparation:
    """EER should be 0 when genuine scores > all impostor scores."""

    def test_eer_perfect_separation(self):
        # Genuine pairs: score = 0.9
        # Impostor pairs: score = 0.1
        n = 100
        scores = np.concatenate([
            np.full(n, 0.9),   # genuine
            np.full(n, 0.1),   # impostor
        ])
        labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

        eer, thr = _compute_eer(scores, labels)

        assert eer == pytest.approx(0.0, abs=1e-6), (
            f"EER should be 0 for perfectly separated scores, got {eer}"
        )

    def test_frr_at_far1_is_zero_when_perfectly_separated(self):
        n = 100
        scores = np.concatenate([np.full(n, 0.9), np.full(n, 0.1)])
        labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

        frr = _compute_frr_at_far(scores, labels, target_far=0.01)
        assert frr == pytest.approx(0.0, abs=1e-6), (
            f"FRR@FAR1% should be 0 for perfectly separated scores, got {frr}"
        )


# ===========================================================================
# Test 2: EER ≈ 0.5 at chance level
# ===========================================================================

class TestEerChanceLevel:
    """EER should be near 0.5 when labels are independent of scores."""

    def test_eer_chance_level(self):
        rng = np.random.default_rng(42)
        n = 2000
        scores = rng.uniform(-1.0, 1.0, size=n)
        labels = rng.integers(0, 2, size=n)

        eer, _ = _compute_eer(scores, labels)

        # At random, EER is near 0.5; allow ±0.1 tolerance for randomness
        assert 0.35 <= eer <= 0.65, (
            f"EER at chance should be near 0.5, got {eer:.4f}"
        )


# ===========================================================================
# Test 3: minDCF formula verification
# ===========================================================================

class TestMinDcfFormula:
    """Verify minDCF against a hand-computed reference value."""

    def test_mindcf_formula(self):
        """
        Hand-computed scenario:
            100 genuine pairs, all score = 0.8
            100 impostor pairs, all score = 0.2

        At threshold = 0.5:
            P_miss  = 0 (all genuine score >= 0.5)
            P_fa    = 0 (all impostor score < 0.5)

        DCF = C_miss * 0 * p_target + C_fa * 0 * (1 - p_target) = 0

        But minDCF is normalised by prior-only cost:
            prior_cost = min(C_miss * p_target, C_fa * (1 - p_target))
                       = min(10 * 0.05, 1 * 0.95)
                       = min(0.5, 0.95)
                       = 0.5

        normalised_minDCF = 0 / 0.5 = 0.0
        """
        n = 100
        scores = np.concatenate([np.full(n, 0.8), np.full(n, 0.2)])
        labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

        result = _compute_min_dcf(
            scores, labels,
            p_target=0.05,
            c_fa=1.0,
            c_miss=10.0,
        )

        assert result == pytest.approx(0.0, abs=1e-6), (
            f"minDCF should be 0 for perfectly separated scores, got {result}"
        )

    def test_mindcf_formula_partial_overlap(self):
        """
        Verify that minDCF > 0 when genuine and impostor distributions overlap.

        Construction:
            200 genuine at 0.6
            200 impostor at 0.4
        → no perfect threshold, so DCF is minimised at 0.5 with P_miss=0, P_fa=0 → 0.
        But with score = 0.5 exactly:
            200 genuine at 0.6, 200 impostor at 0.4  → still perfectly separated.
        Use overlapping distributions: 100 genuine at 0.3, 100 impostor at 0.7.
        That means every threshold that keeps FA=0 has FRR=1 (since all genuine
        score < all impostor scores). Expect minDCF > 0.
        """
        n = 100
        scores = np.concatenate([np.full(n, 0.3), np.full(n, 0.7)])
        labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
        # Genuine scores are LOWER than impostor — worst case for a cosine system
        result = _compute_min_dcf(
            scores, labels,
            p_target=0.05,
            c_fa=1.0,
            c_miss=10.0,
        )
        # minDCF should be > 0 because genuine/impostor are inverted
        assert result > 0.0, (
            f"minDCF should be > 0 for inverted score distributions, got {result}"
        )


# ===========================================================================
# Test 4: sha256 stability
# ===========================================================================

class TestTrialListSha256Stable:
    """Same trial pairs → same sha256, regardless of insertion order."""

    def test_sha256_deterministic_same_order(self):
        pairs = [
            ("enroll_a.wav", "test_b.wav", 1),
            ("enroll_c.wav", "test_d.wav", 0),
        ]
        h1 = _sha256_trial_list(pairs)
        h2 = _sha256_trial_list(pairs)
        assert h1 == h2

    def test_sha256_stable_different_order(self):
        """Insertion order should not matter — sorting makes it stable."""
        pairs_v1 = [
            ("enroll_a.wav", "test_b.wav", 1),
            ("enroll_c.wav", "test_d.wav", 0),
            ("enroll_e.wav", "test_f.wav", 1),
        ]
        pairs_v2 = [
            ("enroll_e.wav", "test_f.wav", 1),
            ("enroll_a.wav", "test_b.wav", 1),
            ("enroll_c.wav", "test_d.wav", 0),
        ]
        h1 = _sha256_trial_list(pairs_v1)
        h2 = _sha256_trial_list(pairs_v2)
        assert h1 == h2, "sha256 must be order-independent"

    def test_sha256_differs_for_different_labels(self):
        pairs_a = [("enroll_a.wav", "test_b.wav", 1)]
        pairs_b = [("enroll_a.wav", "test_b.wav", 0)]
        assert _sha256_trial_list(pairs_a) != _sha256_trial_list(pairs_b)

    def test_sha256_in_trial_list_dataclass(self):
        """TrialList.sha256 is computed deterministically."""
        pairs = [
            ("spk1/utt1.wav", "spk2/utt2.wav", 0),
            ("spk1/utt1.wav", "spk1/utt3.wav", 1),
        ]
        tl = _make_simple_trial_list(pairs)
        assert tl.sha256 == _sha256_trial_list(pairs)
        # Build a second identical TrialList and confirm same hash
        tl2 = _make_simple_trial_list(list(reversed(pairs)))
        assert tl.sha256 == tl2.sha256


# ===========================================================================
# Test 5: Freeze guard
# ===========================================================================

class TestFreezeGuard:
    """Raise RuntimeError when a test manifest is used without freeze=True."""

    def _make_test_trial_list(self) -> TrialList:
        """Trial list whose source_manifest path contains 'test'."""
        pairs = [
            ("enroll_a.wav", "test_b.wav", 1),
            ("enroll_c.wav", "test_d.wav", 0),
        ]
        return _make_simple_trial_list(
            pairs,
            source_manifest="data/manifests/test/vox1_test.csv",
        )

    def _dummy_emb_fn(self, path: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(path)) % (2**32))
        v = rng.standard_normal(192).astype(np.float64)
        v /= np.linalg.norm(v)
        return v

    def test_raises_without_freeze(self):
        """evaluate_model should raise RuntimeError for a test manifest."""
        # Ensure FREEZE_ENABLED is False
        original = toolkit.FREEZE_ENABLED
        try:
            toolkit.FREEZE_ENABLED = False
            tl = self._make_test_trial_list()
            with pytest.raises(RuntimeError, match="--freeze"):
                evaluate_model(
                    embedding_fn=self._dummy_emb_fn,
                    trial_list=tl,
                    freeze=False,
                    model_label="B1",
                )
        finally:
            toolkit.FREEZE_ENABLED = original

    def test_does_not_raise_with_freeze_arg(self):
        """evaluate_model should succeed when freeze=True is passed."""
        original = toolkit.FREEZE_ENABLED
        try:
            toolkit.FREEZE_ENABLED = False
            tl = self._make_test_trial_list()
            # Should not raise
            result = evaluate_model(
                embedding_fn=self._dummy_emb_fn,
                trial_list=tl,
                freeze=True,
                model_label="B1",
            )
            assert isinstance(result, EvalResult)
        finally:
            toolkit.FREEZE_ENABLED = original

    def test_does_not_raise_with_freeze_enabled_flag(self):
        """evaluate_model should succeed when FREEZE_ENABLED = True."""
        original = toolkit.FREEZE_ENABLED
        try:
            toolkit.FREEZE_ENABLED = True
            tl = self._make_test_trial_list()
            result = evaluate_model(
                embedding_fn=self._dummy_emb_fn,
                trial_list=tl,
                freeze=False,
                model_label="AAA-SV",
            )
            assert isinstance(result, EvalResult)
        finally:
            toolkit.FREEZE_ENABLED = original

    def test_non_test_manifest_does_not_raise(self):
        """Val manifests must NOT trigger the freeze guard."""
        original = toolkit.FREEZE_ENABLED
        try:
            toolkit.FREEZE_ENABLED = False
            pairs = [
                ("enroll_a.wav", "enroll_b.wav", 1),
                ("enroll_c.wav", "enroll_d.wav", 0),
            ]
            tl = _make_simple_trial_list(
                pairs,
                source_manifest="data/manifests/val.csv",   # no "test" in path
            )
            result = evaluate_model(
                embedding_fn=self._dummy_emb_fn,
                trial_list=tl,
                freeze=False,
                model_label="B0",
            )
            assert isinstance(result, EvalResult)
        finally:
            toolkit.FREEZE_ENABLED = original


# ===========================================================================
# Bonus: EvalResult fields
# ===========================================================================

class TestEvalResultFields:
    """Smoke-test that EvalResult has all required fields populated."""

    def _emb(self, path: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(path)) % (2**32))
        v = rng.standard_normal(192).astype(np.float64)
        v /= np.linalg.norm(v)
        return v

    def test_eval_result_has_all_fields(self):
        pairs = []
        for i in range(20):
            pairs.append((f"spk{i%5}/utt{i}.wav", f"spk{(i+1)%5}/utt{i}.wav", i % 2))
        tl = _make_simple_trial_list(pairs, source_manifest="data/manifests/val.csv")

        result = evaluate_model(
            embedding_fn=self._emb,
            trial_list=tl,
            freeze=False,
            model_label="AAA-SV",
        )

        assert isinstance(result.eer, float)
        assert isinstance(result.frr_at_far1pct, float)
        assert isinstance(result.min_dcf, float)
        assert isinstance(result.per_bucket_eer, dict)
        assert isinstance(result.bootstrap_ci, dict)
        assert "eer_ci_low" in result.bootstrap_ci
        assert "eer_ci_high" in result.bootstrap_ci
        assert "frr_ci_low" in result.bootstrap_ci
        assert "frr_ci_high" in result.bootstrap_ci
        assert isinstance(result.det_curve, list)
        assert len(result.det_curve) > 0
        assert isinstance(result.trial_list_sha256, str) and len(result.trial_list_sha256) == 64
        assert isinstance(result.run_id, str)
        assert isinstance(result.timestamp, str)
        assert result.model_label == "AAA-SV"


# ===========================================================================
# Bonus: BucketedTrialGenerator
# ===========================================================================

class TestBucketedTrialGenerator:
    """Basic smoke tests for BucketedTrialGenerator."""

    def _write_manifest(self, tmpdir: str, rows: List[dict]) -> str:
        path = os.path.join(tmpdir, "manifest.csv")
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_all_bucket_always_present(self, tmp_path):
        rows = [
            {"wav_path": "spk1/a.wav", "speaker_id": "spk1"},
            {"wav_path": "spk1/b.wav", "speaker_id": "spk1"},
            {"wav_path": "spk2/a.wav", "speaker_id": "spk2"},
            {"wav_path": "spk2/b.wav", "speaker_id": "spk2"},
        ]
        mpath = self._write_manifest(str(tmp_path), rows)
        gen = BucketedTrialGenerator(mpath)
        buckets = gen.generate_all()
        assert "all" in buckets
        tl = buckets["all"]
        assert tl.n_positive > 0
        assert tl.n_negative > 0
        assert len(tl.sha256) == 64

    def test_duration_bucket_created(self, tmp_path):
        rows = [
            {"wav_path": f"spk{i}/utt.wav", "speaker_id": f"spk{i}",
             "duration_s": "2.0"}
            for i in range(4)
        ]
        mpath = self._write_manifest(str(tmp_path), rows)
        gen = BucketedTrialGenerator(mpath)
        buckets = gen.generate_all()
        assert "duration_2s" in buckets

    def test_codec_bucket_created(self, tmp_path):
        rows = [
            {"wav_path": f"spk{i}/utt.wav", "speaker_id": f"spk{i}",
             "codec": "opus"}
            for i in range(4)
        ]
        mpath = self._write_manifest(str(tmp_path), rows)
        gen = BucketedTrialGenerator(mpath)
        buckets = gen.generate_all()
        assert "codec_opus" in buckets

    def test_manifest_hash_stable(self, tmp_path):
        rows = [
            {"wav_path": "spk1/a.wav", "speaker_id": "spk1"},
            {"wav_path": "spk2/a.wav", "speaker_id": "spk2"},
        ]
        mpath = self._write_manifest(str(tmp_path), rows)
        gen1 = BucketedTrialGenerator(mpath)
        gen2 = BucketedTrialGenerator(mpath)
        assert gen1.manifest_hash == gen2.manifest_hash
