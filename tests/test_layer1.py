# =============================================================
# tests/test_layer1.py
# AAA Engine — Layer 1: Speaker Verification Unit Tests
# =============================================================
# Comprehensive smoke tests covering all Layer 1 scenarios:
#   - Cosine similarity math (same, different, orthogonal, negated)
#   - Threshold boundary conditions (>= 0.75 ACCEPT, < 0.75 REJECT)
#   - Verify pipeline (unenrolled, accept, reject, error handling)
#   - Embedding properties (dimensionality, L2 normalization)
#   - Enrollment flow (duplicate rejection, DB mock)
# =============================================================

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from layer1.speaker_verifier import (
    extract_embedding, compute_cosine_similarity, verify_speaker,
    THRESHOLD, MIN_DURATION, MAX_DURATION
)
from layer1.enrollment import enroll_user, re_enroll_user


# ═══════════════════════════════════════════════════════════════
#  COSINE SIMILARITY TESTS
# ═══════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    """Tests for compute_cosine_similarity — the math core of L1."""

    def test_same_vector_returns_one(self):
        """Same vector → similarity must be 1.0"""
        v = np.random.rand(192).astype(np.float32)
        v = v / np.linalg.norm(v)
        score = compute_cosine_similarity(v, v)
        assert abs(score - 1.0) < 1e-5, f'Expected 1.0, got {score}'

    def test_different_vectors_below_threshold(self):
        """Random vectors → similarity should be well below 0.75"""
        np.random.seed(42)  # Fix seed for determinism
        v1 = np.random.rand(192).astype(np.float32)
        v2 = np.random.rand(192).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        score = compute_cosine_similarity(v1, v2)
        assert score < 0.75, f'Expected < 0.75, got {score}'

    def test_orthogonal_vectors_near_zero(self):
        """Orthogonal (perpendicular) vectors → similarity ≈ 0.0"""
        v1 = np.zeros(192, dtype=np.float32)
        v2 = np.zeros(192, dtype=np.float32)
        v1[0] = 1.0   # Unit vector along dim 0
        v2[1] = 1.0   # Unit vector along dim 1
        score = compute_cosine_similarity(v1, v2)
        assert abs(score) < 1e-5, f'Expected ~0.0, got {score}'

    def test_negated_vector_clamped_to_zero(self):
        """Opposite vector → cosine = -1.0, but clamped to 0.0"""
        v = np.random.rand(192).astype(np.float32)
        v = v / np.linalg.norm(v)
        score = compute_cosine_similarity(v, -v)
        assert score == 0.0, f'Expected 0.0 (clamped), got {score}'

    def test_score_always_between_zero_and_one(self):
        """Result must always be in [0.0, 1.0] range"""
        for _ in range(10):
            v1 = np.random.randn(192).astype(np.float32)
            v2 = np.random.randn(192).astype(np.float32)
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)
            score = compute_cosine_similarity(v1, v2)
            assert 0.0 <= score <= 1.0, f'Score {score} outside [0, 1]'

    def test_slightly_perturbed_vector_high_similarity(self):
        """Same person, small noise → score should remain high (> 0.9)"""
        v = np.random.rand(192).astype(np.float32)
        v = v / np.linalg.norm(v)
        noise = np.random.randn(192).astype(np.float32) * 0.01
        v_noisy = v + noise
        v_noisy = v_noisy / np.linalg.norm(v_noisy)
        score = compute_cosine_similarity(v, v_noisy)
        assert score > 0.9, f'Expected > 0.9 for small perturbation, got {score}'


# ═══════════════════════════════════════════════════════════════
#  THRESHOLD / DECISION BOUNDARY TESTS
# ═══════════════════════════════════════════════════════════════

class TestThresholdDecision:
    """Tests for the >= 0.75 ACCEPT / < 0.75 REJECT decision."""

    def _make_vectors_with_similarity(self, target_sim: float):
        """
        Create two 192-dim L2-normalized vectors with an approximate
        target cosine similarity. Uses interpolation trick:
            v2 = target * v1 + sqrt(1 - target^2) * orthogonal
        """
        np.random.seed(0)
        v1 = np.random.randn(192).astype(np.float64)
        v1 = v1 / np.linalg.norm(v1)

        # Create orthogonal vector
        orth = np.random.randn(192).astype(np.float64)
        orth = orth - np.dot(orth, v1) * v1  # Gram-Schmidt
        orth = orth / np.linalg.norm(orth)

        # Interpolate
        v2 = target_sim * v1 + np.sqrt(1.0 - target_sim ** 2) * orth
        v2 = v2 / np.linalg.norm(v2)

        return v1.astype(np.float32), v2.astype(np.float32)

    def test_accept_when_score_above_threshold(self):
        """Mock: score = 0.85 → result must be ACCEPT"""
        stored = np.random.rand(192).astype(np.float32)
        stored = stored / np.linalg.norm(stored)

        # Create a live embedding that will have 0.85 similarity
        live, stored = self._make_vectors_with_similarity(0.85)

        with patch('layer1.speaker_verifier.extract_embedding', return_value=live):
            result = verify_speaker('user-1', b'dummy', stored)
            assert result['result'] == 'ACCEPT'
            assert result['l1_score'] >= THRESHOLD
            assert result['layer_blocked'] is None
            assert result['error'] is None

    def test_reject_when_score_below_threshold(self):
        """Mock: score = 0.60 → result must be REJECT"""
        live, stored = self._make_vectors_with_similarity(0.60)

        with patch('layer1.speaker_verifier.extract_embedding', return_value=live):
            result = verify_speaker('user-2', b'dummy', stored)
            assert result['result'] == 'REJECT'
            assert result['l1_score'] < THRESHOLD
            assert result['layer_blocked'] == 1
            assert result['error'] is None

    def test_exact_threshold_is_accept(self):
        """Boundary: score = exactly 0.75 → ACCEPT (>= threshold)"""
        live, stored = self._make_vectors_with_similarity(0.75)

        with patch('layer1.speaker_verifier.extract_embedding', return_value=live):
            result = verify_speaker('user-3', b'dummy', stored)
            # Score should be very close to 0.75
            assert abs(result['l1_score'] - 0.75) < 0.01
            assert result['result'] == 'ACCEPT'

    def test_just_below_threshold_is_reject(self):
        """Boundary: score ≈ 0.74 → REJECT (< threshold)"""
        live, stored = self._make_vectors_with_similarity(0.74)

        with patch('layer1.speaker_verifier.extract_embedding', return_value=live):
            result = verify_speaker('user-4', b'dummy', stored)
            assert result['l1_score'] < THRESHOLD
            assert result['result'] == 'REJECT'


# ═══════════════════════════════════════════════════════════════
#  VERIFY_SPEAKER PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════

class TestVerifySpeaker:
    """Tests for the full verify_speaker() pipeline."""

    def test_reject_unenrolled_user(self):
        """Unregistered user (stored_embedding=None) → REJECT"""
        result = verify_speaker('unknown-user', b'', None)
        assert result['result'] == 'REJECT'
        assert result['layer_blocked'] == 1
        assert result['l1_score'] == 0.0
        assert 'not enrolled' in result['error']
        assert result['live_embedding'] is None

    def test_error_handling_returns_reject(self):
        """When extract_embedding raises ValueError → REJECT with error"""
        stored = np.random.rand(192).astype(np.float32)
        error_msg = 'Audio too short: 0.50s (min 1.5s)'

        with patch('layer1.speaker_verifier.extract_embedding',
                   side_effect=ValueError(error_msg)):
            result = verify_speaker('user-err', b'short_audio', stored)
            assert result['result'] == 'REJECT'
            assert result['layer_blocked'] == 1
            assert result['l1_score'] == 0.0
            assert result['error'] == error_msg
            assert result['live_embedding'] is None

    def test_response_keys_present_on_accept(self):
        """ACCEPT response must have all required keys"""
        live = np.random.rand(192).astype(np.float32)
        live = live / np.linalg.norm(live)
        stored = live.copy()  # Same vector → will ACCEPT

        with patch('layer1.speaker_verifier.extract_embedding', return_value=live):
            result = verify_speaker('user-keys', b'dummy', stored)
            required_keys = {'result', 'l1_score', 'live_embedding',
                             'layer_blocked', 'error'}
            assert set(result.keys()) == required_keys

    def test_response_keys_present_on_reject(self):
        """REJECT response must have all required keys"""
        result = verify_speaker('no-user', b'', None)
        required_keys = {'result', 'l1_score', 'live_embedding',
                         'layer_blocked', 'error'}
        assert set(result.keys()) == required_keys

    def test_custom_threshold(self):
        """verify_speaker accepts a custom threshold parameter"""
        live = np.random.rand(192).astype(np.float32)
        live = live / np.linalg.norm(live)

        # Create a stored embedding with known ~0.60 similarity
        np.random.seed(99)
        stored = np.random.rand(192).astype(np.float32)
        stored = stored / np.linalg.norm(stored)

        with patch('layer1.speaker_verifier.extract_embedding', return_value=live):
            # Default threshold 0.75 → likely REJECT for random vectors
            result_strict = verify_speaker('user-t', b'dummy', stored, threshold=0.99)
            assert result_strict['result'] == 'REJECT'

            # Very low threshold → ACCEPT
            result_lax = verify_speaker('user-t', b'dummy', stored, threshold=0.01)
            assert result_lax['result'] == 'ACCEPT'


# ═══════════════════════════════════════════════════════════════
#  EMBEDDING PROPERTY TESTS
# ═══════════════════════════════════════════════════════════════

class TestEmbeddingProperties:
    """Tests for embedding extraction output properties (mocked)."""

    def test_embedding_is_192_dim(self):
        """Extracted embedding must be exactly 192 dimensions"""
        fake_emb = np.random.rand(192).astype(np.float32)
        fake_emb = fake_emb / np.linalg.norm(fake_emb)

        with patch('layer1.speaker_verifier.load_model') as mock_model, \
             patch('layer1.speaker_verifier.preprocess_audio') as mock_pre:
            # Mock preprocessing
            mock_pre.return_value = (MagicMock(), 3.0)
            # Mock model — encode_batch returns (1, 1, 192) tensor
            import torch
            mock_model.return_value.encode_batch.return_value = \
                torch.tensor(fake_emb).unsqueeze(0).unsqueeze(0)

            result = extract_embedding(b'dummy_audio')
            assert result.shape == (192,), f'Expected (192,), got {result.shape}'

    def test_embedding_is_l2_normalized(self):
        """Extracted embedding must have L2 norm ≈ 1.0"""
        fake_emb = np.array([3.0, 4.0] + [0.0] * 190, dtype=np.float32)  # Not normalized

        with patch('layer1.speaker_verifier.load_model') as mock_model, \
             patch('layer1.speaker_verifier.preprocess_audio') as mock_pre:
            mock_pre.return_value = (MagicMock(), 3.0)
            import torch
            mock_model.return_value.encode_batch.return_value = \
                torch.tensor(fake_emb).unsqueeze(0).unsqueeze(0)

            result = extract_embedding(b'dummy_audio')
            norm = np.linalg.norm(result)
            assert abs(norm - 1.0) < 1e-5, f'Expected norm ~1.0, got {norm}'


# ═══════════════════════════════════════════════════════════════
#  ENROLLMENT FLOW TESTS
# ═══════════════════════════════════════════════════════════════

class TestEnrollment:
    """Tests for enroll_user and re_enroll_user."""

    def test_reject_already_enrolled_user(self):
        """If user already has a voiceprint, enroll_user must reject"""
        existing_emb = np.random.rand(192).astype(np.float32)

        with patch('layer1.enrollment.fetch_voiceprint', return_value=existing_emb):
            result = enroll_user('existing-user', b'audio_data')
            assert result['success'] is False
            assert 'already enrolled' in result['message']
            assert result['user_id'] == 'existing-user'

    def test_successful_enrollment(self):
        """New user with valid audio → success"""
        fake_emb = np.random.rand(192).astype(np.float32)

        with patch('layer1.enrollment.fetch_voiceprint', return_value=None), \
             patch('layer1.enrollment.extract_embedding', return_value=fake_emb), \
             patch('layer1.enrollment.store_voiceprint', return_value=True):
            result = enroll_user('new-user', b'audio_data')
            assert result['success'] is True
            assert 'enrolled successfully' in result['message']

    def test_enrollment_db_failure(self):
        """DB write fails → success=False"""
        fake_emb = np.random.rand(192).astype(np.float32)

        with patch('layer1.enrollment.fetch_voiceprint', return_value=None), \
             patch('layer1.enrollment.extract_embedding', return_value=fake_emb), \
             patch('layer1.enrollment.store_voiceprint', return_value=False):
            result = enroll_user('db-fail-user', b'audio_data')
            assert result['success'] is False
            assert 'failed' in result['message'].lower()

    def test_enrollment_audio_error(self):
        """Invalid audio → success=False with audio error"""
        with patch('layer1.enrollment.fetch_voiceprint', return_value=None), \
             patch('layer1.enrollment.extract_embedding',
                   side_effect=ValueError('Audio too short: 0.5s (min 1.5s)')):
            result = enroll_user('bad-audio-user', b'short')
            assert result['success'] is False
            assert 'Audio error' in result['message']

    def test_re_enrollment_success(self):
        """re_enroll_user overwrites existing voiceprint"""
        fake_emb = np.random.rand(192).astype(np.float32)

        with patch('layer1.enrollment.extract_embedding', return_value=fake_emb), \
             patch('layer1.enrollment.store_voiceprint', return_value=True):
            result = re_enroll_user('existing-user', b'audio_data')
            assert result['success'] is True
            assert 're-enrolled' in result['message']


# ═══════════════════════════════════════════════════════════════
#  AUDIO PREPROCESSING VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestAudioValidation:
    """Tests for audio duration validation in preprocess_audio."""

    def test_audio_too_short_raises_valueerror(self):
        """Audio < 1.5s → ValueError"""
        import torch
        from layer1.speaker_verifier import preprocess_audio

        # Create a very short WAV-like audio in-memory (0.5 seconds at 16kHz)
        short_samples = int(0.5 * 16000)
        waveform = torch.zeros(1, short_samples)

        with patch('layer1.speaker_verifier.torchaudio.load',
                   return_value=(waveform, 16000)), \
             patch('layer1.speaker_verifier.librosa.effects.trim',
                   return_value=(waveform.squeeze().numpy(), None)):
            with pytest.raises(ValueError, match='too short'):
                preprocess_audio(b'short_audio_bytes')

    def test_audio_too_long_raises_valueerror(self):
        """Audio > 10.0s → ValueError"""
        import torch
        from layer1.speaker_verifier import preprocess_audio

        # Create a long audio (15 seconds at 16kHz)
        long_samples = int(15.0 * 16000)
        waveform = torch.randn(1, long_samples)

        with patch('layer1.speaker_verifier.torchaudio.load',
                   return_value=(waveform, 16000)), \
             patch('layer1.speaker_verifier.librosa.effects.trim',
                   return_value=(waveform.squeeze().numpy(), None)):
            with pytest.raises(ValueError, match='too long'):
                preprocess_audio(b'long_audio_bytes')

    def test_valid_duration_passes(self):
        """Audio of 3.0s → no error"""
        import torch
        from layer1.speaker_verifier import preprocess_audio

        valid_samples = int(3.0 * 16000)
        waveform = torch.randn(1, valid_samples)

        with patch('layer1.speaker_verifier.torchaudio.load',
                   return_value=(waveform, 16000)), \
             patch('layer1.speaker_verifier.librosa.effects.trim',
                   return_value=(waveform.squeeze().numpy(), None)):
            result_waveform, duration = preprocess_audio(b'valid_audio')
            assert abs(duration - 3.0) < 0.01

    def test_stereo_to_mono_conversion(self):
        """Stereo input (2 channels) → converted to mono (1 channel)"""
        import torch
        from layer1.speaker_verifier import preprocess_audio

        valid_samples = int(3.0 * 16000)
        stereo_waveform = torch.randn(2, valid_samples)  # 2 channels
        mono_waveform = stereo_waveform.mean(dim=0, keepdim=True)

        with patch('layer1.speaker_verifier.torchaudio.load',
                   return_value=(stereo_waveform, 16000)), \
             patch('layer1.speaker_verifier.librosa.effects.trim',
                   return_value=(mono_waveform.squeeze().numpy(), None)):
            result_waveform, duration = preprocess_audio(b'stereo_audio')
            assert result_waveform.shape[0] == 1, 'Output must be mono (1 channel)'

    def test_resample_to_16khz(self):
        """Non-16kHz audio → resampled to 16kHz"""
        import torch
        from layer1.speaker_verifier import preprocess_audio

        # Input at 44.1kHz
        sr_input = 44100
        duration_sec = 3.0
        samples = int(duration_sec * sr_input)
        waveform = torch.randn(1, samples)

        # After resampling, samples should be ~3.0 * 16000
        expected_samples = int(duration_sec * 16000)
        resampled = torch.randn(1, expected_samples)

        with patch('layer1.speaker_verifier.torchaudio.load',
                   return_value=(waveform, sr_input)), \
             patch('layer1.speaker_verifier.torchaudio.transforms.Resample') as mock_resample, \
             patch('layer1.speaker_verifier.librosa.effects.trim',
                   return_value=(resampled.squeeze().numpy(), None)):
            mock_resample.return_value = lambda x: resampled
            result_waveform, duration = preprocess_audio(b'hifi_audio')
            mock_resample.assert_called_once_with(sr_input, 16000)


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestConstants:
    """Verify spec-mandated constants are correctly set."""

    def test_threshold_is_075(self):
        """Threshold must be 0.75 per architecture spec"""
        assert THRESHOLD == 0.75

    def test_min_duration(self):
        """Minimum audio duration must be 1.5 seconds"""
        assert MIN_DURATION == 1.5

    def test_max_duration(self):
        """Maximum audio duration must be 10.0 seconds"""
        assert MAX_DURATION == 10.0


# ═══════════════════════════════════════════════════════════════
#  DB UTILS TESTS (mocked — no real DB needed)
# ═══════════════════════════════════════════════════════════════

class TestDbUtils:
    """Tests for db_utils.py with mocked database connections."""

    def test_store_voiceprint_computes_sha256_hash(self):
        """store_voiceprint must compute SHA256 of embedding"""
        import hashlib, json
        from layer1.db_utils import store_voiceprint

        emb = np.array([0.1, 0.2, 0.3] + [0.0] * 189, dtype=np.float32)
        expected_hash = hashlib.sha256(
            json.dumps(emb.tolist()).encode()
        ).hexdigest()

        with patch('layer1.db_utils.get_connection') as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.cursor.return_value = mock_cursor

            result = store_voiceprint('test-user', emb)
            assert result is True

            # Verify the hash was computed correctly
            call_args = mock_cursor.execute.call_args[0]
            assert expected_hash == call_args[1][2]  # 3rd param is emb_hash

    def test_fetch_voiceprint_returns_numpy_array(self):
        """fetch_voiceprint returns np.ndarray of float32"""
        from layer1.db_utils import fetch_voiceprint

        fake_row = ([0.1] * 192,)

        with patch('layer1.db_utils.get_connection') as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = fake_row

            result = fetch_voiceprint('test-user')
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
            assert result.shape == (192,)

    def test_fetch_voiceprint_returns_none_when_not_found(self):
        """User not in DB → returns None"""
        from layer1.db_utils import fetch_voiceprint

        with patch('layer1.db_utils.get_connection') as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.cursor.return_value = mock_cursor
            mock_cursor.fetchone.return_value = None

            result = fetch_voiceprint('nonexistent-user')
            assert result is None

    def test_log_auth_result_inserts_record(self):
        """log_auth_result must INSERT into auth_log"""
        from layer1.db_utils import log_auth_result

        with patch('layer1.db_utils.get_connection') as mock_conn:
            mock_cursor = MagicMock()
            mock_conn.return_value.cursor.return_value = mock_cursor

            log_auth_result(
                session_id='sess-123',
                user_id='user-456',
                result='ACCEPT',
                l1_score=0.88,
                layer_blocked=None
            )

            mock_cursor.execute.assert_called_once()
            call_args = mock_cursor.execute.call_args[0]
            assert 'INSERT INTO auth_log' in call_args[0]
            assert call_args[1] == ('sess-123', 'user-456', 'ACCEPT', None, 0.88)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
