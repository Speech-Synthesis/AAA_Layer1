# =============================================================
# layer1/speaker_verifier.py
# AAA Engine — Layer 1: Speaker Verification
# CORE MODULE: ECAPA-TDNN embedding extraction & cosine verification
# =============================================================
# This is the heart of Layer 1. It handles:
#   - ECAPA-TDNN model loading (singleton pattern)
#   - Audio preprocessing (resample, trim silence, normalize)
#   - 192-dim embedding extraction
#   - Cosine similarity computation
#   - Final ACCEPT/REJECT decision (threshold = 0.75)
# =============================================================

import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from speechbrain.pretrained import SpeakerRecognition
from scipy.spatial.distance import cosine
from typing import Optional, Tuple, Dict
import io

# ── Constants ────────────────────────────────────────────────
SAMPLE_RATE     = 16000      # Required: 16kHz mono
THRESHOLD       = 0.75       # Cosine similarity threshold (from spec)
MIN_DURATION    = 1.5        # Minimum audio duration in seconds
MAX_DURATION    = 10.0       # Maximum audio duration in seconds
import os
MODEL_SOURCE    = os.environ.get(
    'ECAPA_MODEL_PATH',
    'speechbrain/spkrec-ecapa-voxceleb'            # Default: pre-trained (for dev/testing)
    # In production, set ECAPA_MODEL_PATH=pretrained_models/ecapa-tdnn-custom
)
SAVEDIR         = os.environ.get(
    'ECAPA_SAVE_DIR',
    'pretrained_models/spkrec-ecapa-voxceleb'
)

# ── Model Singleton ─────────────────────────────────────────
_model = None


def load_model() -> SpeakerRecognition:
    """
    Load ECAPA-TDNN model once and cache it (singleton pattern).
    Model was custom-trained on our 80GB voice dataset.
    Subsequent calls return the cached model instantly.
    """
    global _model
    if _model is None:
        print("[Layer1] Loading ECAPA-TDNN model...")
        _model = SpeakerRecognition.from_hparams(
            source=MODEL_SOURCE,
            savedir=SAVEDIR,
            run_opts={'device': 'cpu'}  # change to 'cuda' if GPU available
        )
        print("[Layer1] Model loaded successfully.")
    return _model


# ── Audio Preprocessing ──────────────────────────────────────
def preprocess_audio(audio_bytes: bytes) -> Tuple[torch.Tensor, float]:
    """
    Convert raw audio bytes to 16kHz mono tensor.

    Processing steps:
        1. Load audio from bytes (supports WAV, FLAC, MP3)
        2. Convert stereo to mono (if needed)
        3. Resample to 16kHz (if needed)
        4. Trim silence using librosa (top_db=25)
        5. Validate duration (1.5s – 10.0s)

    Returns:
        (waveform_tensor, duration_seconds)

    Raises:
        ValueError: If audio duration is outside the valid range.
    """
    # Load audio from bytes
    audio_io = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(audio_io)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Trim silence using librosa
    audio_np = waveform.squeeze().numpy()
    trimmed, _ = librosa.effects.trim(audio_np, top_db=25)
    waveform = torch.tensor(trimmed).unsqueeze(0)

    # Duration check
    duration = waveform.shape[1] / SAMPLE_RATE
    if duration < MIN_DURATION:
        raise ValueError(f"Audio too short: {duration:.2f}s (min {MIN_DURATION}s)")
    if duration > MAX_DURATION:
        raise ValueError(f"Audio too long: {duration:.2f}s (max {MAX_DURATION}s)")

    return waveform, duration


# ── Embedding Extraction ────────────────────────────────────
def extract_embedding(audio_bytes: bytes) -> np.ndarray:
    """
    Extract 192-dim ECAPA-TDNN voice embedding from audio.

    Steps:
        1. Preprocess audio (resample, trim, validate)
        2. Run ECAPA-TDNN inference (no gradient computation)
        3. Flatten output from (1, 1, 192) to (192,)
        4. L2-normalize the embedding vector

    Returns:
        numpy array of shape (192,) — the voice fingerprint.
    """
    model = load_model()
    waveform, duration = preprocess_audio(audio_bytes)

    with torch.no_grad():
        # SpeechBrain encode_batch returns embeddings
        embedding = model.encode_batch(waveform)

    # Shape: (1, 1, 192) → flatten to (192,)
    embedding_np = embedding.squeeze().numpy()

    # L2 normalize the embedding
    norm = np.linalg.norm(embedding_np)
    if norm > 0:
        embedding_np = embedding_np / norm

    return embedding_np  # shape: (192,)


# ── Cosine Similarity ───────────────────────────────────────
def compute_cosine_similarity(
    embedding_live: np.ndarray,
    embedding_stored: np.ndarray
) -> float:
    """
    Compute cosine similarity between two 192-dim embeddings.

    Uses scipy's cosine distance (1 - similarity), then converts back.
    Result is clamped to [0.0, 1.0] to handle floating point edge cases.

    Args:
        embedding_live:   192-dim vector from live audio
        embedding_stored: 192-dim vector from enrolled voiceprint

    Returns:
        Similarity score between 0.0 (different) and 1.0 (identical).
    """
    # scipy cosine() returns cosine DISTANCE (1 - similarity)
    distance = cosine(embedding_live, embedding_stored)
    similarity = 1.0 - distance
    # Clamp to [0, 1] to handle floating point edge cases
    return float(np.clip(similarity, 0.0, 1.0))


# ── Main Verification Function ──────────────────────────────
def verify_speaker(
    user_id: str,
    audio_bytes: bytes,
    stored_embedding: Optional[np.ndarray],
    threshold: float = THRESHOLD
) -> Dict:
    """
    Full Layer 1 verification pipeline.

    Steps:
        1. Check if user is enrolled (stored_embedding exists)
        2. Extract live 192-dim embedding from audio
        3. Compute cosine similarity against stored embedding
        4. Apply threshold decision: >= 0.75 → ACCEPT, < 0.75 → REJECT

    Args:
        user_id:          The user being authenticated
        audio_bytes:      Raw audio bytes (WAV/FLAC/MP3)
        stored_embedding: 192-dim numpy array from DB (None if not enrolled)
        threshold:        Cosine similarity threshold (default 0.75)

    Returns dict:
        {
          result:           'ACCEPT' | 'REJECT',
          l1_score:         float (0.0–1.0),
          live_embedding:   np.ndarray (192,) | None,
          layer_blocked:    1 | None,
          error:            str | None
        }
    """
    # Case: User not enrolled
    if stored_embedding is None:
        return {
            'result': 'REJECT',
            'l1_score': 0.0,
            'live_embedding': None,
            'layer_blocked': 1,
            'error': f'User {user_id} not enrolled'
        }

    try:
        # Step 1: Extract live embedding
        live_embedding = extract_embedding(audio_bytes)

        # Step 2: Compute cosine similarity
        score = compute_cosine_similarity(live_embedding, stored_embedding)

        # Step 3: Threshold decision
        if score >= threshold:
            return {
                'result': 'ACCEPT',
                'l1_score': score,
                'live_embedding': live_embedding,
                'layer_blocked': None,
                'error': None
            }
        else:
            return {
                'result': 'REJECT',
                'l1_score': score,
                'live_embedding': live_embedding,
                'layer_blocked': 1,
                'error': None
            }

    except ValueError as e:
        return {
            'result': 'REJECT',
            'l1_score': 0.0,
            'live_embedding': None,
            'layer_blocked': 1,
            'error': str(e)
        }
