# =============================================================
# layer1/enrollment.py
# AAA Engine — Layer 1: Speaker Verification
# User enrollment and re-enrollment flows
# =============================================================
# Enrollment is a one-time process per user. It captures the
# user's voice, extracts the ECAPA-TDNN embedding, and stores
# it in PostgreSQL as their permanent voiceprint reference.
# =============================================================

import numpy as np
from layer1.speaker_verifier import extract_embedding
from layer1.db_utils import store_voiceprint, fetch_voiceprint
from typing import Dict

# ── Enrollment Duration Constraints ─────────────────────────
MIN_ENROLL_DURATION = 10.0  # Minimum 10 seconds for enrollment
MAX_ENROLL_DURATION = 15.0  # Maximum 15 seconds for enrollment


def enroll_user(user_id: str, audio_bytes: bytes) -> Dict:
    """
    One-time enrollment: extract embedding and store in DB.

    Steps:
        1. Check if user is already enrolled (reject if so)
        2. Extract 192-dim ECAPA-TDNN embedding from audio
        3. Store embedding + SHA256 hash in PostgreSQL

    Args:
        user_id:     Unique user identifier (UUID string)
        audio_bytes: 10–15 seconds of enrollment audio

    Returns dict:
        { success: bool, message: str, user_id: str }
    """
    # Check: already enrolled?
    existing = fetch_voiceprint(user_id)
    if existing is not None:
        return {
            'success': False,
            'message': f'User {user_id} is already enrolled. Use re-enrollment endpoint.',
            'user_id': user_id
        }

    try:
        # Extract 192-dim ECAPA-TDNN embedding
        embedding = extract_embedding(audio_bytes)

        # Store in PostgreSQL
        stored = store_voiceprint(user_id, embedding)

        if stored:
            return {
                'success': True,
                'message': f'User {user_id} enrolled successfully.',
                'user_id': user_id
            }
        else:
            return {
                'success': False,
                'message': 'Database write failed. Try again.',
                'user_id': user_id
            }

    except ValueError as e:
        return {
            'success': False,
            'message': f'Audio error: {str(e)}',
            'user_id': user_id
        }


def re_enroll_user(user_id: str, audio_bytes: bytes) -> Dict:
    """
    Force re-enrollment — overwrites existing voiceprint.
    Use when user wants to reset their voiceprint.

    The store_voiceprint function uses ON CONFLICT DO UPDATE,
    so this will overwrite the existing embedding and hash.

    Args:
        user_id:     Unique user identifier (UUID string)
        audio_bytes: 10–15 seconds of new enrollment audio

    Returns dict:
        { success: bool, message: str, user_id: str }
    """
    try:
        embedding = extract_embedding(audio_bytes)
        stored = store_voiceprint(user_id, embedding)  # ON CONFLICT DO UPDATE
        if stored:
            return {'success': True, 'message': f'User {user_id} re-enrolled.', 'user_id': user_id}
        return {'success': False, 'message': 'DB write failed.', 'user_id': user_id}
    except ValueError as e:
        return {'success': False, 'message': str(e), 'user_id': user_id}
