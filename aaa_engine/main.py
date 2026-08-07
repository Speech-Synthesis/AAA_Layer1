# =============================================================
# main.py
# AAA Engine — FastAPI Application
# API endpoints for Layer 1: Speaker Verification
# =============================================================
# Endpoints:
#   POST /auth/enroll  — One-time voice enrollment
#   POST /auth/verify  — Speaker verification (Layer 1 gate)
# =============================================================

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from layer1.enrollment import enroll_user
from layer1.speaker_verifier import verify_speaker
from layer1.db_utils import fetch_voiceprint, log_auth_result
import uuid

# ── FastAPI Application ─────────────────────────────────────
app = FastAPI(title='AAA Engine')


# ── POST /auth/enroll ───────────────────────────────────────
@app.post('/auth/enroll')
async def enroll(
    user_id: str = Form(...),
    audio: UploadFile = File(...)
):
    """
    Enroll a new user's voiceprint.

    Accepts:
        - user_id: Unique user identifier (UUID string)
        - audio:   10–15 seconds of enrollment audio (WAV/FLAC/MP3)

    Returns:
        { success: bool, message: str, user_id: str }

    Raises:
        400: If user is already enrolled or audio is invalid.
    """
    audio_bytes = await audio.read()
    result = enroll_user(user_id, audio_bytes)
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['message'])
    return result


# ── POST /auth/verify ───────────────────────────────────────
@app.post('/auth/verify')
async def verify(
    user_id: str = Form(...),
    session_token: str = Form(...),
    audio: UploadFile = File(...)
):
    """
    Verify a user's voice against their enrolled voiceprint.

    Pipeline:
        1. [Layer 2 Anti-Spoofing runs here — placeholder]
        2. Layer 1: Extract live embedding → cosine similarity → ACCEPT/REJECT
        3. Log the attempt to auth_log
        4. [Layer 3 Voiceprint Update runs here — placeholder]

    Accepts:
        - user_id:       The enrolled user to verify against
        - session_token: Authentication session token
        - audio:         1.5–10 seconds of live voice (WAV/FLAC/MP3)

    Returns:
        {
            result:        'ACCEPT' | 'REJECT',
            confidence:    float (cosine similarity 0.0–1.0),
            layer_blocked: 1 | None,
            session_id:    str (UUID)
        }
    """
    session_id = str(uuid.uuid4())
    audio_bytes = await audio.read()

    # --- [Layer 2 Anti-Spoofing runs here — M5's code] ---
    # Assume bonafide for Layer 1 integration

    # --- LAYER 1: Speaker Verification ---
    stored_embedding = fetch_voiceprint(user_id)
    l1_result = verify_speaker(user_id, audio_bytes, stored_embedding)

    # Log the attempt
    log_auth_result(
        session_id=session_id,
        user_id=user_id,
        result=l1_result['result'],
        l1_score=l1_result['l1_score'],
        layer_blocked=l1_result['layer_blocked']
    )

    if l1_result['result'] == 'REJECT':
        return {
            'result': 'REJECT',
            'confidence': l1_result['l1_score'],
            'layer_blocked': 1,
            'session_id': session_id
        }

    # --- [Layer 3 Voiceprint Update runs here — M2's next task] ---

    return {
        'result': 'ACCEPT',
        'confidence': l1_result['l1_score'],
        'layer_blocked': None,
        'session_id': session_id
    }
