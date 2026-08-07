# =============================================================
# AAA Engine — Layer 1: Speaker Verification
# Package Initialization
# =============================================================
# Layer 1 is the Speaker Verification gate of the AAA Engine.
# It converts voice audio into 192-dim ECAPA-TDNN embeddings
# and compares them using cosine similarity (threshold = 0.75).
# =============================================================

from layer1.speaker_verifier import (
    extract_embedding,
    compute_cosine_similarity,
    verify_speaker,
    load_model,
    preprocess_audio,
)
from layer1.enrollment import enroll_user, re_enroll_user
from layer1.db_utils import (
    store_voiceprint,
    fetch_voiceprint,
    log_auth_result,
)

__all__ = [
    "extract_embedding",
    "compute_cosine_similarity",
    "verify_speaker",
    "load_model",
    "preprocess_audio",
    "enroll_user",
    "re_enroll_user",
    "store_voiceprint",
    "fetch_voiceprint",
    "log_auth_result",
]
