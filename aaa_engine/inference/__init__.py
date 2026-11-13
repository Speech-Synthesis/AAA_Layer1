# =============================================================
# aaa_engine/inference/__init__.py
# AAA Engine – Inference package
# =============================================================

from .aaasv import (
    AAASV,
    AAASVResponse,
    EmbedResult,
    Voiceprint,
    VerifyResult,
    get_instance,
    # Error codes
    E_FORMAT,
    E_DECODE,
    E_TOO_SHORT,
    E_TOO_LONG,
    E_SILENT,
    E_NONFINITE,
    E_NO_ENROLLMENT,
    E_MODEL_UNAVAILABLE,
    E_VERSION_MISMATCH,
    E_INTERNAL,
)

__all__ = [
    "AAASV",
    "AAASVResponse",
    "EmbedResult",
    "Voiceprint",
    "VerifyResult",
    "get_instance",
    "E_FORMAT",
    "E_DECODE",
    "E_TOO_SHORT",
    "E_TOO_LONG",
    "E_SILENT",
    "E_NONFINITE",
    "E_NO_ENROLLMENT",
    "E_MODEL_UNAVAILABLE",
    "E_VERSION_MISMATCH",
    "E_INTERNAL",
]
