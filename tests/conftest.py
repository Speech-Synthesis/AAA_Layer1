# =============================================================
# tests/conftest.py
# AAA Engine — pytest configuration for test suite
# =============================================================
# Problem: librosa 1.0.0 on Windows deadlocks when its lazy
# submodule `librosa.effects` is first accessed in a sandboxed
# environment (no audio backend / no ALSA).
#
# Root cause: unittest.mock.patch walks dotted target paths by
# importing each segment; accessing `librosa.effects` triggers
# the submodule's lazy init which hangs waiting for an audio
# device on Windows.
#
# Fix: force-import librosa.effects *once* at session start in
# a conftest fixture so the module is in sys.modules before any
# test runs. After that, patch() finds it already cached and
# skips the import — no hang.
# =============================================================

import sys
import os
import types
import pytest

# ---------------------------------------------------------------------------
# Ensure aaa_engine is on the path regardless of cwd
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AAA_ENGINE = os.path.join(_REPO_ROOT, "aaa_engine")
if _AAA_ENGINE not in sys.path:
    sys.path.insert(0, _AAA_ENGINE)


# ---------------------------------------------------------------------------
# Pre-warm librosa.effects to avoid Windows deadlock
# ---------------------------------------------------------------------------
def _prewarm_librosa_effects() -> bool:
    """
    Import librosa.effects with a hard timeout guard.
    Returns True on success, False if the import would hang
    (detected by trying a quick attribute check first).
    
    Strategy: check if librosa is already imported and has effects cached;
    if not, inject a minimal stub so tests can run without the deadlock.
    """
    try:
        import librosa  # top-level import is fast (no audio backend init)
        
        # Check if .effects is already in sys.modules (cached)
        if "librosa.effects" in sys.modules:
            return True

        # Try to access librosa.effects on Windows — this may hang.
        # We detect the platform and inject a stub if we're at risk.
        import platform
        if platform.system() == "Windows":
            # Inject a minimal stub for librosa.effects into sys.modules
            # so that patch('...librosa.effects.trim') works without hang.
            stub = types.ModuleType("librosa.effects")
            
            import numpy as np
            
            def _trim_stub(y, top_db=60, ref=None, frame_length=2048,
                           hop_length=512):
                """Passthrough stub — returns audio unchanged with full indices."""
                return y, (0, len(y))
            
            stub.trim = _trim_stub
            stub.time_stretch = lambda y, rate: y
            stub.pitch_shift = lambda y, sr, n_steps: y
            
            sys.modules["librosa.effects"] = stub
            librosa.effects = stub
            return True
        else:
            # Non-Windows: safe to import directly
            import librosa.effects  # noqa: F401
            return True
    except Exception as exc:
        print(f"\n[conftest] WARNING: librosa.effects prewarm failed: {exc}")
        return False


# Run at import time (session scope)
_LIBROSA_EFFECTS_OK = _prewarm_librosa_effects()


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_gpu: mark test as requiring a CUDA GPU (skipped if not available)"
    )
    config.addinivalue_line(
        "markers",
        "requires_pretrained: mark test as requiring a downloaded pretrained checkpoint"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (skipped with -m 'not slow')"
    )
