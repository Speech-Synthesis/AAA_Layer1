# =============================================================
# aaa_engine/calibration/__init__.py
# AAA Engine – Calibration sub-package
# =============================================================
"""
Calibration sub-package for the AAA-SV speaker verification system.

Exports
-------
ASNorm              : Adaptive Score Normalisation (Top-K)
LogisticCalibrator  : Logistic calibrator with optional stability head
select_threshold    : FAR-based threshold selection
save_calibration    : Persist calibration artefact to JSON
load_calibration    : Load calibration artefact from JSON
"""

from aaa_engine.calibration.calibrator import (  # noqa: F401
    ASNorm,
    LogisticCalibrator,
    select_threshold,
    save_calibration,
    load_calibration,
)
