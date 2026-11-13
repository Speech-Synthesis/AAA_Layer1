# =============================================================
# aaa_engine/evaluation/__init__.py
# AAA Engine — Evaluation sub-package
# =============================================================
# Public API re-exports for the evaluation toolkit.
# =============================================================

from aaa_engine.evaluation.eval_toolkit import (
    BucketedTrialGenerator,
    EvalResult,
    TrialList,
    evaluate_model,
    FREEZE_ENABLED,
)

__all__ = [
    "BucketedTrialGenerator",
    "EvalResult",
    "TrialList",
    "evaluate_model",
    "FREEZE_ENABLED",
]
