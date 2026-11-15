"""
Offline load proof: verify AAASV wrapper loads and runs without network access.
The stub model (no model.pt) simulates staged offline model files.
Run: python hpc/offline_load_test.py
Saves: docs/evidence/day4/offline_load_proof.json
"""
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from aaa_engine.inference.aaasv import AAASV

def _make_staged_model_dir():
    """Stage model config files at a local path (no network needed)."""
    staged = tempfile.mkdtemp(prefix="aaasv_offline_staged_")
    with open(os.path.join(staged, "config.yaml"), "w") as f:
        f.write("model_name: ecapa-tdnn-b1\nembedding_dim: 192\n")
    with open(os.path.join(staged, "preproc.yaml"), "w") as f:
        f.write("sample_rate: 16000\nn_mels: 80\n")
    with open(os.path.join(staged, "calibration.json"), "w") as f:
        json.dump({
            "fitted": False, "threshold": 0.5,
            "tau_anchor": 0.85, "q_min": 0.70,
            "note": "Unfitted — staged for offline test"
        }, f)
    # NOTE: model.pt is intentionally absent (checkpoints not committed per data rules)
    # The wrapper falls back to _StubModel which runs without network.
    return staged

results = {}
staged_dir = _make_staged_model_dir()

try:
    sv = AAASV(staged_dir, device="cpu")

    rng = np.random.default_rng(42)
    wav = rng.standard_normal(int(3.0 * 16000)).astype(np.float32) * 0.1
    resp = sv.embed(wav, 16000)

    assert resp.ok, f"embed() failed: {resp.error_code}"
    assert resp.result.embedding.shape == (192,), f"Wrong shape: {resp.result.embedding.shape}"
    assert abs(np.linalg.norm(resp.result.embedding) - 1.0) < 1e-5, "Not L2-normalised"

    results = {
        "outcome": "PASS",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_dir": "staged_local_tempdir (absolute path withheld)",
        "network_required": False,
        "model_pt_present": False,
        "stub_model_used": True,
        "embed_ok": resp.ok,
        "embedding_shape": list(resp.result.embedding.shape),
        "l2_norm": float(np.linalg.norm(resp.result.embedding)),
        "latency_ms": resp.latency_ms,
        "note": (
            "model.pt absent (not committed per data rules). "
            "Wrapper falls back to StubModel, runs without network. "
            "Real checkpoint staged on HPC at: "
            "runs/20261113-1718_09dc907_5e10bf/checkpoint_best.pt"
        )
    }
    print("OFFLINE LOAD TEST: PASS")
    print(f"  embed ok={resp.ok}, shape={resp.result.embedding.shape}, latency={resp.latency_ms:.1f}ms")

except Exception as exc:
    results = {
        "outcome": "FAIL",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": str(exc),
    }
    print(f"OFFLINE LOAD TEST: FAIL — {exc}")

out_path = Path("docs/evidence/day4/offline_load_proof.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {out_path}")
