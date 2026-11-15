"""
Measure CPU inference latency of AAASV wrapper.
Run: python hpc/measure_latency.py
Saves: docs/evidence/day4/latency_result.json
"""
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from aaa_engine.inference.aaasv import AAASV
import tempfile, json as _json

# Create a minimal stub model_dir (no model.pt → uses stub)
def _make_stub_model_dir():
    tmpdir = tempfile.mkdtemp(prefix="aaasv_stub_")
    with open(os.path.join(tmpdir, "config.yaml"), "w") as f:
        f.write("model_name: ecapa-tdnn-stub\n")
    with open(os.path.join(tmpdir, "preproc.yaml"), "w") as f:
        f.write("sample_rate: 16000\n")
    with open(os.path.join(tmpdir, "calibration.json"), "w") as f:
        _json.dump({"fitted": False, "threshold": 0.5, "tau_anchor": 0.85, "q_min": 0.70}, f)
    return tmpdir

N_WARMUP = 10
N_CALLS = 100
AUDIO_DURATION_S = 3.0
SAMPLE_RATE = 16000

model_dir = _make_stub_model_dir()
sv = AAASV(model_dir, device="cpu")

rng = np.random.default_rng(42)
wav = rng.standard_normal(int(AUDIO_DURATION_S * SAMPLE_RATE)).astype(np.float32) * 0.1

# Warmup
for _ in range(N_WARMUP):
    sv.embed(wav, SAMPLE_RATE)

# Measure
times_ms = []
for _ in range(N_CALLS):
    t0 = time.perf_counter()
    resp = sv.embed(wav, SAMPLE_RATE)
    times_ms.append((time.perf_counter() - t0) * 1000.0)

times_ms.sort()
p50 = float(np.percentile(times_ms, 50))
p95 = float(np.percentile(times_ms, 95))
p99 = float(np.percentile(times_ms, 99))

result = {
    "run_id": str(uuid.uuid4()),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "n_warmup": N_WARMUP,
    "n_calls": N_CALLS,
    "audio_duration_s": AUDIO_DURATION_S,
    "device": "cpu",
    "model": "stub (no model.pt — real model not committed)",
    "latency_p50_ms": p50,
    "latency_p95_ms": p95,
    "latency_p99_ms": p99,
    "note": "Stub model used. Real checkpoint latency will differ (larger model). Run on HPC with model.pt for real numbers."
}

out_path = Path("docs/evidence/day4/latency_result.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"Latency (stub, CPU, {AUDIO_DURATION_S}s audio):")
print(f"  p50 = {p50:.1f} ms")
print(f"  p95 = {p95:.1f} ms")
print(f"  p99 = {p99:.1f} ms")
print(f"Saved to {out_path}")
