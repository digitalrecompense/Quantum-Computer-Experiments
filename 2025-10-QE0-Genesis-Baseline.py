# ============================================================
#  2025-10-QE0-Genesis-Baseline
#  RAiTHE INDUSTRIES INC.© 2025 — Project Cynric / Genesis Epoch
#  Originator: Robert R. S. Perreault
#
#  Purpose:
#     Establish the absolute zero-separation baseline.
#     This run performs a null 2Q test (no entanglement)
#     to confirm instrument coherence, sampling stability,
#     and to verify the Archivist auto-archive chain.
#
#  Compatible with:
#     qiskit-ibm-runtime >= 0.42.x (sessionless, open plan)
# ============================================================

import sys
from pathlib import Path
sys.path.append(str(Path("C:/Projects/cynric-genesis/core")))

import os
import numpy as np
from datetime import datetime, timezone
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2, SamplerOptions

from cynric_utils import write_json, timestamp_utc, ensure_dir, auto_archive
from check_backend import select_backend

# ------------------------------------------------------------
#  Experiment configuration
# ------------------------------------------------------------
ART_DIR = "C:/Projects/cynric-genesis/artifacts"
ensure_dir(ART_DIR)

SHOTS = 128
TIMESTAMP = timestamp_utc()

# ------------------------------------------------------------
#  Baseline circuits
# ------------------------------------------------------------
def mk_idle_circuit() -> QuantumCircuit:
    """Two-qubit idle circuit (no entanglement)."""
    qc = QuantumCircuit(2, 2)
    qc.barrier()
    qc.measure([0, 1], [0, 1])
    return qc

def mk_hadamard_probe() -> QuantumCircuit:
    """Two-qubit superposition probe for base interference response."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.h(1)
    qc.measure([0, 1], [0, 1])
    return qc

# ------------------------------------------------------------
#  Run Sampler sessionless
# ------------------------------------------------------------
def run_counts(backend, circuits, shots: int):
    opts = SamplerOptions()
    opts.default_shots = shots
    opts.environment = {"job_tags": ["cynric-qe0"]}
    sampler = SamplerV2(mode=backend, options=opts)
    tc = transpile(circuits, backend=backend, optimization_level=1)
    result = sampler.run(tc).result()

    counts_out = []
    for pub in result:
        c = {}
        data = getattr(pub, "data", None)
        if data is not None:
            meas = getattr(data, "meas", None)
            if isinstance(meas, dict):
                c = meas
            else:
                qd = getattr(data, "quasi_dists", None)
                if qd:
                    qdict = qd[0] if isinstance(qd, (list, tuple)) else qd
                    total = sum(qdict.values())
                    if total > 0:
                        c = {format(k, "02b"): int(round(v / total * shots)) for k, v in qdict.items()}
        counts_out.append(c)
    return counts_out

# ------------------------------------------------------------
#  Metrics
# ------------------------------------------------------------
def prob(counts: dict, bit: str) -> float:
    s = sum(counts.values()) if counts else 0
    return counts.get(bit, 0) / s if s else 0.0

def mean_vis(counts_idle: dict, counts_h: dict) -> float:
    """Rough base visibility metric."""
    p00_i = prob(counts_idle, "00")
    p00_h = prob(counts_h, "00")
    return abs(p00_i - p00_h)

# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------
if __name__ == "__main__":
    backend = select_backend()
    if backend is None:
        raise RuntimeError("QE0 aborted: no backend available.")

    print(f"⚡ QE0 — Genesis Baseline on {backend.name}")
    circuits = [mk_idle_circuit(), mk_hadamard_probe()]
    counts_idle, counts_h = run_counts(backend, circuits, SHOTS)

    vis = mean_vis(counts_idle, counts_h)
    print(f"   Baseline visibility ≈ {vis:.4f}")

    payload = {
        "timestamp_utc": TIMESTAMP,
        "backend": backend.name,
        "shots": SHOTS,
        "counts_idle": counts_idle,
        "counts_hadamard": counts_h,
        "visibility": vis,
    }
    write_json(os.path.join(ART_DIR, "qe0_genesis_baseline.json"), payload)

    action = {
        "experiment": "2025-10-QE0-Genesis-Baseline",
        "timestamp": TIMESTAMP,
        "backend": backend.name,
        "V": float(vis),
        "R": 0.0,
        "tau": 0.0,
        "Lambda": 0.0,
    }
    write_json(os.path.join(ART_DIR, "genesis_action.json"), action)

    print("✅ QE0 complete.")
    print(f"   Artifacts saved → {ART_DIR}/qe0_genesis_baseline.json, genesis_action.json")

    # ------------------------------------------------------------
    #  Auto-Archival Trigger
    # ------------------------------------------------------------
    auto_archive("2025-10-QE0-Genesis-Baseline", "Zero-Separation Genesis Baseline Calibration")

# ZS-Link: 2025-10-QE0-Genesis-Baseline | Establish Zero-Point Reference
