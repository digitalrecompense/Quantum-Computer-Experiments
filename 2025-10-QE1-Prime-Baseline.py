# ============================================================
#  2025-10-QE1-Prime-Baseline
#  RAiTHE INDUSTRIES INC.© 2025 — Project Cynric / Genesis Epoch
#  Originator: Robert R. S. Perreault
#
#  Purpose:
#     Perform the primary τ-normalized baseline calibration for
#     all subsequent QE-series experiments.
#     Establishes reference fringe, contrast, and parity values
#     using non-entangling two-qubit primitives.
#
#  Compatible with:
#     qiskit-ibm-runtime >= 0.42.x (sessionless)
# ============================================================

import sys
from pathlib import Path
sys.path.append(str(Path("C:/Projects/cynric-genesis/core")))

import os
import numpy as np
from datetime import datetime, timezone
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import SamplerV2, SamplerOptions

from check_backend import select_backend
from cynric_utils import write_json, ensure_dir, timestamp_utc, write_wav_from_series, auto_archive

# ------------------------------------------------------------
#  Experiment setup
# ------------------------------------------------------------
ART_DIR = "C:/Projects/cynric-genesis/artifacts"
ensure_dir(ART_DIR)
SHOTS = 256
PHI_POINTS = 48
THETA_POINTS = 36
TIMESTAMP = timestamp_utc()

# ------------------------------------------------------------
#  Circuit definitions
# ------------------------------------------------------------
def make_fringe_circuit(phi: float) -> QuantumCircuit:
    """Single-qubit phase fringe test on q0."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.rz(phi, 0)
    qc.h(0)
    qc.measure(0, 0)
    return qc

def make_parity_circuit(theta: float) -> QuantumCircuit:
    """Two-qubit parity test for baseline correlation."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    return qc

# ------------------------------------------------------------
#  Execute circuits sessionlessly
# ------------------------------------------------------------
def run_counts(backend, circuits, shots: int):
    opts = SamplerOptions()
    opts.default_shots = shots
    opts.environment = {"job_tags": ["cynric-qe1"]}
    sampler = SamplerV2(mode=backend, options=opts)
    tc = transpile(circuits, backend=backend, optimization_level=1)
    result = sampler.run(tc).result()

    outputs = []
    for pub in result:
        counts = {}
        data = getattr(pub, "data", None)
        if data is not None:
            meas = getattr(data, "meas", None)
            if isinstance(meas, dict):
                counts = meas
            else:
                qd = getattr(data, "quasi_dists", None)
                if qd:
                    qdict = qd[0] if isinstance(qd, (list, tuple)) else qd
                    total = sum(qdict.values())
                    if total > 0:
                        counts = {format(k, "b").zfill(2): int(round(v / total * shots)) for k, v in qdict.items()}
        outputs.append(counts)
    return outputs

# ------------------------------------------------------------
#  Metrics
# ------------------------------------------------------------
def prob(counts: dict, key: str) -> float:
    total = sum(counts.values()) if counts else 0
    return counts.get(key, 0) / total if total else 0.0

def fringe_visibility(p_series):
    """Compute classical fringe visibility (max-min)/(max+min)."""
    if not p_series:
        return 0.0
    mx, mn = max(p_series), min(p_series)
    denom = (mx + mn) if (mx + mn) != 0 else 1e-9
    return (mx - mn) / denom

# ------------------------------------------------------------
#  Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    backend = select_backend()
    if backend is None:
        raise RuntimeError("QE1 aborted: no backend available.")

    print(f"⚡ QE1 — Prime Baseline Calibration on {backend.name}")

    # --- Phase A: Single-qubit fringe ---
    phis = np.linspace(0, 2 * np.pi, PHI_POINTS)
    circuits_phi = [make_fringe_circuit(phi) for phi in phis]
    counts_phi = run_counts(backend, circuits_phi, SHOTS)
    p1_series = [prob(c, "1") for c in counts_phi]
    V = fringe_visibility(p1_series)

    # --- Phase B: Two-qubit parity ---
    thetas = np.linspace(0, 2 * np.pi, THETA_POINTS)
    circuits_theta = [make_parity_circuit(theta) for theta in thetas]
    counts_theta = run_counts(backend, circuits_theta, SHOTS)
    p00_series = [prob(c, "00") for c in counts_theta]
    R = fringe_visibility(p00_series)

    # Derived τ and Λ (placeholders)
    tau = float(1.0 / (1.0 + np.var(p1_series))) if len(p1_series) > 1 else 0.0
    Lambda = 0.0

    # Write artifacts
    write_json(os.path.join(ART_DIR, "qe1prime_contrast.json"), {"phis": phis.tolist(), "p1_series": p1_series})
    write_json(os.path.join(ART_DIR, "qe1prime_parity.json"), {"thetas": thetas.tolist(), "p00_series": p00_series})
    write_wav_from_series(os.path.join(ART_DIR, "qe1prime_fringe.wav"), p1_series)

    # Archivist schema
    summary = {
        "timestamp": TIMESTAMP,
        "experiment": "2025-10-QE1-Prime-Baseline",
        "backend": backend.name,
        "V": float(V),
        "R": float(R),
        "tau": float(tau),
        "Lambda": float(Lambda),
    }
    write_json(os.path.join(ART_DIR, "genesis_action.json"), summary)

    print("✅ QE1 complete.")
    print(f"   V≈{V:.4f}, R≈{R:.4f}, τ≈{tau:.4f}, Λ≈{Lambda:.4f}")
    print(f"   Artifacts saved → {ART_DIR}/qe1prime_contrast.json, qe1prime_parity.json, genesis_action.json")

    # ------------------------------------------------------------
    #  Auto-Archival Trigger
    # ------------------------------------------------------------
    auto_archive("2025-10-QE1-Prime-Baseline", "Prime Baseline Calibration (τ-normalized)")


# ZS-Link: 2025-10-QE1-Prime-Baseline | τ-Normalized Baseline
