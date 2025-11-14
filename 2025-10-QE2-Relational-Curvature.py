# ============================================================
#  2025-10-QE2-Relational-Curvature
#  © 2025  Perreault Enterprises — Project Cynric / Genesis Epoch
#  Originator: Robert R. S. Perreault
#
#  Purpose:
#     Sweep a relational “curvature” parameter Λ and a phase bias φ0
#     across a native 2Q entangler and quantify relational structure.
#
#  Replaces:
#     Old QE3 — same intent, refactored under new canonical IDs.
#
#  Compatible with:
#     qiskit-ibm-runtime >= 0.42.x (sessionless, open plan)
# ============================================================

import sys
from pathlib import Path
sys.path.append(str(Path("C:/Projects/cynric-genesis/core")))

import os
import math
import numpy as np
from datetime import datetime, timezone
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ECRGate, RZXGate, RZGate, RXGate
from qiskit_ibm_runtime import SamplerV2, SamplerOptions

from check_backend import select_backend
from cynric_utils import ensure_dir, write_json, timestamp_utc, auto_archive

# ------------------------------------------------------------
#  Parameters
# ------------------------------------------------------------
ART_DIR = "C:/Projects/cynric-genesis/artifacts"
ensure_dir(ART_DIR)

SHOTS = 200
PHI0 = np.linspace(-np.pi, np.pi, 9)          # phase bias grid
LAMBDAS = np.linspace(-0.5, 0.5, 9)           # curvature-like knob
TIMESTAMP = timestamp_utc()

# ------------------------------------------------------------
#  Helpers: backend & pairing
# ------------------------------------------------------------
def pick_pair(backend):
    """Pick the first connected pair from coupling map; fallback to (0,1)."""
    try:
        cmap = getattr(backend.configuration(), "coupling_map", None)
        if cmap and len(cmap) > 0:
            a, b = cmap[0]
            return int(a), int(b)
    except Exception:
        pass
    return 0, 1

# ------------------------------------------------------------
#  Circuit: “Relational Curvature” block
# ------------------------------------------------------------
def curvature_block(qc: QuantumCircuit, lam: float, phi0: float, q0: int, q1: int):
    """
    One relational block:
      - phase bias (RZ) on each qubit
      - entangler with Λ-shaped angle
      - balancing rx rotations
    """
    # phase bias
    qc.append(RZGate(phi0), [q0])
    qc.append(RZGate(-phi0), [q1])

    # entangler: RZX with angle mapped from Λ
    theta = (math.pi / 2.0) * lam
    qc.append(RZXGate(theta), [q0, q1])

    # balance to keep measurement informative
    qc.append(RXGate(+np.pi/2), [q0])
    qc.append(RXGate(-np.pi/2), [q1])

def relational_circuit(lam: float, phi0: float, q0: int, q1: int) -> QuantumCircuit:
    """
    Build a minimal two-qubit probe (one block); measure Z⊗Z parity landscape
    as a function of φ0 and Λ.
    """
    qc = QuantumCircuit(2, 2)
    # prepare superpositions so entangler is effective
    qc.h(0)
    qc.h(1)

    # if hardware pair != (0,1), wrap by swaps
    if (q0, q1) != (0, 1):
        qc.swap(0, 1)

    curvature_block(qc, lam, phi0, 0, 1)

    if (q0, q1) != (0, 1):
        qc.swap(0, 1)

    qc.measure([0, 1], [0, 1])
    return qc

# ------------------------------------------------------------
#  Execute via sessionless SamplerV2 and convert to counts-like
# ------------------------------------------------------------
def run_counts(backend, circuits, shots: int):
    """
    Transpile and execute, then coerce quasi-dists into int-like counts for metrics.
    """
    tc = transpile(circuits, backend=backend, optimization_level=2)
    opts = SamplerOptions()
    opts.default_shots = shots
    opts.environment = {"job_tags": ["cynric-qe2"]}
    sampler = SamplerV2(mode=backend, options=opts)
    result = sampler.run(tc).result()

    outs = []
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
                    total = float(sum(qdict.values()))
                    if total > 0:
                        # keys can be ints (basis states) or bitstrings depending on backend
                        tmp = {}
                        for k, v in qdict.items():
                            if isinstance(k, int):
                                key = format(k, "02b")
                            else:
                                key = str(k).zfill(2)
                            tmp[key] = tmp.get(key, 0) + int(round(v / total * shots))
                        # ensure all four present
                        for key in ("00", "01", "10", "11"):
                            tmp.setdefault(key, 0)
                        counts = tmp
        outs.append(counts)
    return outs

# ------------------------------------------------------------
#  Metrics (local, ASCII-safe)
# ------------------------------------------------------------
def counts_total(c): return int(sum(c.values())) if c else 0
def prob(c, s): 
    t = counts_total(c); 
    return (c.get(s, 0) / t) if t else 0.0

def expval_ZZ(c):
    """⟨Z⊗Z⟩ = P00+P11 − P01−P10"""
    return (prob(c, "00") + prob(c, "11")) - (prob(c, "01") + prob(c, "10"))

def expval_Z0(c):
    """⟨Z⊗I⟩ = P(0*) − P(1*)"""
    p0 = prob(c, "00") + prob(c, "01")
    p1 = prob(c, "10") + prob(c, "11")
    return p0 - p1

def expval_Z1(c):
    """⟨I⊗Z⟩ = P(*0) − P(*1)"""
    p0 = prob(c, "00") + prob(c, "10")
    p1 = prob(c, "01") + prob(c, "11")
    return p0 - p1

# ------------------------------------------------------------
#  Aggregation
# ------------------------------------------------------------
def summarize_surface(phi_list, lam_list, counts_grid):
    """
    counts_grid indexed as [i_phi][j_lam] → counts dict.
    Returns global aggregates used by Archivist and a detailed table.
    """
    rows = []
    zz_vals = []

    for i, phi0 in enumerate(phi_list):
        for j, lam in enumerate(lam_list):
            c = counts_grid[i][j]
            zz = float(expval_ZZ(c))
            z0 = float(expval_Z0(c))
            z1 = float(expval_Z1(c))
            rows.append({
                "phi0": float(phi0),
                "Lambda": float(lam),
                "shots": counts_total(c),
                "p00": prob(c, "00"),
                "p01": prob(c, "01"),
                "p10": prob(c, "10"),
                "p11": prob(c, "11"),
                "zz": zz,
                "z0": z0,
                "z1": z1
            })
            zz_vals.append(zz)

    zz_vals = np.asarray(zz_vals, dtype=float)
    if len(zz_vals) >= 3:
        d1 = np.diff(zz_vals)
        d2 = np.diff(zz_vals, n=2)
        tau = 1.0 / (1.0 + float(np.var(d1)))     # smoothness proxy (bounded-ish 0..1)
        Lambda_mag = float(np.mean(np.abs(d2)))    # curvature magnitude
    elif len(zz_vals) == 2:
        d1 = np.diff(zz_vals)
        tau = 1.0 / (1.0 + float(np.var(d1)))
        Lambda_mag = 0.0
    else:
        tau = 0.0
        Lambda_mag = 0.0

    # Relational magnitude & local visibility proxies
    R = float(np.mean(np.abs(zz_vals))) if len(zz_vals) else 0.0
    V = float(np.mean(np.abs([r["z0"] for r in rows] + [r["z1"] for r in rows]))) if rows else 0.0

    return {
        "grid": rows,
        "V": V,
        "R": R,
        "tau": tau,
        "Lambda": Lambda_mag
    }

# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------
if __name__ == "__main__":
    backend = select_backend()
    if backend is None:
        raise RuntimeError("QE2 aborted: no suitable backend available.")

    q0, q1 = pick_pair(backend)
    print(f"⚡ QE2 — Relational Curvature Sweep on {backend.name} (pair=({q0},{q1}))")

    # Build grid of circuits
    circuits = []
    labels = []
    for phi0 in PHI0:
        row = []
        labrow = []
        for lam in LAMBDAS:
            row.append(relational_circuit(float(lam), float(phi0), q0, q1))
            labrow.append((float(phi0), float(lam)))
        circuits.extend(row)
        labels.append(labrow)

    # Execute
    counts_all = run_counts(backend, circuits, SHOTS)

    # Reshape into [phi][lam]
    counts_grid = []
    idx = 0
    for _ in PHI0:
        row = []
        for _ in LAMBDAS:
            row.append(counts_all[idx])
            idx += 1
        counts_grid.append(row)

    # Summarize
    agg = summarize_surface(PHI0, LAMBDAS, counts_grid)

    # Persist detailed artifact
    payload = {
        "timestamp_utc": TIMESTAMP,
        "backend": backend.name,
        "pair": [int(q0), int(q1)],
        "shots": int(SHOTS),
        "phi0_list": [float(x) for x in PHI0],
        "lambda_list": [float(x) for x in LAMBDAS],
        "summary": {
            "V": float(agg["V"]),
            "R": float(agg["R"]),
            "tau": float(agg["tau"]),
            "Lambda": float(agg["Lambda"])
        },
        "grid": agg["grid"]
    }
    write_json(os.path.join(ART_DIR, "qe2_relational_curvature.json"), payload)

    # Archivist action (ASCII keys only)
    action = {
        "experiment": "2025-10-QE2-Relational-Curvature",
        "timestamp": TIMESTAMP,
        "backend": backend.name,
        "artifacts": ["qe2_relational_curvature.json"],
        "V": float(agg["V"]),
        "R": float(agg["R"]),
        "tau": float(agg["tau"]),
        "Lambda": float(agg["Lambda"])
    }
    write_json(os.path.join(ART_DIR, "genesis_action.json"), action)

    print("✅ QE2 complete.")
    print(f"   V≈{action['V']:.3f}, R≈{action['R']:.3f}, tau≈{action['tau']:.3f}, Λ≈{action['Lambda']:.3f}")
    print(f"   Artifacts → {ART_DIR}/qe2_relational_curvature.json, genesis_action.json")

    # Auto-archive handoff
    auto_archive("2025-10-QE2-Relational-Curvature", "Relational Curvature Sweep (Cross-Qubit Relativity)")

# ZS-Link: 2025-10-QE2-Relational-Curvature | Λ-Phase Relational Surface