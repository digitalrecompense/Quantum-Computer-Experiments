# ============================================================
#  2025-10-QE7-Calibration-Drift-Monitor
#  © 2025  Perreault Enterprises — Project Cynric / Genesis Epoch
#  Goal:
#     Track short-timescale calibration drift by repeating quick
#     parity-sensitive measurements across a small set of time bins.
#     Uses hardware delay() to create idle intervals between prep
#     and measure. Reports odd-parity, <ZZ>, and a trend slope.
#
#  Runtime design: ~3–5 min (SamplerV2, sessionless).
#  Artifacts:
#     - artifacts/qe7_drift_monitor.json
#     - artifacts/genesis_action.json (Archivist schema)
#  Auto-archive: yes (core.auto_archive)
# ============================================================

import sys, os, math, numpy as np
from pathlib import Path
sys.path.append(str(Path("C:/Projects/cynric-genesis/core")))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ECRGate, RZXGate, RXGate, RZGate
from qiskit_ibm_runtime import SamplerV2, SamplerOptions

from check_backend import select_backend
from cynric_utils import (
    ensure_dir, write_json, timestamp_utc,
    echo_summary, auto_archive
)

ART_DIR = "C:/Projects/cynric-genesis/artifacts"
ensure_dir(ART_DIR)

# -------------------- Parameters (runtime-safe) --------------------
SHOTS        = 128
BINS         = 5                         # small to keep total runtime low
# Delays per bin; unit is 'dt' by default. We pick a mild geometric spread.
# If backend.dt~0.22ns (typical), these are small sub-microsecond to a few μs.
BIN_DELAYS_DT = [0, 256, 1024, 4096, 8192]
TIMESTAMP    = timestamp_utc()

# -------------------- Backend helpers ------------------------------
def coupling_map(backend):
    try:
        return getattr(backend.configuration(), "coupling_map", None) or []
    except Exception:
        return []

def pick_pair(backend):
    cmap = coupling_map(backend)
    if cmap:
        a, b = cmap[0]
        return int(a), int(b)
    return 0, 1

def supports_gate(backend, circ):
    try:
        transpile(circ, backend=backend, optimization_level=1)
        return True
    except Exception:
        return False

def pick_entangler(backend):
    """Prefer ECR, else RZX(pi/4)."""
    qc_ecr = QuantumCircuit(2, 2)
    qc_ecr.append(ECRGate(), [0,1]); qc_ecr.measure([0,1],[0,1])
    if supports_gate(backend, qc_ecr):
        return "ecr"

    qc_rzx = QuantumCircuit(2, 2)
    qc_rzx.append(RZXGate(np.pi/4), [0,1]); qc_rzx.measure([0,1],[0,1])
    if supports_gate(backend, qc_rzx):
        return "rzx"

    return "rzx"  # fall back

# -------------------- Metrics --------------------------------------
def counts_total(counts: dict) -> int:
    return int(sum(counts.values())) if counts else 0

def prob(counts: dict, bitstring: str) -> float:
    shots = counts_total(counts)
    return (counts.get(bitstring, 0) / shots) if shots else 0.0

def odd_parity(counts: dict) -> float:
    return prob(counts, "01") + prob(counts, "10")

def expval_ZZ(counts: dict) -> float:
    p00 = prob(counts, "00"); p01 = prob(counts, "01")
    p10 = prob(counts, "10"); p11 = prob(counts, "11")
    return (p00 + p11) - (p01 + p10)

def linfit_slope(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 2: return 0.0
    xm = x.mean(); ym = y.mean()
    denom = float(((x - xm) ** 2).sum()) or 1.0
    return float(((x - xm) * (y - ym)).sum() / denom)

# -------------------- Circuit family --------------------------------
def drift_probe_circuit(entangler: str, delay_dt: int) -> QuantumCircuit:
    """
    Prepare a parity-sensitive state, idle for 'delay_dt' in dt units,
    then measure. Logical qubits are [0,1]; mapping to physical handled
    by transpiler.
    """
    qc = QuantumCircuit(2, 2)
    # Light pre-rotations to spread population a bit
    qc.append(RXGate(np.pi/2), [0])
    qc.append(RXGate(np.pi/2), [1])

    # Entangle
    if entangler == "ecr":
        qc.append(ECRGate(), [0,1])
    else:
        qc.append(RZXGate(np.pi/4), [0,1])

    # Idle (both qubits) to expose drift / dephasing
    if delay_dt > 0:
        qc.delay(delay_dt, 0, unit="dt")
        qc.delay(delay_dt, 1, unit="dt")

    # Balance back toward measurement
    qc.append(RXGate(-np.pi/2), [0])
    qc.append(RXGate(-np.pi/2), [1])
    qc.measure([0,1],[0,1])
    return qc

# -------------------- Sampler run -----------------------------------
def run_sampler(backend, circuits, shots):
    tc = transpile(circuits, backend=backend, optimization_level=2)
    opts = SamplerOptions()
    opts.default_shots = shots
    opts.environment  = {"job_tags": ["cynric-qe7"]}
    sampler = SamplerV2(mode=backend, options=opts)
    result = sampler.run(tc).result()

    outs = []
    for pub in result:
        counts = {}
        data = getattr(pub, "data", None)
        if data is not None:
            meas = getattr(data, "meas", None)
            if isinstance(meas, dict):
                counts = {k: int(v) for k, v in meas.items()}
            else:
                qd = getattr(data, "quasi_dists", None)
                if qd:
                    qdict = qd[0] if isinstance(qd, (list, tuple)) else qd
                    total = sum(qdict.values()) or 1.0
                    counts = {format(k, "02b"): int(round(v/total*shots)) for k, v in qdict.items()}
                    for key in ("00","01","10","11"):
                        counts.setdefault(key, 0)
        outs.append(counts)
    return outs

# -------------------- Main ------------------------------------------
if __name__ == "__main__":
    backend = select_backend()
    if backend is None:
        raise RuntimeError("QE7 aborted: no suitable backend available.")

    q0, q1 = pick_pair(backend)
    ent = pick_entangler(backend)

    print(f"⚡ QE7 — Calibration Drift Monitor on {backend.name}")
    print(f"   Pair=({q0},{q1}), entangler={ent}, bins={BINS}, shots/bin={SHOTS}")

    # Build circuits for bins
    delays = BIN_DELAYS_DT[:BINS]
    labels = []
    circs  = []
    for i, d in enumerate(delays):
        circs.append(drift_probe_circuit(ent, d))
        labels.append(int(d))

    # Execute
    counts_list = run_sampler(backend, circs, SHOTS)

    # Per-bin metrics
    per_bin = []
    zz_vals = []
    odd_vals = []
    for d, c in zip(labels, counts_list):
        zz = float(expval_ZZ(c))
        op = float(odd_parity(c))
        per_bin.append({
            "delay_dt": d,
            "shots": int(counts_total(c)),
            "odd_parity": op,
            "ZZ": zz,
            "counts": c
        })
        zz_vals.append(zz)
        odd_vals.append(op)

    # Trend (slope vs delay_dt)
    x = np.asarray(labels, dtype=float)
    slope_ZZ  = linfit_slope(x, np.asarray(zz_vals, dtype=float))
    slope_odd = linfit_slope(x, np.asarray(odd_vals, dtype=float))

    # Rollups (presentation-friendly)
    R_mean     = float(np.mean(np.abs(zz_vals))) if zz_vals else 0.0
    V_mean     = float(np.mean(np.abs(1.0 - 2.0*np.asarray(odd_vals)))) if odd_vals else 0.0
    tau_mean   = 1.0 / (1.0 + float(np.var(np.diff(zz_vals)))) if len(zz_vals) > 1 else 0.0
    Lambda_mean= float(np.mean(np.abs(np.diff(zz_vals, n=2)))) if len(zz_vals) > 2 else 0.0

    payload = {
        "timestamp_utc": TIMESTAMP,
        "backend": backend.name,
        "pair": [int(q0), int(q1)],
        "entangler": ent,
        "bins": BINS,
        "delays_dt": labels,
        "shots_per_bin": int(SHOTS),
        "per_bin": per_bin,
        "trends": {
            "slope_ZZ_per_dt": float(slope_ZZ),
            "slope_odd_per_dt": float(slope_odd)
        },
        "rollups": {
            "V_mean": V_mean,
            "R_mean": R_mean,
            "tau_mean": float(tau_mean),
            "Lambda_mean": float(Lambda_mean)
        }
    }

    art_path = os.path.join(ART_DIR, "qe7_drift_monitor.json")
    write_json(art_path, payload)

    # Archivist action for visualizer rollup
    action = {
        "experiment": "2025-10-QE7-Calibration-Drift-Monitor",
        "timestamp": TIMESTAMP,
        "backend": backend.name,
        "artifacts": ["qe7_drift_monitor.json"],
        "V_mean": V_mean,
        "R_mean": R_mean,
        "tau_mean": float(tau_mean),
        "Lambda_mean": float(Lambda_mean),
        "notes": "Parity drift across delay bins; slopes indicate short-timescale calibration drift."
    }
    write_json(os.path.join(ART_DIR, "genesis_action.json"), action)

    echo_summary("QE7 trends", payload["trends"])
    echo_summary("QE7 rollups", payload["rollups"])
    print("✅ QE7 complete.")
    print(f"   slopes: ZZ≈{slope_ZZ:.4g}/dt, odd≈{slope_odd:.4g}/dt")
    print(f"   Artifacts → {art_path}, genesis_action.json")

    # Auto-archive + visualize
    auto_archive(
        name="2025-10-QE7-Calibration-Drift-Monitor",
        purpose="Short-timescale parity drift monitor across hardware delay bins"
    )