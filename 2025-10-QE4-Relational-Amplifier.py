# ============================================================
#  2025-10-QE4-RelationalAmplifier
#  © 2025  Perreault Enterprises — Project Cynric / Genesis Epoch
#  Originator: Robert R. S. Perreault
#
#  Purpose:
#     Amplify and quantify relational coupling by sweeping entangling
#     angle (θ) and depth (1..4) using native 2Q operations.
#
#  Canonical Mapping:
#     New QE4 (relabels old QE5).
#
#  Artifacts:
#     - artifacts/qe4_relational_amplifier.json
#     - artifacts/genesis_action.json   (Archivist schema)
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
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RZXGate, ECRGate, RXGate, RZGate
from qiskit_ibm_runtime import SamplerV2, SamplerOptions

from check_backend import select_backend
from cynric_utils import ensure_dir, write_json, timestamp_utc, auto_archive

# ------------------------------------------------------------
#  Parameters
# ------------------------------------------------------------
ART_DIR = "C:/Projects/cynric-genesis/artifacts"
ensure_dir(ART_DIR)

SHOTS   = 200
DEPTHS  = [1, 2, 3, 4]                     # “up the quantum output value to 3 and 4” included
THETAS  = np.linspace(0.0, 2*np.pi, 13)    # 0..2π (13 points)
PREFS   = ["ecr", "rzx"]                   # native-ish preference order
TIMESTAMP = timestamp_utc()

# ------------------------------------------------------------
#  Backend / pairing
# ------------------------------------------------------------
def pick_pair(backend):
    """Pick first connected pair; fallback to (0,1)."""
    try:
        cmap = getattr(backend.configuration(), "coupling_map", None)
        if cmap and len(cmap) > 0:
            a, b = cmap[0]
            return int(a), int(b)
    except Exception:
        pass
    return 0, 1

def try_transpile_probe(backend, gate_name: str) -> bool:
    """Quick probe to check that a candidate entangler can transpile."""
    try:
        test = amplifier_circuit(gate_name, depth=1, theta=float(math.pi/4), q0=0, q1=1)
        _ = transpile(test, backend=backend, optimization_level=1)
        return True
    except Exception:
        return False

def pick_native_gate(backend) -> str:
    """Prefer ECR; fall back to RZX; default to RZX if both fail probes."""
    for cand in PREFS:
        if try_transpile_probe(backend, cand):
            return cand
    return "rzx"

# ------------------------------------------------------------
#  Counts / Observables
# ------------------------------------------------------------
def counts_total(c): return int(sum(c.values())) if c else 0

def prob(c, key):
    t = counts_total(c)
    return (c.get(key, 0) / t) if t else 0.0

def expval_ZZ(c):
    """⟨Z⊗Z⟩ = P(00)+P(11) - P(01)-P(10)."""
    p00 = prob(c, "00"); p01 = prob(c, "01")
    p10 = prob(c, "10"); p11 = prob(c, "11")
    return (p00 + p11) - (p01 + p10)

def expval_Z0(c):
    """⟨Z⊗I⟩ = P(0*) - P(1*)."""
    p0 = prob(c, "00") + prob(c, "01")
    p1 = prob(c, "10") + prob(c, "11")
    return p0 - p1

def expval_Z1(c):
    """⟨I⊗Z⟩ = P(*0) - P(*1)."""
    p0 = prob(c, "00") + prob(c, "10")
    p1 = prob(c, "01") + prob(c, "11")
    return p0 - p1

def parity_odd(c):
    """P(01)+P(10)."""
    return prob(c, "01") + prob(c, "10")

# ------------------------------------------------------------
#  Circuit family (amplifier)
# ------------------------------------------------------------
def amplifier_block(qc: QuantumCircuit, gate_name: str, theta: float, q0: int, q1: int):
    """
    One layer:
      - pre-rotations to spread population
      - entangler (ECR or RZX(θ))
      - post-rotations to measurement-stable frame
    """
    qc.append(RXGate(np.pi/2), [q0])
    qc.append(RXGate(np.pi/2), [q1])
    qc.append(RZGate(theta/2), [q0])
    qc.append(RZGate(-theta/3), [q1])

    if gate_name == "ecr":
        qc.append(ECRGate(), [q0, q1])
        # emulate a tunable effective angle via relative phases
        qc.append(RZGate(theta/4), [q0])
        qc.append(RZGate(-theta/4), [q1])
    elif gate_name == "rzx":
        qc.append(RZXGate(theta), [q0, q1])
    else:
        raise ValueError(f"Unsupported entangler: {gate_name}")

    qc.append(RXGate(-np.pi/2), [q0])
    qc.append(RXGate(-np.pi/2), [q1])

def amplifier_circuit(gate_name: str, depth: int, theta: float, q0: int, q1: int) -> QuantumCircuit:
    """Depth-parameterized amplifier on a physical pair projected to [0,1]."""
    qc = QuantumCircuit(2, 2)
    if (q0, q1) != (0, 1):
        qc.swap(0, 1)
    for _ in range(int(depth)):
        amplifier_block(qc, gate_name, theta, 0, 1)
    if (q0, q1) != (0, 1):
        qc.swap(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

# ------------------------------------------------------------
#  Execute (sessionless SamplerV2) → counts-like
# ------------------------------------------------------------
def run_counts(backend, circuits, shots: int):
    """
    Transpile, execute via sessionless SamplerV2(mode=backend),
    convert quasi-dists to integer-like counts for downstream metrics.
    """
    tc = transpile(circuits, backend=backend, optimization_level=3)
    opts = SamplerOptions()
    opts.default_shots = shots
    opts.environment = {"job_tags": ["cynric-qe4"]}
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
                        tmp = {}
                        for k, v in qdict.items():
                            key = format(k, "02b") if isinstance(k, int) else str(k).zfill(2)
                            tmp[key] = tmp.get(key, 0) + int(round(v / total * shots))
                        for key in ("00", "01", "10", "11"):
                            tmp.setdefault(key, 0)
                        counts = tmp
        outs.append(counts)
    return outs

# ------------------------------------------------------------
#  Aggregation across θ sweep
# ------------------------------------------------------------
def summarize_theta_sweep(theta_list, counts_list):
    """
    Aggregate per-theta metrics + rollups:
      V: mean(|Z0|, |Z1|)   — stability proxy
      R: mean(|ZZ|)         — relational coupling proxy
      tau: 1/(1+Var(ΔZZ))   — smoothness proxy
      Lambda: mean|Δ²ZZ|    — curvature magnitude (presentation metric)
    """
    per = []
    zz_vals = []
    for th, c in zip(theta_list, counts_list):
        z0 = float(expval_Z0(c))
        z1 = float(expval_Z1(c))
        zz = float(expval_ZZ(c))
        per.append({
            "theta": float(th),
            "shots": counts_total(c),
            "z0": z0,
            "z1": z1,
            "zz": zz,
            "odd_parity": float(parity_odd(c)),
            "counts": c
        })
        zz_vals.append(zz)

    zz_vals = np.asarray(zz_vals, dtype=float)
    if len(zz_vals) >= 3:
        d1 = np.diff(zz_vals)
        d2 = np.diff(zz_vals, n=2)
        tau = 1.0 / (1.0 + float(np.var(d1)))
        lam = float(np.mean(np.abs(d2)))
    elif len(zz_vals) == 2:
        d1 = np.diff(zz_vals)
        tau = 1.0 / (1.0 + float(np.var(d1)))
        lam = 0.0
    else:
        tau = 0.0
        lam = 0.0

    V = float(np.mean(np.abs([x["z0"] for x in per] + [x["z1"] for x in per]))) if per else 0.0
    R = float(np.mean(np.abs([x["zz"] for x in per]))) if per else 0.0

    return {"per_theta": per, "V": V, "R": R, "tau": float(tau), "Lambda": float(lam)}

# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------
if __name__ == "__main__":
    backend = select_backend()
    if backend is None:
        raise RuntimeError("QE4 aborted: no suitable backend available.")

    q0, q1 = pick_pair(backend)
    ent_name = pick_native_gate(backend)

    print(f"⚡ QE4 — Relational Amplifier on {backend.name}")
    print(f"   Pair=({q0},{q1}), Entangler={ent_name}, Depths={DEPTHS}, Thetas={len(THETAS)} points")

    # Build circuits (depth, theta grid)
    circuits = []
    labels = []
    for d in DEPTHS:
        for th in THETAS:
            labels.append((int(d), float(th)))
            circuits.append(amplifier_circuit(ent_name, depth=d, theta=float(th), q0=q0, q1=q1))

    # Execute
    counts_all = run_counts(backend, circuits, SHOTS)

    # Organize and summarize per depth
    results_by_depth = {}
    off = 0
    for d in DEPTHS:
        span = len(THETAS)
        clist = counts_all[off:off+span]
        results_by_depth[d] = summarize_theta_sweep([float(t) for t in THETAS], clist)
        off += span

    # Pick best depth by highest R (+ tie-break on V)
    best_depth, best_score = None, -1e9
    for d, agg in results_by_depth.items():
        score = agg["R"] + 0.15 * agg["V"]
        if score > best_score:
            best_score, best_depth = score, d

    # Persist detailed artifact
    write_json(
        os.path.join(ART_DIR, "qe4_relational_amplifier.json"),
        {
            "timestamp_utc": TIMESTAMP,
            "backend": backend.name,
            "pair": [int(q0), int(q1)],
            "entangler": ent_name,
            "shots": int(SHOTS),
            "depths": DEPTHS,
            "thetas": [float(t) for t in THETAS],
            "results": results_by_depth,
            "best_depth": int(best_depth) if best_depth is not None else None,
        },
    )

    # Archivist action (ASCII keys only) — publish best-depth aggregates
    if best_depth is not None:
        agg = results_by_depth[best_depth]
        action = {
            "experiment": "2025-10-QE4-RelationalAmplifier",
            "timestamp": TIMESTAMP,
            "backend": backend.name,
            "artifacts": ["qe4_relational_amplifier.json"],
            "V": float(agg["V"]),
            "R": float(agg["R"]),
            "tau": float(agg["tau"]),
            "Lambda": float(agg["Lambda"]),
        }
    else:
        action = {
            "experiment": "2025-10-QE4-RelationalAmplifier",
            "timestamp": TIMESTAMP,
            "backend": backend.name,
            "artifacts": ["qe4_relational_amplifier.json"],
            "V": 0.0, "R": 0.0, "tau": 0.0, "Lambda": 0.0,
        }

    write_json(os.path.join(ART_DIR, "genesis_action.json"), action)

    print("✅ QE4 complete.")
    if best_depth is not None:
        print(f"   Best depth: {best_depth}  |  R≈{action['R']:.3f}, V≈{action['V']:.3f}, "
              f"tau≈{action['tau']:.3f}, Lambda≈{action['Lambda']:.3f}")
    else:
        print("   No best depth determined (insufficient signal); published zeros.")

    print(f"   Artifacts → {ART_DIR}/qe4_relational_amplifier.json, genesis_action.json")

    # Auto-archive (Phase pipeline)
    auto_archive("2025-10-QE4-RelationalAmplifier", "Relational Amplifier (depth/θ sweep)")
# ZS-Link: 2025-10-QE4-RelationalAmplifier | Relational Depth Sweep