# ============================================================
#  2025-10-QE5-Echo-Calibrated-RZX-Scan
#  © 2025  Perreault Enterprises — Project Cynric / Genesis Epoch
#  Originator: Robert R. S. Perreault
#
#  Goal:
#     Echo-protected scan of RZX(θ) amplitude on a native pair,
#     robust to slow phase noise. Measures:
#         - ⟨ZZ⟩ correlation vs θ
#         - single-qubit visibilities (Z-marginals)
#     + optional tiny spin-echo T2* proxy (contrast gain w/ echo)
#
#  Runtime target: ~3–5 min (small θ-grid, modest shots)
#
#  Artifacts:
#     - artifacts/qe5_rzx_echo_scan.json
#     - artifacts/genesis_action.json   (Archivist schema)
#
#  Compatible with:
#     qiskit-ibm-runtime >= 0.42.x   (SamplerV2, sessionless)
# ============================================================

# --- absolute import path for core ---
import sys
from pathlib import Path
sys.path.append(str(Path("C:/Projects/cynric-genesis/core")))

import os, math, numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RZXGate, RXGate, RZGate, XGate, HGate, SXGate
from qiskit_ibm_runtime import SamplerOptions, SamplerV2

from check_backend import select_backend
from cynric_utils import (
    ensure_dir, timestamp_utc, write_json,
    auto_archive, safe_mean
)

# ------------------------------------------------------------
#  Parameters (tweak to hit 3–5 min wall-clock)
# ------------------------------------------------------------
ART_DIR   = "C:/Projects/cynric-genesis/artifacts"
ensure_dir(ART_DIR)

SHOTS     = 256              # modest, keeps queue/runtime sane
THETAS    = np.linspace(0, 2*np.pi, 9)  # 9 points: 0 ... 2π
ECHO_SPLIT = True            # implement θ as two θ/2 lobes with echo π’s

T2STAR_ENABLE   = True
T2STAR_REPEATS  = 4          # a few quick repeats (short)
T2STAR_ECHO_PI  = True       # enable mid-echo π in proxy
# (We avoid absolute delays to stay backend-agnostic & fast.)

TIMESTAMP = timestamp_utc()

# ------------------------------------------------------------
#  Helpers: counts → probabilities / expectations
# ------------------------------------------------------------
def counts_total(c: Dict[str, int]) -> int:
    return int(sum(c.values())) if c else 0

def p(c: Dict[str, int], bit: str) -> float:
    shots = counts_total(c)
    return (c.get(bit, 0) / shots) if shots else 0.0

def ex_zz(c: Dict[str, int]) -> float:
    return (p(c, "00") + p(c, "11")) - (p(c, "01") + p(c, "10"))

def ex_z0(c: Dict[str, int]) -> float:
    return (p(c, "00") + p(c, "01")) - (p(c, "10") + p(c, "11"))

def ex_z1(c: Dict[str, int]) -> float:
    return (p(c, "00") + p(c, "10")) - (p(c, "01") + p(c, "11"))

def odd_parity(c: Dict[str,int]) -> float:
    return p(c,"01") + p(c,"10")

# ------------------------------------------------------------
#  Backend + pair selection
# ------------------------------------------------------------
def pick_pair(backend) -> Tuple[int,int]:
    try:
        cmap = getattr(backend.configuration(), "coupling_map", None)
        if cmap and len(cmap) > 0:
            return tuple(cmap[0])
    except Exception:
        pass
    return (0,1)

# ------------------------------------------------------------
#  Echo-protected RZX block
#     Trick: H on q0 maps X → Z, so:
#       H(q0) · RZX(θ) · H(q0) ≈ exp(-i θ/2 Z⊗Z) up to 1Q frames
#     Echo: split θ as two θ/2 with π’s in between to cancel slow drift
# ------------------------------------------------------------
def rzx_echo_block(q0: int, q1: int, theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    # bring X⊗Z into Z⊗Z basis via H on q0
    qc.append(HGate(), [q0])

    if ECHO_SPLIT:
        half = float(theta)/2.0
        qc.append(RZXGate(half), [q0, q1])
        # spin-echo π on both to refocus quasi-static phases
        qc.append(XGate(), [q0])
        qc.append(XGate(), [q1])
        qc.append(RZXGate(half), [q0, q1])
    else:
        qc.append(RZXGate(theta), [q0, q1])

    # return basis
    qc.append(HGate(), [q0])
    return qc

# One full experiment circuit at angle θ
def rzx_echo_circuit(theta: float, pair: Tuple[int,int]) -> QuantumCircuit:
    q0, q1 = pair
    qc = QuantumCircuit(2,2)
    # If physical pair isn’t (0,1), wrap with SWAPs to keep layout trivial
    if (q0,q1) != (0,1): qc.swap(0,1)

    # pre-bias to improve sensitivity
    qc.append(SXGate(), [0]); qc.append(SXGate(), [1])
    qc.compose(rzx_echo_block(0,1,theta), qubits=[0,1], inplace=True)
    # post-balance
    qc.append(SXGate().inverse(), [0]); qc.append(SXGate().inverse(), [1])

    if (q0,q1) != (0,1): qc.swap(0,1)
    qc.measure([0,1],[0,1])
    return qc

# ------------------------------------------------------------
#  Tiny spin-echo T2* proxy (contrast improvement)
#     Not absolute T2*, just a quick “echo vs no-echo” contrast check.
# ------------------------------------------------------------
def t2star_proxy_circuits(pair: Tuple[int,int]) -> List[QuantumCircuit]:
    q0, q1 = pair
    circs = []
    for echo in (False, True):
        qc = QuantumCircuit(2,2)
        if (q0,q1)!=(0,1): qc.swap(0,1)

        # Ramsey-like: Rx(pi/2) on q0 (probe), idle synthesized by two “do-nothing” frames
        qc.append(RXGate(np.pi/2), [0])
        if echo:
            # π refocus midway
            qc.append(XGate(), [0])
            qc.append(XGate(), [1])

        # weak entangler to expose dephasing sensitivity without deep circuits
        qc.append(RZXGate(np.pi/8), [0,1])
        qc.append(RXGate(-np.pi/2), [0])

        if (q0,q1)!=(0,1): qc.swap(0,1)
        qc.measure([0,1],[0,1])
        circs.append(qc)
    return circs

# ------------------------------------------------------------
#  SamplerV2 runner (sessionless)
# ------------------------------------------------------------
def run_counts(backend, circuits: List[QuantumCircuit], shots: int) -> List[Dict[str,int]]:
    tc = transpile(circuits, backend=backend, optimization_level=3)
    opts = SamplerOptions()
    opts.default_shots = shots
    opts.environment = {"job_tags": ["cynric-qe5-echo-rzx"]}
    sampler = SamplerV2(mode=backend, options=opts)
    result = sampler.run(tc).result()

    outputs: List[Dict[str,int]] = []
    for pub in result:
        counts = {}
        data = getattr(pub, "data", None)
        if data is not None:
            meas = getattr(data, "meas", None)
            if isinstance(meas, dict):
                counts = {k: int(v) for k,v in meas.items()}
            else:
                qd = getattr(data, "quasi_dists", None)
                if qd:
                    qdict = qd[0] if isinstance(qd,(list,tuple)) else qd
                    total = sum(qdict.values())
                    if total > 0:
                        counts = {format(k,"02b"): int(round(v/total*shots)) for k,v in qdict.items()}
                        for key in ("00","01","10","11"): counts.setdefault(key,0)
        outputs.append(counts)
    return outputs

# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------
if __name__ == "__main__":
    backend = select_backend()
    if backend is None:
        raise RuntimeError("QE5 aborted: no suitable backend.")

    q0,q1 = pick_pair(backend)
    pair   = (q0,q1)
    print(f"⚡ QE5 — Echo-Calibrated RZX Scan on {backend.name}")
    print(f"   Pair={pair}, θ-points={len(THETAS)}, shots={SHOTS}, echo_split={ECHO_SPLIT}")

    # Build θ-scan circuits
    theta_list = [float(t) for t in THETAS]
    circs = [rzx_echo_circuit(t, pair) for t in theta_list]

    # Optional: tiny echo proxy
    proxy_circs = []
    if T2STAR_ENABLE:
        for _ in range(T2STAR_REPEATS):
            proxy_circs.extend(t2star_proxy_circuits(pair))

    all_circs = circs + proxy_circs

    # Execute
    counts_all = run_counts(backend, all_circs, SHOTS)

    # Split results back
    counts_theta = counts_all[:len(theta_list)]
    counts_proxy = counts_all[len(theta_list):]

    # Metrics per θ
    per_theta = []
    zz_series, z0_series, z1_series, odd_series = [], [], [], []
    for th, c in zip(theta_list, counts_theta):
        zz  = ex_zz(c)
        z0  = ex_z0(c)
        z1  = ex_z1(c)
        op  = odd_parity(c)
        per_theta.append({
            "theta": th,
            "shots": counts_total(c),
            "zz": float(zz),
            "z0": float(z0),
            "z1": float(z1),
            "odd_parity": float(op),
            "counts": c
        })
        zz_series.append(zz)
        z0_series.append(z0)
        z1_series.append(z1)
        odd_series.append(op)

    # Aggregates
    zz_series = np.array(zz_series, dtype=float)
    if len(zz_series) >= 3:
        d1 = np.diff(zz_series)
        d2 = np.diff(zz_series, n=2)
        tau     = 1.0 / (1.0 + float(np.var(d1)))            # smoothness proxy
        Lambda  = float(np.mean(np.abs(d2)))                  # curvature magnitude
    elif len(zz_series) == 2:
        d1 = np.diff(zz_series)
        tau, Lambda = 1.0/(1.0+float(np.var(d1))), 0.0
    else:
        tau, Lambda = 0.0, 0.0

    V = float(np.mean(np.abs(z0_series) + np.abs(z1_series)) / 2.0) if len(z0_series) else 0.0
    R = float(np.mean(np.abs(zz_series))) if len(zz_series) else 0.0

    # Echo proxy summary
    proxy = {"enabled": bool(T2STAR_ENABLE), "repeats": int(T2STAR_REPEATS)}
    if T2STAR_ENABLE and counts_proxy:
        # order is [no-echo, echo] repeated
        pairs = [counts_proxy[i:i+2] for i in range(0, len(counts_proxy), 2) if i+1 < len(counts_proxy)]
        delta_vis = []
        for no_echo, echo in pairs:
            # “visibility” proxy: |⟨Z0⟩|+|⟨Z1⟩|
            vis_no   = abs(ex_z0(no_echo)) + abs(ex_z1(no_echo))
            vis_echo = abs(ex_z0(echo))    + abs(ex_z1(echo))
            delta_vis.append(float(vis_echo - vis_no))
        proxy["delta_visibility_mean"] = float(safe_mean(delta_vis))
        proxy["samples"] = len(delta_vis)
    else:
        proxy["delta_visibility_mean"] = 0.0
        proxy["samples"] = 0

    # Persist artifacts
    payload = {
        "timestamp_utc": TIMESTAMP,
        "backend": backend.name,
        "pair": [int(q0), int(q1)],
        "shots": int(SHOTS),
        "thetas": theta_list,
        "echo_split": bool(ECHO_SPLIT),
        "per_theta": per_theta,
        "aggregates": {
            "V": float(V),
            "R": float(R),
            "tau": float(tau),
            "Lambda": float(Lambda)
        },
        "t2star_proxy": proxy
    }
    write_json(os.path.join(ART_DIR, "qe5_rzx_echo_scan.json"), payload)

    # Archivist action (schema for visualizer)
    action = {
        "experiment": "2025-10-QE5-Echo-Calibrated-RZX-Scan",
        "timestamp": TIMESTAMP,
        "backend": backend.name,
        "artifacts": ["qe5_rzx_echo_scan.json"],
        "V": float(V),
        "R": float(R),
        "tau": float(tau),
        "Lambda": float(Lambda),
    }
    write_json(os.path.join(ART_DIR, "genesis_action.json"), action)

    print("✅ QE5 complete.")
    print(f"   Aggregates: R≈{R:.3f}, V≈{V:.3f}, tau≈{tau:.3f}, Λ≈{Lambda:.3f}")
    if proxy["enabled"]:
        print(f"   Echo proxy Δvis≈{proxy['delta_visibility_mean']:.3f} (n={proxy['samples']})")
    print(f"   Artifacts → {ART_DIR}/qe5_rzx_echo_scan.json, genesis_action.json")

    # Auto-archive (Phase A -> B -> Visualizer)
    auto_archive("2025-10-QE5-Echo-Calibrated-RZX-Scan",
                 "Echo-protected RZX amplitude scan (θ grid) with tiny T2* proxy")