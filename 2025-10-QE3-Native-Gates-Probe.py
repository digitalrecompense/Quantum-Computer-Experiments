# ============================================================
#  2025-10-QE3-Native-Gates-Probe
#  RAiTHE INDUSTRIES INC.© 2025 — Project Cynric / Genesis Epoch
#  Originator: Robert R. S. Perreault
#
#  Purpose:
#     Probe native 2Q gate reliability on the selected backend using:
#       - mirror tests  (G · G⁻¹ → |00⟩ ideally)
#       - stack tests   (G^m, m ∈ {1..4})
#     Scores candidates and publishes a concise Archivist action.
#
#  Canonical Mapping:
#     New QE3 (relabels old QE4).
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
from qiskit.circuit.library import CXGate, ECRGate, RZXGate
from qiskit_ibm_runtime import SamplerV2, SamplerOptions

from check_backend import select_backend
from cynric_utils import ensure_dir, write_json, timestamp_utc, auto_archive

# ------------------------------------------------------------
#  Parameters
# ------------------------------------------------------------
ART_DIR = "C:/Projects/cynric-genesis/artifacts"
ensure_dir(ART_DIR)

SHOTS = 150
STACK_DEPTHS = [1, 2, 3, 4]
TRY_GATES = ["ecr", "cx", "rzx"]  # order matters for reporting
TIMESTAMP = timestamp_utc()

# ------------------------------------------------------------
#  Backend helpers
# ------------------------------------------------------------
def pick_pair(backend):
    """Choose first connected qubit pair; fallback to (0,1)."""
    try:
        cmap = getattr(backend.configuration(), "coupling_map", None)
        if cmap and len(cmap) > 0:
            a, b = cmap[0]
            return int(a), int(b)
    except Exception:
        pass
    return 0, 1

# ------------------------------------------------------------
#  Metrics
# ------------------------------------------------------------
def counts_total(c): return int(sum(c.values())) if c else 0
def prob(c, s):
    t = counts_total(c)
    return (c.get(s, 0) / t) if t else 0.0

def prob00(c): return prob(c, "00")

def parity_odd(c):
    """P(01)+P(10)."""
    return prob(c, "01") + prob(c, "10")

# ------------------------------------------------------------
#  Gate factory
# ------------------------------------------------------------
def gate_instance(name):
    if name == "cx":   return CXGate()
    if name == "ecr":  return ECRGate()
    if name == "rzx":  return RZXGate(np.pi/4)
    raise ValueError(f"Unsupported gate: {name}")

# ------------------------------------------------------------
#  Circuits
# ------------------------------------------------------------
def mirror_test_circuit(name, q0, q1):
    """G followed by G⁻¹; ideal return to |00⟩."""
    G = gate_instance(name)
    qc = QuantumCircuit(2, 2)
    if (q0, q1) != (0, 1):
        qc.swap(0, 1)
    qc.append(G, [0, 1])
    qc.append(G.inverse(), [0, 1])
    if (q0, q1) != (0, 1):
        qc.swap(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

def stack_test_circuit(name, q0, q1, m):
    """G applied m times; measures stability under repetition."""
    G = gate_instance(name)
    qc = QuantumCircuit(2, 2)
    if (q0, q1) != (0, 1):
        qc.swap(0, 1)
    for _ in range(int(m)):
        qc.append(G, [0, 1])
    if (q0, q1) != (0, 1):
        qc.swap(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc

# ------------------------------------------------------------
#  Support probe
# ------------------------------------------------------------
def supports_gate(backend, name) -> bool:
    """Return True if the gate transpiles on this backend."""
    try:
        test = mirror_test_circuit(name, 0, 1)
        _ = transpile(test, backend=backend, optimization_level=1)
        return True
    except Exception:
        return False

# ------------------------------------------------------------
#  Execute (sessionless SamplerV2) → counts-like
# ------------------------------------------------------------
def run_counts(backend, circuits, shots: int):
    tc = transpile(circuits, backend=backend, optimization_level=3)
    opts = SamplerOptions()
    opts.default_shots = shots
    opts.environment = {"job_tags": ["cynric-qe3"]}
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
#  Main
# ------------------------------------------------------------
if __name__ == "__main__":
    backend = select_backend()
    if backend is None:
        raise RuntimeError("QE3 aborted: no suitable backend available.")

    q0, q1 = pick_pair(backend)
    print(f"⚡ QE3 — Native Gate Probe on {backend.name} using pair ({q0},{q1})")

    # filter gates by support
    gates = [g for g in TRY_GATES if supports_gate(backend, g)]
    if not gates:
        raise RuntimeError("No candidate 2Q gates transpile on this backend.")
    print(f"   Candidate gates: {', '.join(gates)}")

    # build experiments
    labels, circs = [], []
    for g in gates:
        labels.append((g, "mirror", 0))
        circs.append(mirror_test_circuit(g, q0, q1))
        for m in STACK_DEPTHS:
            labels.append((g, "stack", m))
            circs.append(stack_test_circuit(g, q0, q1, m))

    # run
    counts_list = run_counts(backend, circs, SHOTS)

    # score
    records = []
    best = {"gate": None, "mode": None, "m": 0, "p00": -1.0, "parity_odd": 1.0, "score": -1e9}
    for (g, mode, m), c in zip(labels, counts_list):
        p = prob00(c)
        po = parity_odd(c)
        rec = {
            "gate": g,
            "mode": mode,
            "m": int(m),
            "shots": counts_total(c),
            "p00": float(p),
            "parity_odd": float(po),
            "counts": c
        }
        records.append(rec)
        # favor mirror returning to |00>, penalize odd parity; lighter penalty for stacks
        penalty = 0.5 if mode == "mirror" else 0.3
        score = p - penalty * po
        if score > best["score"]:
            best = {"gate": g, "mode": mode, "m": int(m), "p00": float(p), "parity_odd": float(po), "score": float(score)}

    # artifact (detailed)
    write_json(
        os.path.join(ART_DIR, "qe3_native_probe.json"),
        {
            "timestamp_utc": TIMESTAMP,
            "backend": backend.name,
            "pair": [int(q0), int(q1)],
            "shots": int(SHOTS),
            "records": records,
            "best": best,
        },
    )

    # Archivist action (ASCII keys only)
    action = {
        "experiment": "2025-10-QE3-Native-Gates-Probe",
        "timestamp": TIMESTAMP,
        "backend": backend.name,
        "artifacts": ["qe3_native_probe.json"],
        # Map best test to roll-up proxies
        "V": float(max(0.0, best["p00"])),         # visibility proxy
        "R": float(max(0.0, 1.0 - best["parity_odd"])),  # relational “goodness” proxy
        "tau": 0.0,
        "Lambda": 0.0,
    }
    write_json(os.path.join(ART_DIR, "genesis_action.json"), action)

    print("✅ QE3 complete.")
    print(f"   Best: {best['gate']} / {best['mode']} m={best['m']} "
          f"| p00≈{best['p00']:.3f}, odd≈{best['parity_odd']:.3f}, score≈{best['score']:.3f}")
    print(f"   Artifacts → {ART_DIR}/qe3_native_probe.json, genesis_action.json")

    # auto-archive
    auto_archive("2025-10-QE3-Native-Gates-Probe", "Native Gate Probe (Zero-Separation Kernel)")

# ZS-Link: 2025-10-QE3-Native-Gates-Probe | SU(4) contact reliability
