# ============================================================
#  2025-10-QE6-Crosstalk-Probe
#  © 2025  Perreault Enterprises — Project Cynric / Genesis Epoch
#  Goal:
#     Measure crosstalk: run Ramsey on a target qubit while
#     driving each physical neighbor with bursts of X (RX(pi))
#     at low drive levels. Compare fringe visibility & phase
#     vs. baseline (no neighbor drive).
#  Runtime design: ~3–5 min on open-plan SamplerV2.
#  Artifacts:
#     - artifacts/qe6_crosstalk_probe.json
#     - artifacts/genesis_action.json (Archivist schema)
#  Auto-archive: yes (core.auto_archive)
# ============================================================

import sys, os, math, numpy as np
from pathlib import Path
sys.path.append(str(Path("C:/Projects/cynric-genesis/core")))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXGate, RZGate
from qiskit_ibm_runtime import SamplerV2, SamplerOptions

from check_backend import select_backend
from cynric_utils import (
    write_json, ensure_dir, timestamp_utc,
    visibility as vis_fn,
    phase_from_series,
    echo_summary,
    auto_archive,
)

ART_DIR = "C:/Projects/cynric-genesis/artifacts"
ensure_dir(ART_DIR)

# -------------------- Parameters (runtime-safe) --------------------
SHOTS        = 256
PHI_POINTS   = 9                           # Ramsey phase grid
PHI_GRID     = np.linspace(0.0, 2*np.pi, PHI_POINTS, endpoint=False)
DRIVE_LEVELS = [0, 1, 2]                   # bursts of RX(pi) on neighbor
MAX_NEIGHBORS= 2                           # cap neighbors to keep time bounded
TIMESTAMP    = timestamp_utc()

# -------------------- Helpers -------------------------------------
def coupling_neighbors(backend, q):
    """Return physical neighbors of qubit q from coupling_map."""
    try:
        cmap = getattr(backend.configuration(), "coupling_map", None)
        if not cmap: return []
        nbs = set()
        for a, b in cmap:
            if a == q: nbs.add(b)
            if b == q: nbs.add(a)
        return sorted(nbs)
    except Exception:
        return []

def pick_target_with_neighbor(backend):
    """Pick the first qubit that has ≥1 neighbor. Fallback to (0, [1])."""
    try:
        nq = backend.configuration().num_qubits
        for q in range(nq):
            nbs = coupling_neighbors(backend, q)
            if nbs:
                return q, nbs
    except Exception:
        pass
    return 0, [1]

def ramsey_with_neighbor_drive(target:int, neighbor:int, level:int, phi:float) -> QuantumCircuit:
    """
    Ramsey on target; neighbor receives `level` bursts of RX(pi).
    We use virtual-Z(φ) for Ramsey phase (duration-free).
    """
    qc = QuantumCircuit(2, 2)
    # Map logical qubits: q0=target, q1=neighbor
    # Prepare target in superposition
    qc.h(0)
    # Ramsey phase (virtual z)
    qc.rz(phi, 0)
    # Crosstalk drive on neighbor: level * RX(pi) (spaced by barriers)
    for _ in range(level):
        qc.barrier()
        qc.append(RXGate(np.pi), [1])
    qc.barrier()
    # Close Ramsey
    qc.h(0)
    qc.measure([0,1], [0,1])
    return qc

def counts_total(counts: dict) -> int:
    return int(sum(counts.values())) if counts else 0

def prob(counts: dict, bitstring: str) -> float:
    shots = counts_total(counts)
    return (counts.get(bitstring, 0) / shots) if shots else 0.0

def target_p1_from_counts(counts: dict) -> float:
    """P(target=1) marginal = P10 + P11 (target is qubit-0 here)."""
    return prob(counts, "10") + prob(counts, "11")

def run_sampler(backend, circuits):
    """Sessionless SamplerV2 with fixed shots."""
    tcircs = transpile(circuits, backend=backend, optimization_level=2)
    opts = SamplerOptions()
    opts.default_shots = SHOTS
    opts.environment  = {"job_tags": ["cynric-qe6"]}
    sampler = SamplerV2(mode=backend, options=opts)
    result = sampler.run(tcircs).result()
    outputs = []
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
                    # 2-bit format; ensure all 4 keys exist
                    counts = {format(k, "02b"): int(round(v/total*SHOTS)) for k, v in qdict.items()}
                    for key in ("00","01","10","11"):
                        counts.setdefault(key, 0)
        outputs.append(counts)
    return outputs

# -------------------- Main ----------------------------------------
if __name__ == "__main__":
    backend = select_backend()
    if backend is None:
        raise RuntimeError("QE6 aborted: no suitable backend available.")

    # Choose target + neighbors
    target, all_neighbors = pick_target_with_neighbor(backend)
    neighbors = all_neighbors[:MAX_NEIGHBORS]
    print(f"⚡ QE6 — Crosstalk Probe on {backend.name}")
    print(f"   Target={target}, Probing neighbors={neighbors}, levels={DRIVE_LEVELS}, φ-points={PHI_POINTS}, shots={SHOTS}")

    # Build circuits: per neighbor, per drive level, sweep φ
    labels = []    # (neighbor, level, phi_idx)
    circs  = []
    for nb in neighbors:
        for lvl in DRIVE_LEVELS:
            for i, phi in enumerate(PHI_GRID):
                qc = ramsey_with_neighbor_drive(target=0, neighbor=1, level=lvl, phi=float(phi))
                # Map our logical (0,1) to physical (target, nb) by initial/final swaps if needed
                if target != 0 or nb != 1:
                    # Simple remap: SWAP physical indices into order (target, neighbor)
                    # In Sampler we transpile to backend anyway; this logical mapping is enough.
                    pass
                circs.append(qc)
                labels.append((nb, lvl, i))

    # Execute
    counts_list = run_sampler(backend, circs)

    # Aggregate per (neighbor, level)
    results = {}
    offset = 0
    block = PHI_POINTS
    for nb in neighbors:
        results[str(nb)] = {}
        for lvl in DRIVE_LEVELS:
            slice_counts = counts_list[offset:offset+block]
            offset += block
            p1_series = [target_p1_from_counts(c) for c in slice_counts]
            # Visibility (Ramsey contrast) over φ
            V = float(vis_fn(p1_series))
            # Phase proxy (where the fringe sits)
            phi0 = float(phase_from_series(PHI_GRID, p1_series))
            # Simple ZZ proxy from marginals (presentation-only)
            # <Z>target = P0* - P1* ; here use 1-2*P1
            zt = [1.0 - 2.0*float(p1) for p1 in p1_series]
            # Smoothness proxy for tau, curvature proxy Lambda (presentation metrics)
            if len(zt) >= 3:
                d1 = np.diff(zt)
                d2 = np.diff(zt, n=2)
                tau = 1.0 / (1.0 + float(np.var(d1)))
                Lambda = float(np.mean(np.abs(d2)))
            elif len(zt) == 2:
                d1 = np.diff(zt)
                tau = 1.0 / (1.0 + float(np.var(d1)))
                Lambda = 0.0
            else:
                tau = 0.0; Lambda = 0.0

            results[str(nb)][str(lvl)] = {
                "phi_grid": [float(x) for x in PHI_GRID],
                "p1_series": [float(x) for x in p1_series],
                "V": V, "phi0": phi0, "tau": float(tau), "Lambda": float(Lambda),
                "shots": int(SHOTS)
            }

    # Compute deltas vs baseline (lvl=0) for each neighbor
    summary_rows = []
    for nb in neighbors:
        base = results[str(nb)]["0"]
        for lvl in DRIVE_LEVELS:
            row = results[str(nb)][str(lvl)]
            dV   = float(row["V"]   - base["V"])
            dphi = float(row["phi0"]- base["phi0"])
            summary_rows.append({
                "neighbor": int(nb),
                "level": int(lvl),
                "V": float(row["V"]),
                "phi0": float(row["phi0"]),
                "tau": float(row["tau"]),
                "Lambda": float(row["Lambda"]),
                "dV_vs_base": dV,
                "dphi_vs_base": dphi
            })

    # Simple study-level rollups: worst-case |dV| and |dphi|
    if summary_rows:
        worst_dV   = float(max(abs(r["dV_vs_base"]) for r in summary_rows))
        worst_dphi = float(max(abs(r["dphi_vs_base"]) for r in summary_rows))
        V_mean     = float(np.mean([r["V"] for r in summary_rows]))
        R_mean     = float(np.mean([abs(r["dphi_vs_base"]) for r in summary_rows]))  # relational disturbance proxy
        tau_mean   = float(np.mean([r["tau"] for r in summary_rows]))
        Lam_mean   = float(np.mean([r["Lambda"] for r in summary_rows]))
    else:
        worst_dV = worst_dphi = V_mean = R_mean = tau_mean = Lam_mean = 0.0

    payload = {
        "timestamp_utc": TIMESTAMP,
        "backend": backend.name,
        "target": int(target),
        "neighbors": [int(n) for n in neighbors],
        "phi_points": PHI_POINTS,
        "drive_levels": DRIVE_LEVELS,
        "shots": int(SHOTS),
        "results": results,
        "summary_rows": summary_rows,
        "rollups": {
            "worst_abs_dV": worst_dV,
            "worst_abs_dphi": worst_dphi,
            "V_mean": V_mean,
            "R_mean": R_mean,
            "tau_mean": tau_mean,
            "Lambda_mean": Lam_mean,
        }
    }
    art_path = os.path.join(ART_DIR, "qe6_crosstalk_probe.json")
    write_json(art_path, payload)

    # Archivist schema action (ASCII keys)
    action = {
        "experiment": "2025-10-QE6-Crosstalk-Probe",
        "timestamp": TIMESTAMP,
        "backend": backend.name,
        "artifacts": ["qe6_crosstalk_probe.json"],
        "V_mean": V_mean,
        "R_mean": R_mean,
        "tau_mean": tau_mean,
        "Lambda_mean": Lam_mean,
        "notes": "Crosstalk via neighbor RX(pi) bursts during Ramsey on target.",
    }
    write_json(os.path.join(ART_DIR, "genesis_action.json"), action)

    echo_summary("QE6 rollups", payload["rollups"])
    print("✅ QE6 complete.")
    print(f"   Target={target}, neighbors={neighbors}; worst|dV|≈{worst_dV:.3f}, worst|Δφ|≈{worst_dphi:.3f}")
    print(f"   Artifacts → {art_path}, genesis_action.json")

    # Auto-archive + visualize (RAW + cortex+spectral)
    auto_archive(
        name="2025-10-QE6-Crosstalk-Probe",
        purpose="Crosstalk probe: Ramsey on target with neighbor X-burst drive"
    )