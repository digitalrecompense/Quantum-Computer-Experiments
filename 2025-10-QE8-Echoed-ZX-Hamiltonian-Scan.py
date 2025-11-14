# ============================================================
#  2025-10-QE8-Echoed-ZX-Hamiltonian-Scan  (v2, ISA-safe)
#  RAiTHE INDUSTRIES INC.© 2025 — Project Cynric / Genesis Epoch
#
#  Goal:
#    Estimate effective ZX coupling via echoed RZX fragments.
#    Sweep repetition count m across two echo styles and
#    two θ-settings; fit parity/ZZ oscillations.
#
#  Design:
#    - Sessionless SamplerV2 (Open Plan)
#    - Echo styles: 'symmetric', 'asymmetric'
#    - θ-set: {π/6, π/3}
#    - Extended m-grid (~4–5 min runtime)
#    - Fully compliant with IBM ISA validation (post-2024)
#
#  Artifacts:
#    - artifacts/qe8_echoed_zx_scan.json
#    - artifacts/genesis_action.json
#  Auto-archive: yes
# ============================================================

from __future__ import annotations
import os, sys, math, json
from pathlib import Path
import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RZXGate, RXGate
from qiskit_ibm_runtime import SamplerV2, SamplerOptions

# --- absolute import path for core ---
sys.path.append(str(Path("C:/Projects/cynric-genesis/core")))
from check_backend import select_backend
from cynric_utils import ensure_dir, write_json, timestamp_utc, echo_summary, auto_archive

# ------------------------------------------------------------
# Tunables (~4–5 min on open plan)
# ------------------------------------------------------------
ART_DIR   = "C:/Projects/cynric-genesis/artifacts"
ensure_dir(ART_DIR)

SHOTS     = 4096
M_LIST    = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
THETA_SET = [math.pi/6, math.pi/3]
ECHO_MODES = ["symmetric", "asymmetric"]
IDLE_DT   = 0

SEED_TRANS  = 42
LAYOUT_METHOD  = "sabre"
ROUTING_METHOD = "sabre"

TIMESTAMP = timestamp_utc()

# ------------------------------------------------------------
# Helpers & Metrics
# ------------------------------------------------------------
def pick_pair(backend):
    """Pick the first connected pair from coupling map; fallback (0,1)."""
    try:
        cmap = getattr(backend.configuration(), "coupling_map", None) or []
        if cmap:
            a, b = cmap[0]
            return int(a), int(b)
    except Exception:
        pass
    return 0, 1

def counts_total(c): 
    return int(sum(c.values())) if c else 0

def prob(c, b):
    s = counts_total(c)
    return (c.get(b, 0) / s) if s else 0.0

def odd_parity(c):
    return prob(c, "01") + prob(c, "10")

def expval_ZZ(c):
    p00 = prob(c, "00"); p01 = prob(c, "01")
    p10 = prob(c, "10"); p11 = prob(c, "11")
    return (p00 + p11) - (p01 + p10)

def fft_peak_freq(y):
    """Return normalized frequency index of largest non-DC FFT component."""
    y = np.asarray(y, dtype=float)
    if len(y) < 4:
        return 0.0
    y = y - y.mean()
    spec = np.abs(np.fft.rfft(y))
    if len(spec) <= 2:
        return 0.0
    spec[0] = 0.0
    k = int(np.argmax(spec))
    return float(k) / (len(y) - 1)

# ------------------------------------------------------------
# Echoed RZX fragment family
# ------------------------------------------------------------
def echoed_rzx_block(qc: QuantumCircuit, theta: float, q0=0, q1=1, mode="symmetric", idle_dt=0):
    """ZX-preserving echo block."""
    if mode == "symmetric":
        qc.append(RZXGate(theta/2), [q0, q1])
        qc.x(q0); qc.x(q1)
        if idle_dt > 0:
            qc.delay(idle_dt, q0, unit="dt"); qc.delay(idle_dt, q1, unit="dt")
        qc.append(RZXGate(-theta/2), [q0, q1])
        qc.x(q0); qc.x(q1)
    else:
        qc.append(RZXGate(theta/2), [q0, q1])
        qc.x(q0)
        if idle_dt > 0:
            qc.delay(idle_dt, q0, unit="dt"); qc.delay(idle_dt, q1, unit="dt")
        qc.append(RZXGate(-theta/2), [q0, q1])
        qc.x(q0)

def zx_probe_circuit(m: int, theta: float, echo_mode: str, q0=0, q1=1, idle_dt=0) -> QuantumCircuit:
    """Prepare |++>, apply m echoed fragments, unwrap to Z, measure."""
    qc = QuantumCircuit(2, 2)
    qc.append(RXGate(np.pi/2), [q0])
    qc.append(RXGate(np.pi/2), [q1])
    for _ in range(int(m)):
        echoed_rzx_block(qc, theta, q0=q0, q1=q1, mode=echo_mode, idle_dt=idle_dt)
    qc.append(RXGate(-np.pi/2), [q0])
    qc.append(RXGate(-np.pi/2), [q1])
    qc.measure([q0, q1], [0, 1])
    return qc

# ------------------------------------------------------------
# Sampler Run (ISA-safe)
# ------------------------------------------------------------
def run_sampler(backend, circuits, shots):
    """Transpile ISA-safe and execute via SamplerV2."""
    tc = transpile(
        circuits,
        backend=backend,
        optimization_level=1,   # ensures RX→native SX/U mapping
        seed_transpiler=SEED_TRANS,
        layout_method=LAYOUT_METHOD,
        routing_method=ROUTING_METHOD,
    )

    opts = SamplerOptions()
    opts.default_shots = shots
    opts.environment = {"job_tags": ["cynric-qe8-isa"]}

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
                    q = qd[0] if isinstance(qd, (list, tuple)) else qd
                    total = float(sum(q.values())) or 1.0
                    counts = {format(k, "02b"): int(round(v / total * shots)) for k, v in q.items()}
        for key in ("00", "01", "10", "11"):
            counts.setdefault(key, 0)
        outs.append(counts)
    return outs

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    backend = select_backend()
    if backend is None:
        raise RuntimeError("QE8 aborted: no suitable backend available.")

    q0, q1 = pick_pair(backend)
    print(f"⚡ QE8 — Echoed-ZX Hamiltonian Scan (v2) on {backend.name}")
    print(f"   Pair=({q0},{q1}), m={len(M_LIST)}, θ-set={len(THETA_SET)}, echo={ECHO_MODES}, shots={SHOTS}")

    # Build circuit set
    labels, circs = [], []
    for mode in ECHO_MODES:
        for theta in THETA_SET:
            for m in M_LIST:
                labels.append((mode, float(theta), int(m)))
                circs.append(zx_probe_circuit(m=int(m), theta=float(theta), echo_mode=mode, q0=q0, q1=q1, idle_dt=IDLE_DT))

    counts_all = run_sampler(backend, circs, SHOTS)

    # Aggregate
    per_setting = {}
    for (mode, theta, m), c in zip(labels, counts_all):
        odd = float(odd_parity(c))
        zz  = float(expval_ZZ(c))
        key = (mode, float(theta))
        bucket = per_setting.setdefault(key, {"per_m": [], "odd_series": [], "zz_series": []})
        bucket["per_m"].append({"m": int(m), "shots": counts_total(c), "odd_parity": odd, "ZZ": zz})
        bucket["odd_series"].append(odd)
        bucket["zz_series"].append(zz)

    results = []
    for (mode, theta), pack in per_setting.items():
        odd_series, zz_series = pack["odd_series"], pack["zz_series"]
        f_odd = fft_peak_freq(odd_series)
        f_zz  = fft_peak_freq(zz_series)
        amp_odd = float((np.max(odd_series)-np.min(odd_series))/2.0) if odd_series else 0.0
        amp_zz  = float((np.max(zz_series)-np.min(zz_series))/2.0) if zz_series else 0.0
        V_mean  = float(np.mean(np.abs(1.0-2.0*np.asarray(odd_series)))) if odd_series else 0.0
        R_mean  = float(np.mean(np.abs(zz_series))) if zz_series else 0.0
        tau_mean = 1.0/(1.0+float(np.var(np.diff(zz_series)))) if len(zz_series)>1 else 0.0
        Lambda_mean = float(np.mean(np.abs(np.diff(zz_series, n=2)))) if len(zz_series)>2 else 0.0
        zx_strength_proxy = float(f_odd / max(1e-9, theta))
        results.append({
            "echo_mode": mode,
            "theta": float(theta),
            "oscillation": {
                "freq_odd_cyc_per_step": f_odd,
                "freq_ZZ_cyc_per_step": f_zz,
                "amp_odd": amp_odd,
                "amp_ZZ": amp_zz,
                "ZX_strength_proxy": zx_strength_proxy,
            },
            "rollups": {
                "V_mean": V_mean,
                "R_mean": R_mean,
                "tau_mean": tau_mean,
                "Lambda_mean": Lambda_mean,
            }
        })

    # Select best for archivist
    best_key = max(results, key=lambda r: abs(r["rollups"]["R_mean"]) + 0.1*r["oscillation"]["amp_odd"], default=None)

    payload = {
        "timestamp_utc": TIMESTAMP,
        "backend": backend.name,
        "pair": [int(q0), int(q1)],
        "shots": int(SHOTS),
        "idle_dt": int(IDLE_DT),
        "m_list": M_LIST,
        "theta_set": [float(t) for t in THETA_SET],
        "echo_modes": list(ECHO_MODES),
        "results": results,
        "selection_for_visual": {
            "echo_mode": best_key["echo_mode"] if best_key else None,
            "theta": best_key["theta"] if best_key else None,
        }
    }
    art_path = os.path.join(ART_DIR, "qe8_echoed_zx_scan.json")
    write_json(art_path, payload)

    if best_key:
        r = best_key["rollups"]
        action = {
            "experiment": "2025-10-QE8-Echoed-ZX-Hamiltonian-Scan",
            "timestamp": TIMESTAMP,
            "backend": backend.name,
            "artifacts": ["qe8_echoed_zx_scan.json"],
            "V_mean": r["V_mean"], "R_mean": r["R_mean"],
            "tau_mean": r["tau_mean"], "Lambda_mean": r["Lambda_mean"],
            "notes": f"Echoed ZX parity vs m; mode={best_key['echo_mode']}, theta={best_key['theta']:.3f}."
        }
    else:
        action = {
            "experiment": "2025-10-QE8-Echoed-ZX-Hamiltonian-Scan",
            "timestamp": TIMESTAMP,
            "backend": backend.name,
            "artifacts": ["qe8_echoed_zx_scan.json"],
            "V_mean": 0.0, "R_mean": 0.0, "tau_mean": 0.0, "Lambda_mean": 0.0,
            "notes": "No best setting found."
        }
    write_json(os.path.join(ART_DIR, "genesis_action.json"), action)

    print(f"   Settings evaluated: echo={ECHO_MODES}, theta_set={[f'{t:.3f}' for t in THETA_SET]}")
    if best_key:
        echo_summary("QE8 (selected) oscillation", best_key["oscillation"])
        echo_summary("QE8 (selected) rollups", best_key["rollups"])
        print(f"✅ QE8 complete. Selected: mode={best_key['echo_mode']}, θ={best_key['theta']:.3f}")
    else:
        print("✅ QE8 complete (no selection).")

    print(f"   Artifacts → {art_path}, genesis_action.json")

    auto_archive(
        name="2025-10-QE8-Echoed-ZX-Hamiltonian-Scan",
        purpose="Echoed-RZX parity/ZZ oscillations vs repetition across echo modes & θ-set"

    )
