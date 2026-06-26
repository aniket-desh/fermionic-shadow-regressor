"""B9 2C (E2.4) — classical per-geometry spectral estimation baseline.

At each HELD-OUT geometry, given the exact time series D_mu(t), recover dominant
frequencies by FFT peak-pick (aggregate power spectrum over the 120 channels) and compare
to the exact dominant Bohr lines (diagonalization). This is the "what can classical spectral
estimation do with the same time data, per geometry" baseline — it has NO cross-geometry
generalization, so it sets the per-geometry ceiling. Report dominant-peak recovery rate.
Code pending Aniket review for dev.
"""
import os, sys, json
import numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fermionic_pipeline.data.exact_conditional_dataset import split_r_indices
from fermionic_pipeline.data.line_spectrum_preflight import (
    build_molecule, build_majorana_matrices, line_spectrum, aggregate_lines, dominant_stats,
)

def recovered_peaks(D, t, n_peaks):
    # aggregate power spectrum over channels, pick n_peaks dominant frequencies
    P = np.zeros(len(t) // 2 + 1)
    for k in range(D.shape[1]):
        P += np.abs(np.fft.rfft(D[:, k] - D[:, k].mean())) ** 2
    w = np.fft.rfftfreq(len(t), d=float(t[1] - t[0])) * 2 * np.pi
    # local maxima
    idx = [i for i in range(1, len(P) - 1) if P[i] > P[i - 1] and P[i] >= P[i + 1]]
    idx = sorted(idx, key=lambda i: -P[i])[:n_peaks]
    return np.sort(w[idx])

def run(mol, h5path, seed=42, tol=0.05):
    with h5py.File(h5path) as f:
        R = f["R_values"][:].astype(float); t = f["times"][:].astype(float)
        D = f["expectations"][:].astype(float)
    a, b = split_r_indices(len(R), 0.2, seed)
    _, test_idx = (a, b) if len(a) >= len(b) else (b, a)
    gammas = None; rates = []
    for g in test_idx:
        Rg = float(R[g])
        H, nq, ne = build_molecule(mol, Rg)
        if gammas is None: gammas = build_majorana_matrices(nq)
        ls = line_spectrum(H, ne, gammas)
        fr, w = aggregate_lines(ls["freqs"], ls["weights"], round_dp=4)
        st = dominant_stats(fr, w, bw_mass=0.99, strong_frac=0.05)
        exact_lines = np.sort([x[0] for x in st["top"]]) if st["top"] else np.array([])
        n_strong = max(1, len(exact_lines))
        rec = recovered_peaks(D[g], t, n_strong)
        # recovery: fraction of exact strong lines matched within tol (absolute Eh) by a recovered peak
        if len(exact_lines):
            matched = sum(np.min(np.abs(rec - L)) < tol for L in exact_lines) if len(rec) else 0
            rates.append(matched / len(exact_lines))
    rate = float(np.mean(rates)) if rates else np.nan
    print(f"{mol}: classical FFT dominant-peak recovery (held-out) = {rate:.3f}  (tol {tol} Eh, n_test {len(test_idx)})")
    return dict(mol=mol, peak_recovery_rate=rate, n_test=len(test_idx), tol=tol)

if __name__ == "__main__":
    R = "results/fermionic_pipeline/regression"
    OUT = f"{R}/revision/2C_spectral"; os.makedirs(OUT, exist_ok=True)
    for m in (sys.argv[1:] or ["h4", "lih", "beh2", "n2"]):
        o = run(m, f"{R}/{m}_regress_v1/regression_targets.h5")
        json.dump(o, open(f"{OUT}/baseline_spectral_{m}.json", "w"), indent=2)
    print("[ok] ->", OUT)
