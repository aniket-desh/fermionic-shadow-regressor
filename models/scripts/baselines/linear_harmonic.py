"""B9 1C (E2.3) — linear-harmonic baseline: FIXED frequency bank, NO learned freqs.

Tests whether the FSR's LEARNED, geometry-conditioned frequencies beat a fixed pre-flight
bank. Per TRAINING geometry, least-squares fit per-channel {a_uk, b_uk, c_u} to a FIXED
uniform frequency bank up to omega_op. Interpolate the fitted amplitudes in R (1-D) to the
held-out geometries (SAME split as the FSR: split_r_indices), reconstruct D_hat(R*,t), and
score held-out pearson_mean with the SAME metric as regressor_eval. Code pending Aniket review.
"""
import json, os, sys
import numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fermionic_pipeline.data.exact_conditional_dataset import split_r_indices

def design(t, freqs):
    cols = [np.ones_like(t)]
    for w in freqs:
        cols += [np.cos(w * t), np.sin(w * t)]
    return np.stack(cols, axis=1)  # (n_t, 1+2K)

def per_geom_pearson(De, Dm):  # De,Dm: (n_t, K) -> mean over channels of temporal pearson
    ps = []
    for i in range(De.shape[1]):
        a, b = De[:, i], Dm[:, i]
        if a.std() < 1e-12 or b.std() < 1e-12: continue
        ps.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(ps)) if ps else np.nan

def run(mol, h5path, n_freqs=None, seed=42):
    with h5py.File(h5path) as f:
        R = f["R_values"][:].astype(float); t = f["times"][:].astype(float)
        D = f["expectations"][:].astype(float)              # (nR, nT, K)
        wop = f["omega_op"][:].astype(float) if "omega_op" in f else None
    nR = len(R)
    a, b = split_r_indices(nR, 0.2, seed)
    train_idx, test_idx = (a, b) if len(a) >= len(b) else (b, a)  # test = smaller (~20%)
    wmax = float(np.nanmax(wop)) * 1.1 if wop is not None else 8.0
    if n_freqs is None:  # match the resolvable spectrum: bins up to wmax at the FFT resolution 2pi/T
        T = float(t[-1] - t[0]); n_freqs = int(np.clip(wmax * T / (2 * np.pi) + 20, 100, 600))
    freqs = np.linspace(0.0, wmax, n_freqs + 1)[1:]         # fixed uniform bank
    A = design(t, freqs)                                    # (nT, P)
    P = A.shape[1]; K = D.shape[2]
    # fit amplitudes at every training geometry: coeffs[g] = lstsq(A, D[g]) -> (P,K)
    coeffs = np.zeros((len(train_idx), P, K))
    insample = []
    for j, g in enumerate(train_idx):
        c, *_ = np.linalg.lstsq(A, D[g], rcond=None)
        coeffs[j] = c
        insample.append(per_geom_pearson(D[g], A @ c))
    # interpolate coeffs in R to held-out (vectorized over P,K), reconstruct, score
    from scipy.interpolate import interp1d
    Rtr = R[train_idx]; order = np.argsort(Rtr); Rtr_s = Rtr[order]; coeffs_s = coeffs[order]
    interp = interp1d(Rtr_s, coeffs_s, axis=0, bounds_error=False,
                      fill_value=(coeffs_s[0], coeffs_s[-1]))  # (P,K) per R
    held = []
    for g in test_idx:
        ci = interp(R[g])                                   # (P,K)
        held.append(dict(R=float(R[g]), pearson=per_geom_pearson(D[g], A @ ci)))
    pm = float(np.nanmean([h["pearson"] for h in held]))
    out = dict(mol=mol, n_freqs=n_freqs, omega_max=wmax, seed=seed,
               n_train=len(train_idx), n_test=len(test_idx),
               insample_pearson_mean=float(np.nanmean(insample)),
               heldout_pearson_mean=pm,
               heldout_pearson_median=float(np.nanmedian([h["pearson"] for h in held])),
               per_geom=held)
    print(f"{mol}: in-sample {out['insample_pearson_mean']:.4f} | HELD-OUT linharm pearson_mean={pm:.4f} (median {out['heldout_pearson_median']:.4f})")
    return out

if __name__ == "__main__":
    R = "results/fermionic_pipeline/regression"
    OUT = f"{R}/revision/1C_linharm"; os.makedirs(OUT, exist_ok=True)
    FL = dict(h4=8.0, lih=0.5, beh2=0.6, n2=1.5)
    mols = sys.argv[1:] or ["h4", "lih", "beh2", "n2"]
    for m in mols:
        o = run(m, f"{R}/{m}_regress_v1/regression_targets.h5")
        json.dump(o, open(f"{OUT}/baseline_linharm_{m}.json", "w"), indent=2)
    print("[ok] ->", OUT)
