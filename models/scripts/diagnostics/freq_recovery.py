"""B9 3B (T4) — frequency-recovery diagnostic.

For each held-out geometry, FFT-peak-pick the FSR's PREDICTED signal D_pred(t) and compare the
recovered dominant frequencies to the exact Bohr lines (active-space diagonalization). Quantify
nearest-line distance and plot it vs the co-dominant line spacing delta-omega (error rises where
lines crowd). Uses the trained s42 FSR via predict_signal_matrix. Code pending Aniket review.
"""
import os, sys, json
import numpy as np, torch, h5py
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fermionic_pipeline.data.exact_conditional_dataset import split_r_indices
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model
from fermionic_pipeline.eval.regressor_eval import predict_signal_matrix
from fermionic_pipeline.eval.plot_regression import _get_orb_energies, _get_omega_op
from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.data.line_spectrum_preflight import (
    build_molecule, build_majorana_matrices, line_spectrum, aggregate_lines, dominant_stats,
)

R = "results/fermionic_pipeline/regression"
OUT = f"{R}/revision/3B_freqrec"; os.makedirs(OUT, exist_ok=True)

def peaks(D, t, n):
    Pw = sum(np.abs(np.fft.rfft(D[:, k] - D[:, k].mean())) ** 2 for k in range(D.shape[1]))
    w = np.fft.rfftfreq(len(t), d=float(t[1] - t[0])) * 2 * np.pi
    idx = [i for i in range(1, len(Pw) - 1) if Pw[i] > Pw[i - 1] and Pw[i] >= Pw[i + 1]]
    return np.sort(w[sorted(idx, key=lambda i: -Pw[i])[:n]])

def run(mol, dev="cuda"):
    h5 = f"{R}/{mol}_regress_v1/regression_targets.h5"
    ckpt = f"{R}/revision/1A/{mol}_s42_model/regressor.pt"
    if not os.path.exists(ckpt): print(f"  [skip] {mol}: no s42 ckpt"); return None
    handle = RegressionDatasetHandle(h5)
    model, payload = load_checkpoint_model(ckpt, device=torch.device(dev))
    from fermionic_pipeline.eval.omega_source import OmegaOpSource
    from fermionic_pipeline.eval import plot_regression as _pr
    _pr.set_omega_source(OmegaOpSource("train-interp", handle=handle, payload=payload))
    Rv = handle.R_values; t = handle.times
    a, b = split_r_indices(len(Rv), 0.2, 42); _, test_idx = (a, b) if len(a) >= len(b) else (b, a)
    gammas = None; dist = []; dws = []
    for g in test_idx:
        Rg = float(Rv[g])
        H, nq, ne = build_molecule(mol, Rg)
        if gammas is None: gammas = build_majorana_matrices(nq)
        st = dominant_stats(*aggregate_lines(*[line_spectrum(H, ne, gammas)[k] for k in ("freqs", "weights")], round_dp=4))
        exact = np.sort([x[0] for x in st["top"]]) if st["top"] else np.array([])
        if not len(exact): continue
        Dp = predict_signal_matrix(model, Rg, t, torch.device(dev),
                                   orb_energies=_get_orb_energies(handle, g), omega_op=_get_omega_op(handle, g)).T
        rec = peaks(Dp, t, len(exact))
        d = np.mean([np.min(np.abs(rec - L)) for L in exact]) if len(rec) else np.nan
        dist.append(d); dws.append(st["dw_strong"] if np.isfinite(st["dw_strong"]) else np.nan)
    dist = np.array(dist); dws = np.array(dws)
    np.savez(f"{OUT}/freq_recovery_{mol}.npz", nearest_line_dist=dist, dw_strong=dws, R=Rv[test_idx])
    print(f"{mol}: FSR freq nearest-line dist mean={np.nanmean(dist):.4f} Eh (median {np.nanmedian(dist):.4f})")
    return dict(mol=mol, nearest_line_dist_mean=float(np.nanmean(dist)), n=int(len(dist)))

if __name__ == "__main__":
    summ = {}; fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for m in (sys.argv[1:] or ["h4", "lih", "beh2", "n2"]):
        r = run(m)
        if r: summ[m] = r; d = np.load(f"{OUT}/freq_recovery_{m}.npz"); ax.scatter(d["dw_strong"], d["nearest_line_dist"], s=12, label=m)
    ax.set_xlabel(r"co-dominant line spacing $\delta\omega$"); ax.set_ylabel("FSR nearest-line freq error"); ax.legend(fontsize=7); ax.set_xscale("log"); ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(f"{OUT}/freq_recovery.pdf")
    json.dump(summ, open(f"{OUT}/freq_recovery_summary.json", "w"), indent=2)
    print("[ok] ->", OUT)
