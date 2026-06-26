"""B9 3C (T2) — smoothness diagnostic: empirical Proposition-1 constant.

Finite-difference norms of the exact targets across each molecule's scan:
  S_R(R) = mean_mu || d/dR <Gamma_mu>(R, .) ||_2   (geometry-smoothness; spikes at crossings)
  S_t(R) = mean_mu || d/dt <Gamma_mu>(R, .) ||_2   (temporal-bandwidth proxy)
Highlights the N2 avoided crossing and (if present) the failed H2O stretch. Dumps arrays +
a figure. Code pending Aniket review for dev.
"""
import os, sys, json
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

R = "results/fermionic_pipeline/regression"
OUT = f"{R}/revision/3C_smoothness"; os.makedirs(OUT, exist_ok=True)

def run(mol):
    h5 = f"{R}/{mol}_regress_v1/regression_targets.h5"
    if not os.path.exists(h5): print(f"  [skip] {mol}: no h5"); return None
    with h5py.File(h5) as f:
        Rv = f["R_values"][:].astype(float); t = f["times"][:].astype(float)
        D = f["expectations"][:].astype(float)              # (nR, nT, K)
    dt = float(t[1] - t[0])
    dR_D = np.gradient(D, Rv, axis=0)                        # (nR,nT,K)
    dt_D = np.gradient(D, dt, axis=1)
    S_R = np.mean(np.linalg.norm(dR_D, axis=1), axis=1)     # (nR,)  norm over t, mean over mu
    S_t = np.mean(np.linalg.norm(dt_D, axis=1), axis=1)
    np.savez(f"{OUT}/smoothness_{mol}.npz", R=Rv, S_R=S_R, S_t=S_t)
    pk = float(Rv[int(np.argmax(S_R))])
    print(f"{mol}: S_R peak at R={pk:.2f} (max {S_R.max():.2f}, median {np.median(S_R):.2f}); S_t median {np.median(S_t):.2f}")
    return dict(mol=mol, S_R_peak_R=pk, S_R_max=float(S_R.max()), S_R_median=float(np.median(S_R)))

if __name__ == "__main__":
    mols = sys.argv[1:] or ["h4", "lih", "beh2", "n2", "h2o"]
    summ = {}
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for m in mols:
        r = run(m)
        if r is None: continue
        summ[m] = r
        d = np.load(f"{OUT}/smoothness_{m}.npz")
        axs[0].plot(d["R"], d["S_R"], label=m); axs[1].plot(d["R"], d["S_t"], label=m)
    axs[0].set_title("geometry-smoothness S_R(R)"); axs[0].set_xlabel("R"); axs[0].legend(fontsize=7)
    axs[1].set_title("temporal-bandwidth S_t(R)"); axs[1].set_xlabel("R")
    fig.tight_layout(); fig.savefig(f"{OUT}/smoothness.pdf");
    json.dump(summ, open(f"{OUT}/smoothness_summary.json", "w"), indent=2)
    print("[ok] ->", OUT)
