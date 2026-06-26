"""B9 2B (E2.1) — KRR/GP over (R,t) -> observable baseline.

Generic kernel regression (RBF KRR) over (R, t) -> the 120 channels (PCA-reduced),
trained on the SAME split as the FSR, evaluated at held-out geometries. A generic
smoothness-prior regressor with no harmonic structure; expected to struggle on the
oscillatory time signal -> isolates the value of the FSR's explicit harmonic head.
(R,t) subsampled for tractability. Code pending Aniket review for dev.
"""
import os, sys, json
import numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fermionic_pipeline.data.exact_conditional_dataset import split_r_indices
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA

def per_geom_pearson(De, Dm):
    ps = []
    for i in range(De.shape[1]):
        a, b = De[:, i], Dm[:, i]
        if a.std() < 1e-12 or b.std() < 1e-12: continue
        ps.append(np.corrcoef(a, b)[0, 1])
    return float(np.nanmean(ps)) if ps else np.nan

def run(mol, h5path, seed=42, n_sub=4000, n_pca=24):
    with h5py.File(h5path) as f:
        R = f["R_values"][:].astype(float); t = f["times"][:].astype(float)
        D = f["expectations"][:].astype(float)              # (nR,nT,K)
    a, b = split_r_indices(len(R), 0.2, seed)
    train_idx, test_idx = (a, b) if len(a) >= len(b) else (b, a)
    nT, K = D.shape[1], D.shape[2]
    # PCA the 120 channels on training data
    Xtr_full = D[train_idx].reshape(-1, K)
    pca = PCA(n_components=n_pca).fit(Xtr_full)
    # build (R,t) -> pca-coeffs training set, subsampled
    rng = np.random.default_rng(0)
    RT = []; Y = []
    for g in train_idx:
        RT.append(np.stack([np.full(nT, R[g]), t], axis=1)); Y.append(pca.transform(D[g]))
    RT = np.concatenate(RT); Y = np.concatenate(Y)
    if len(RT) > n_sub:
        sel = rng.choice(len(RT), n_sub, replace=False); RT, Y = RT[sel], Y[sel]
    # normalize inputs
    mu, sd = RT.mean(0), RT.std(0) + 1e-9
    krr = KernelRidge(kernel="rbf", alpha=1e-3, gamma=1.0).fit((RT - mu) / sd, Y)
    held = []
    for g in test_idx:
        Xq = (np.stack([np.full(nT, R[g]), t], axis=1) - mu) / sd
        Dhat = pca.inverse_transform(krr.predict(Xq))
        held.append(dict(R=float(R[g]), pearson=per_geom_pearson(D[g], Dhat)))
    pm = float(np.nanmean([h["pearson"] for h in held]))
    print(f"{mol}: KRR(R,t) held-out pearson_mean={pm:.4f} (n_sub {len(RT)}, n_pca {n_pca})")
    return dict(mol=mol, heldout_pearson_mean=pm, n_sub=int(len(RT)), n_pca=n_pca, per_geom=held)

if __name__ == "__main__":
    R = "results/fermionic_pipeline/regression"
    OUT = f"{R}/revision/2B_gpkrr"; os.makedirs(OUT, exist_ok=True)
    for m in (sys.argv[1:] or ["h4", "lih", "beh2", "n2"]):
        o = run(m, f"{R}/{m}_regress_v1/regression_targets.h5")
        json.dump(o, open(f"{OUT}/baseline_gpkrr_{m}.json", "w"), indent=2)
    print("[ok] ->", OUT)
