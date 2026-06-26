"""B9 3E (T3, proxy pre-flight) + 3F (E3, scaling table) — both cheap/model-free.

3E: compare the oracle omega_op(R) (stored, from full active-space diagonalization) to a
    CHEAP proxy = max HF orbital-energy gap among active orbitals (no diagonalization).
    Reports correlation + ratio; if close, a cheap proxy could replace the oracle pre-flight.
3F: per-molecule scaling-table data (qubits, Hilbert dim, N_T, N_R, N_Q, datagen/train wall
    from logs where available). Peak GPU mem is measured separately.
Code pending Aniket review for dev.
"""
import os, sys, json, glob, re
import numpy as np, h5py
R = "results/fermionic_pipeline/regression"
OUT = f"{R}/revision/3EF"; os.makedirs(OUT, exist_ok=True)
MOLS = ["h4", "lih", "beh2", "n2"]

def proxy(mol):
    h5 = f"{R}/{mol}_regress_v1/regression_targets.h5"
    if not os.path.exists(h5): return None
    with h5py.File(h5) as f:
        Rv = f["R_values"][:].astype(float)
        wop = f["omega_op"][:].astype(float)              # oracle ceiling per R
        eps = f["hf_orbital_energies"][:].astype(float)   # (nR, n_active)
    # cheap proxy: spread of active HF orbital energies (max gap) — no active-space diag
    proxy_bw = eps.max(axis=1) - eps.min(axis=1)
    cor = float(np.corrcoef(proxy_bw, wop)[0, 1])
    ratio = float(np.median(wop / np.where(proxy_bw > 1e-9, proxy_bw, np.nan)))
    np.savez(f"{OUT}/proxy_{mol}.npz", R=Rv, omega_op=wop, proxy_bw=proxy_bw)
    return dict(mol=mol, corr_oracle_vs_proxy=cor, median_ratio=ratio,
                omega_op_range=[float(wop.min()), float(wop.max())],
                proxy_range=[float(proxy_bw.min()), float(proxy_bw.max())])

def scaling(mol):
    h5 = f"{R}/{mol}_regress_v1/regression_targets.h5"
    if not os.path.exists(h5): return None
    with h5py.File(h5) as f:
        nR, nT, K = f["expectations"].shape
    # datagen wall from per-geom log if present; train wall from history.json if present
    dg = None
    for lg in glob.glob("runpod_logs/*datagen*%s*.log" % mol) + glob.glob("runpod_logs/instance_%s.log" % mol):
        txt = open(lg, errors="ignore").read()
        ts = [float(x) for x in re.findall(r"time=([0-9.]+)s", txt)]
        if ts: dg = round(sum(ts), 1); break
    return dict(mol=mol, qubits=8, hilbert_dim=256, N_T=nT, N_R=nR, N_Q=500, K=K,
                datagen_geom_sum_s=dg)

if __name__ == "__main__":
    pr = {m: proxy(m) for m in MOLS}; pr = {k: v for k, v in pr.items() if v}
    sc = {m: scaling(m) for m in MOLS}; sc = {k: v for k, v in sc.items() if v}
    json.dump(pr, open(f"{OUT}/proxy_preflight.json", "w"), indent=2)
    json.dump(sc, open(f"{OUT}/scaling_table.json", "w"), indent=2)
    for m, v in pr.items():
        print(f"3E {m}: corr(oracle omega_op, HF-gap proxy)={v['corr_oracle_vs_proxy']:.3f} ratio~{v['median_ratio']:.2f}")
    print("[ok] ->", OUT)
