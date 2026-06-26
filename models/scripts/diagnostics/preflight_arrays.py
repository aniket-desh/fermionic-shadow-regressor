"""B9 3D (W3) — per-molecule pre-flight diagnostic arrays.

For each molecule across its R scan, dump bw99(R) (99%-power bandwidth ~ omega_op),
dw_strong(R) (min spacing among co-dominant >=5%-power Bohr lines), n_strong(R), and
t_need(R)=2pi/dw_strong (the horizon needed to resolve the co-dominant pair = the
'congestion threshold'). Source/cross-check for Local Claude's main-text pre-flight figure.
Code pending Aniket review for dev.
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fermionic_pipeline.data.line_spectrum_preflight import (
    build_molecule, build_majorana_matrices, line_spectrum, aggregate_lines, dominant_stats,
)

# frozen grids (R range from run_molecules_runpod.sh); coarse scan step for the arrays
GRIDS = {
    "h4":   dict(r=(0.50, 3.00), step=0.05),
    "lih":  dict(r=(1.00, 3.20), step=0.05),
    "beh2": dict(r=(1.00, 3.00), step=0.05),
    "n2":   dict(r=(0.90, 2.30), step=0.05),
}
OUT = "results/fermionic_pipeline/regression/revision/3D_preflight"
os.makedirs(OUT, exist_ok=True)

def run(mol, r, step):
    Rs = np.round(np.arange(r[0], r[1] + 1e-9, step), 3)
    bw99 = np.full(len(Rs), np.nan); dws = np.full(len(Rs), np.nan)
    nstr = np.zeros(len(Rs), int); tneed = np.full(len(Rs), np.nan)
    gammas = None
    for i, R in enumerate(Rs):
        H, nq, ne = build_molecule(mol, float(R))
        if gammas is None:
            gammas = build_majorana_matrices(nq)
        ls = line_spectrum(H, ne, gammas)
        fr, w = aggregate_lines(ls["freqs"], ls["weights"], round_dp=4)
        st = dominant_stats(fr, w, bw_mass=0.99, strong_frac=0.05)
        bw99[i] = st["bw99"]; nstr[i] = st["n_strong"]
        d = st["dw_strong"]; dws[i] = d
        tneed[i] = (2 * np.pi / d) if np.isfinite(d) and d > 0 else np.inf
    np.savez(f"{OUT}/preflight_{mol}.npz", R=Rs, bw99=bw99, dw_strong=dws,
             n_strong=nstr, t_need=tneed)
    summ = dict(mol=mol, R_min=float(Rs.min()), R_max=float(Rs.max()), n=len(Rs),
                bw99_min=float(np.nanmin(bw99)), bw99_max=float(np.nanmax(bw99)),
                dw_strong_min=float(np.nanmin(dws)), t_need_max=float(np.nanmax(tneed[np.isfinite(tneed)])) if np.any(np.isfinite(tneed)) else None,
                n_strong_range=[int(nstr.min()), int(nstr.max())])
    print(f"{mol}: bw99 {summ['bw99_min']:.3f}-{summ['bw99_max']:.3f} | "
          f"dw_strong_min {summ['dw_strong_min']:.4f} | t_need_max {summ['t_need_max']} | n_strong {summ['n_strong_range']}")
    return summ

if __name__ == "__main__":
    mols = sys.argv[1:] or list(GRIDS)
    allsum = {m: run(m, **GRIDS[m]) for m in mols}
    json.dump(allsum, open(f"{OUT}/preflight_summary.json", "w"), indent=2)
    print("[ok] ->", OUT)
