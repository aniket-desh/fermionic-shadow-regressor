#!/usr/bin/env python3
"""B10 finalize: collect pearson-vs-S (3A finite-shot) + pearson-vs-N_Q (2D) curves
from the trained eval JSONs, write per-molecule JSON tables, and render two figures.
Deterministic; safe to re-run. Missing models are recorded as null (not skipped)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REV = "results/fermionic_pipeline/regression/revision"
FS, NQ = f"{REV}/3A_fs", f"{REV}/2D_nq"
OUT = f"{REV}/B10_finite_shot_nq"
os.makedirs(OUT, exist_ok=True)


def pm(path):
    """Held-out pearson_mean = mean over held-out geometries of the per-geometry
    pearson_mean (mean over 120 channels). None if absent/unreadable/non-finite."""
    try:
        with open(path) as f:
            res = json.load(f)["results"]
        v = float(np.mean([r["pearson_mean"] for r in res]))
        return round(v, 4) if v == v else None  # NaN -> None
    except Exception:
        return None


def fs_point(mol, S):
    tag = "Sinf" if S == "inf" else f"S{S}"
    return pm(f"{FS}/{mol}_{tag}_model/eval/regressor_eval.json")


def nq_point(mol, nq):
    if nq == 500:  # N_Q=500 reuses 3A's exact (S=inf) model
        return pm(f"{FS}/{mol}_Sinf_model/eval/regressor_eval.json")
    return pm(f"{NQ}/{mol}_nq{nq}_model/eval/regressor_eval.json")


# ---- finite-shot (E1): pearson vs shots S ---------------------------------
SVALS = ["inf", 1000, 100, 10]
fs = {}
for mol in ["lih", "h4"]:
    fs[mol] = {str(S): fs_point(mol, S) for S in SVALS}
    with open(f"{OUT}/finite_shot_{mol}.json", "w") as f:
        json.dump(fs[mol], f, indent=2)

# ---- N_Q sweep (E4): pearson vs library size N_Q --------------------------
NQS = [50, 100, 250, 500, 1000]
nq = {}
for mol in ["lih", "h4"]:
    nq[mol] = {str(n): nq_point(mol, n) for n in NQS}
    with open(f"{OUT}/nq_sweep_{mol}.json", "w") as f:
        json.dump(nq[mol], f, indent=2)

# ---- figures --------------------------------------------------------------
def _finite_x(S):  # plot S=inf as a large finite anchor on a log axis
    return 1e5 if S == "inf" else float(S)

fig, ax = plt.subplots(figsize=(5, 3.4))
for mol, c in [("lih", "C0"), ("h4", "C3")]:
    xs, ys = [], []
    for S in SVALS:
        y = fs[mol][str(S)]
        if y is not None:
            xs.append(_finite_x(S)); ys.append(y)
    ax.plot(xs, ys, "o-", color=c, label=mol)
ax.set_xscale("log"); ax.set_xlabel("shots per (Q,R,t)   [10^5 = exact marginal]")
ax.set_ylabel("held-out pearson_mean (vs exact)"); ax.set_ylim(-0.05, 1.02)
ax.set_title("E1: finite-shot degradation"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/pearson_vs_S.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 3.4))
for mol, c in [("lih", "C0"), ("h4", "C3")]:
    xs, ys = [], []
    for n in NQS:
        y = nq[mol][str(n)]
        if y is not None:
            xs.append(n); ys.append(y)
    ax.plot(xs, ys, "o-", color=c, label=mol)
ax.set_xscale("log"); ax.set_xlabel("N_Q (matchgate library size)")
ax.set_ylabel("held-out pearson_mean (vs N_Q=500 ref)"); ax.set_ylim(-0.05, 1.02)
ax.set_title("E4: N_Q sensitivity"); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/pearson_vs_nq.pdf"); plt.close(fig)

print("=== finite_shot ==="); print(json.dumps(fs, indent=2))
print("=== nq_sweep ===");   print(json.dumps(nq, indent=2))
print(f"wrote -> {OUT}/")
