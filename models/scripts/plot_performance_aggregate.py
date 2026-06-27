#!/usr/bin/env python3
"""Held-out temporal correlation vs normalized bond length, over 4 molecules x 5 seeds
(peer-review E4 + E6 redesign).

Each `--seed` reseeds BOTH weight init and the 20% geometry split, so seeds hold out
different geometries; we therefore pool points into R/R_eq bins rather than aligning
per geometry.

Top: small multiples, one per molecule, all 5 training seeds {42,1729,7,13,101} overlaid
(faint) with a binned-median line; the pre-flight-congested N2 region is shaded. The
seed-13 training-stability failure (n2 collapse, h4 degraded) is shown as-is.
Bottom: aggregate median across molecules and seeds with a shaded inter-quartile band
(robust to the one collapsed seed) -- this is cross-system + cross-seed spread, NOT
estimator uncertainty.

Reads the B9 multi-seed eval JSONs (revision/1A). Run from models/:
    python -m scripts.plot_performance_aggregate
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, DOUBLE_COL

HERE = os.path.dirname(__file__)
A1 = os.path.join(HERE, "..", "results", "fermionic_pipeline", "regression", "revision", "1A")
OUT = os.path.join(HERE, "..", "results", "paper_figures", "performance_aggregate.pdf")

R_EQ = {"h4": 0.895, "lih": 1.609, "beh2": 1.332, "n2": 1.114}   # CCSD(T)/cc-pVTZ
MOLS = [("h4", "H$_4$"), ("n2", "N$_2$"), ("beh2", "BeH$_2$"), ("lih", "LiH")]
SEEDS = [42, 1729, 7, 13, 101]
CONGEST = {"n2": (2.10, 2.40)}       # pre-flight-congested bond range (Angstrom)
BINS = np.arange(0.55, 2.65, 0.15)
YLIM = (-0.05, 1.03)


def load(key):
    """All (R/R_eq, pearson) points pooled over the 5 seeds for one molecule."""
    xs, ps = [], []
    for sd in SEEDS:
        p = os.path.join(A1, f"{key}_s{sd}_model", "eval", "regressor_eval.json")
        if not os.path.exists(p):
            continue
        for r in json.load(open(p))["results"]:
            xs.append(r["R"] / R_EQ[key])
            ps.append(r["pearson_mean"])
    return np.array(xs), np.array(ps)


def binned(x, y, stat):
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    out = np.full(len(centers), np.nan)
    for i, (a, b) in enumerate(zip(BINS[:-1], BINS[1:])):
        v = y[(x >= a) & (x < b)]
        if len(v):
            out[i] = stat(v)
    return centers, out


def main():
    apply_nature_style(usetex=False)
    base = plt.get_cmap("viridis")
    color = {m[0]: base(c) for m, c in zip(MOLS, np.linspace(0.75, 0.20, len(MOLS)))}
    data = {k: load(k) for k, _ in MOLS}

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.62))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.12)

    # ---- top: per-molecule small multiples, 5 seeds overlaid ----
    for i, (key, lab) in enumerate(MOLS):
        ax = fig.add_subplot(gs[0, i])
        x, p = data[key]
        ax.axvline(1.0, color="0.6", ls=":", lw=0.8, zorder=1)
        if key in CONGEST:
            lo, hi = np.array(CONGEST[key]) / R_EQ[key]
            ax.axvspan(lo, hi, color="#d62728", alpha=0.12, zorder=0)
        ax.plot(x, p, marker="o", lw=0, ms=1.3, color=color[key], alpha=0.30, zorder=2)
        c, med = binned(x, p, np.median)
        ax.plot(c, med, color=color[key], lw=1.6, zorder=4)
        ax.set_title(lab, fontsize=7)
        ax.set_ylim(*YLIM); ax.set_xlim(0.5, 2.6)
        ax.grid(True, color="0.9", lw=0.4)
        ax.set_xlabel(r"$R/R_\mathrm{eq}$", fontsize=6)
        if i == 0:
            ax.set_ylabel("held-out $r$ (5 seeds)", fontsize=6)
        else:
            ax.set_yticklabels([])

    # ---- bottom: aggregate median + IQR over molecules AND seeds ----
    axg = fig.add_subplot(gs[1, :])
    allx = np.concatenate([data[k][0] for k, _ in MOLS])
    ally = np.concatenate([data[k][1] for k, _ in MOLS])
    c, med = binned(allx, ally, np.median)
    _, q1 = binned(allx, ally, lambda v: np.percentile(v, 25))
    _, q3 = binned(allx, ally, lambda v: np.percentile(v, 75))
    axg.axvline(1.0, color="0.6", ls=":", lw=0.9)
    for k, _ in MOLS:
        axg.plot(data[k][0], data[k][1], marker="o", lw=0, ms=1.3, color=color[k],
                 alpha=0.22, zorder=2)
    v = np.isfinite(med)
    axg.fill_between(c[v], q1[v], q3[v], color="0.25", alpha=0.15, zorder=3,
                     label="inter-quartile (molecules \\& seeds)")
    axg.plot(c[v], med[v], color="0.1", lw=2.0, marker="o", ms=3.2, zorder=5,
             label="median (molecules \\& seeds)")
    axg.text(1.0, YLIM[0] + 0.06, " equilibrium", rotation=90, va="bottom", ha="left",
             color="0.5", fontsize=5)
    axg.set_xlabel(r"$R/R_\mathrm{eq}$")
    axg.set_ylabel("held-out temporal Pearson $r$")
    axg.set_ylim(*YLIM); axg.set_xlim(0.5, 2.6)
    axg.grid(True, color="0.9", lw=0.5)
    axg.legend(loc="lower left", frameon=False, fontsize=6)

    fig.suptitle("Held-out temporal correlation versus normalized bond length "
                 "(4 molecules, 5 seeds)", y=0.98)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[done] {OUT}")
    for cc, m, a, b in zip(c, med, q1, q3):
        if np.isfinite(m):
            print(f"  R/Req~{cc:.2f}: median={m:.3f}  IQR=[{a:.3f},{b:.3f}]")


if __name__ == "__main__":
    main()
