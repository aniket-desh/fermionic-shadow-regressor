#!/usr/bin/env python3
"""Held-out temporal correlation vs normalized bond length, over the four molecules
(peer-review E6 redesign of the old "money plot").

Top row: small multiples, one panel per molecule, shared axes, raw per-geometry
held-out Pearson r vs R/R_eq; the equilibrium is marked and the pre-flight-congested
region (N2's avoided crossing) is shaded.
Bottom: an aggregate panel showing the MEDIAN across molecules with a shaded
min--max envelope -- this is cross-system spread, NOT estimator uncertainty.

R_eq = CCSD(T)/cc-pVTZ (briefing #5). Congested R-ranges are from the line-spectrum
pre-flight (scripts/plot_preflight_diagnostic.py). Run from models/:
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
JDIR = os.path.join(HERE, "..", "results", "paper_figures", "overlay_jsons")
OUT = os.path.join(HERE, "..", "results", "paper_figures", "performance_aggregate.pdf")

R_EQ = {"h4": 0.895, "lih": 1.609, "beh2": 1.332, "n2": 1.114}   # CCSD(T)/cc-pVTZ
MOLS = [("h4", "H$_4$"), ("n2", "N$_2$"), ("beh2", "BeH$_2$"), ("lih", "LiH")]
# pre-flight-congested bond ranges (Angstrom); shaded as predicted-failure regions.
# Filled from plot_preflight_diagnostic.py output (N2's avoided crossing).
CONGEST = {"n2": (2.10, 2.40)}
BINS = np.arange(0.55, 2.65, 0.15)
YLIM = (0.0, 1.02)


def load(key):
    r = json.load(open(os.path.join(JDIR, f"{key}.json")))["results"]
    R = np.array([d["R"] for d in r])
    p = np.array([d["pearson_mean"] for d in r])
    return R, p


def main():
    apply_nature_style(usetex=False)
    base = plt.get_cmap("viridis")
    color = {m[0]: base(c) for m, c in zip(MOLS, np.linspace(0.75, 0.20, len(MOLS)))}
    data = {k: load(k) for k, _ in MOLS}

    fig = plt.figure(figsize=(DOUBLE_COL, DOUBLE_COL * 0.62))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.12)

    # ---- top: small multiples, shared axes ----
    for i, (key, lab) in enumerate(MOLS):
        ax = fig.add_subplot(gs[0, i])
        R, p = data[key]
        x = R / R_EQ[key]
        ax.axvline(1.0, color="0.6", ls=":", lw=0.8, zorder=1)
        if key in CONGEST:
            lo, hi = (np.array(CONGEST[key]) / R_EQ[key])
            ax.axvspan(lo, hi, color="#d62728", alpha=0.12, zorder=0)
        ax.plot(x, p, marker="o", lw=0.8, ms=2.0, color=color[key])
        ax.set_title(lab, fontsize=7)
        ax.set_ylim(*YLIM)
        ax.set_xlim(0.5, 2.6)
        ax.grid(True, color="0.9", lw=0.4)
        ax.set_xlabel(r"$R/R_\mathrm{eq}$", fontsize=6)
        if i == 0:
            ax.set_ylabel("held-out $r$", fontsize=6)
        else:
            ax.set_yticklabels([])

    # ---- bottom: robust aggregate (median + min-max across molecules) ----
    axg = fig.add_subplot(gs[1, :])
    centers = 0.5 * (BINS[:-1] + BINS[1:])
    med, lo_env, hi_env, nmol = [], [], [], []
    for a, b in zip(BINS[:-1], BINS[1:]):
        vals = [data[k][1][(data[k][0] / R_EQ[k] >= a) & (data[k][0] / R_EQ[k] < b)].mean()
                for k, _ in MOLS if ((data[k][0] / R_EQ[k] >= a) & (data[k][0] / R_EQ[k] < b)).any()]
        vals = np.array(vals)
        nmol.append(len(vals))
        med.append(np.median(vals) if len(vals) else np.nan)
        lo_env.append(vals.min() if len(vals) else np.nan)
        hi_env.append(vals.max() if len(vals) else np.nan)
    med, lo_env, hi_env, nmol = map(np.array, (med, lo_env, hi_env, nmol))
    v = nmol >= 2
    axg.axvline(1.0, color="0.6", ls=":", lw=0.9)
    for k, _ in MOLS:                       # faint per-molecule points behind
        x, p = data[k][0] / R_EQ[k], data[k][1]
        axg.plot(x, p, marker="o", lw=0, ms=1.8, color=color[k], alpha=0.30, zorder=2)
    axg.fill_between(centers[v], lo_env[v], hi_env[v], color="0.25", alpha=0.15, zorder=3,
                     label="min--max across molecules")
    axg.plot(centers[nmol >= 1], med[nmol >= 1], color="0.1", lw=2.0, marker="o", ms=3.4,
             zorder=5, label="median across molecules")
    axg.text(1.0, YLIM[0] + 0.04, " equilibrium", rotation=90, va="bottom", ha="left",
             color="0.5", fontsize=5)
    axg.set_xlabel(r"$R/R_\mathrm{eq}$")
    axg.set_ylabel("held-out temporal Pearson $r$")
    axg.set_ylim(*YLIM)
    axg.set_xlim(0.5, 2.6)
    axg.grid(True, color="0.9", lw=0.5)
    axg.legend(loc="lower left", frameon=False, fontsize=6)

    fig.suptitle("Held-out temporal correlation versus normalized bond length", y=0.98)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[done] {OUT}")
    for c, m, l, h, n in zip(centers, med, lo_env, hi_env, nmol):
        if n >= 1:
            print(f"  R/Req~{c:.2f}: median={m:.3f}  min={l:.3f}  max={h:.3f}  n_mol={n}")


if __name__ == "__main__":
    main()
