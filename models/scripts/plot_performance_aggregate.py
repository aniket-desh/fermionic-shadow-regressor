#!/usr/bin/env python3
"""Aggregate held-out performance vs distance from equilibrium, over the 4 molecules.

Pools the per-geometry held-out Pearson r of {H4, LiH, BeH2, N2} onto a common
R/R_eq axis, bins it, and plots the mean OVER MOLECULES with a +/-1 std band -- the
spread ACROSS molecules being the cross-system uncertainty (Luis's idea). Faint raw
per-molecule points behind. This summarizes "performance over different systems".

R_eq below are the documented calculated equilibria (CCSD(T)/cc-pVTZ, briefing #5);
they sit within ~1.5% of NIST CCCBDB experimental values for N2/LiH/BeH2.

Run from models/:  python -m scripts.plot_performance_aggregate
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, ONEHALF_COL

HERE = os.path.dirname(__file__)
JDIR = os.path.join(HERE, "..", "results", "paper_figures", "overlay_jsons")
OUT = os.path.join(HERE, "..", "results", "paper_figures", "performance_aggregate.pdf")

R_EQ = {"h4": 0.895, "lih": 1.609, "beh2": 1.332, "n2": 1.114}   # CCSD(T)/cc-pVTZ (briefing #5)
MOLS = [("h4", "H$_4$"), ("n2", "N$_2$"), ("beh2", "BeH$_2$"), ("lih", "LiH")]
BINS = np.arange(0.55, 2.65, 0.15)


def load(key):
    r = json.load(open(os.path.join(JDIR, f"{key}.json")))["results"]
    R = np.array([d["R"] for d in r])
    p = np.array([d["pearson_mean"] for d in r])
    return R / R_EQ[key], p


def main():
    apply_nature_style(usetex=False)
    base = plt.get_cmap("viridis")
    color = {m[0]: base(c) for m, c in zip(MOLS, np.linspace(0.75, 0.20, len(MOLS)))}
    data = {k: load(k) for k, _ in MOLS}

    centers = 0.5 * (BINS[:-1] + BINS[1:])
    means, stds, ns = [], [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        permol = [data[k][1][(data[k][0] >= lo) & (data[k][0] < hi)].mean()
                  for k, _ in MOLS if ((data[k][0] >= lo) & (data[k][0] < hi)).any()]
        permol = np.array(permol)
        ns.append(len(permol))
        means.append(permol.mean() if len(permol) else np.nan)
        stds.append(permol.std(ddof=1) if len(permol) >= 2 else np.nan)
    means, stds, ns = np.array(means), np.array(stds), np.array(ns)

    fig, ax = plt.subplots(figsize=(ONEHALF_COL, ONEHALF_COL * 0.74))
    ax.set_axisbelow(True)
    ax.grid(True, color="0.88", lw=0.5)
    ax.axvline(1.0, color="0.6", ls=":", lw=0.9, zorder=1)
    for k, _ in MOLS:
        x, p = data[k]
        ax.plot(x, p, marker="o", lw=0, ms=2.2, color=color[k], alpha=0.35, zorder=2)
    v = ns >= 2
    ax.fill_between(centers[v], (means - stds)[v], (means + stds)[v],
                    color="0.25", alpha=0.18, zorder=3,
                    label=r"$\pm1\sigma$ across molecules")
    ax.plot(centers[ns >= 1], means[ns >= 1], color="0.1", lw=2.2, marker="o", ms=3.6,
            zorder=5, label="mean over molecules")
    ax.text(1.0, 0.34, "equilibrium", rotation=90, va="bottom", ha="right",
            color="0.5", fontsize=5)
    ax.set_xlabel(r"$R / R_\mathrm{eq}$")
    ax.set_ylabel("Held-out temporal Pearson $r$")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Performance vs distance from equilibrium, over 4 molecules")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[done] {OUT}")
    for c, m, s, n in zip(centers, means, stds, ns):
        if n >= 1:
            print(f"  R/Req~{c:.2f}: mean r={m:.3f}  std={(s if np.isfinite(s) else 0):.3f}  n_mol={n}")


if __name__ == "__main__":
    main()
