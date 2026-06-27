#!/usr/bin/env python3
"""Paper-styled re-render of the RunPod revision diagnostics (3B + 3C).

freq_recovery.pdf (T4): FSR nearest-Bohr-line frequency error vs the co-dominant line
    spacing delta-omega, per molecule, log-log, with the error=spacing reference. Points
    below the line mean the FSR resolves the co-dominant lines.
smoothness.pdf (T2): geometry-smoothness S_R(R)=mean_mu||d/dR <Gamma_mu>|| per molecule,
    the empirical Proposition-1 constant; peaks fall on physical features (N2 avoided
    crossing, H4 compression, H2O onset).

Reads the revision npz arrays. Run from models/:
    python -m scripts.plot_diagnostics
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, DOUBLE_COL, ONEHALF_COL

RR = os.path.join(os.path.dirname(__file__), "..", "results", "fermionic_pipeline",
                  "regression", "revision")
ODIR = os.path.join(os.path.dirname(__file__), "..", "results", "paper_figures")

MOLS = [("h4", "H$_4$"), ("n2", "N$_2$"), ("beh2", "BeH$_2$"), ("lih", "LiH")]
N2_CROSSING = 2.30      # from 3C summary (S_R peak)


def colors():
    base = plt.get_cmap("viridis")
    return {m[0]: base(c) for m, c in zip(MOLS, np.linspace(0.75, 0.20, len(MOLS)))}


def freq_recovery():
    col = colors()
    fig, ax = plt.subplots(figsize=(ONEHALF_COL, ONEHALF_COL * 0.72))
    lo, hi = 1e-3, 1.0
    ax.plot([lo, hi], [lo, hi], color="0.6", ls="--", lw=0.8, zorder=1)
    ax.text(0.18, 0.18, "error = spacing", rotation=45, color="0.5", fontsize=5,
            ha="center", va="center", transform=ax.transAxes)
    for key, lab in MOLS:
        p = os.path.join(RR, "3B_freqrec", f"freq_recovery_{key}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        ax.plot(d["dw_strong"], d["nearest_line_dist"], lw=0, marker="o", ms=2.6,
                color=col[key], alpha=0.8, label=lab, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"co-dominant line spacing $\delta\omega$ ($E_\mathrm{h}$)")
    ax.set_ylabel(r"FSR nearest-line freq.\ error ($E_\mathrm{h}$)")
    ax.set_title("Recovered Bohr frequencies vs. line spacing")
    ax.grid(True, which="major", color="0.88", lw=0.5)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ODIR, "freq_recovery.pdf"))
    plt.close(fig)
    print("[done] freq_recovery.pdf")


def smoothness():
    col = colors()
    fig, (axR, axt) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.40))
    extra = {"h2o": "H$_2$O"}
    for key, lab in MOLS + [("h2o", "H$_2$O")]:
        p = os.path.join(RR, "3C_smoothness", f"smoothness_{key}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        c = col.get(key, "#b15928")
        axR.plot(d["R"], d["S_R"], color=c, lw=1.4, label=lab)
        axt.plot(d["R"], d["S_t"], color=c, lw=1.4, label=lab)
    axR.axvline(N2_CROSSING, color="#d62728", ls=":", lw=1.0)
    axR.text(N2_CROSSING, axR.get_ylim()[1] * 0.96, " N$_2$ crossing", color="#d62728",
             fontsize=5, va="top", ha="left", rotation=90)
    for ax, ttl, yl in [(axR, "Geometry smoothness $S_R(R)$", r"$\langle\|\partial_R\langle\Gamma\rangle\|\rangle$"),
                        (axt, "Temporal bandwidth $S_t(R)$", r"$\langle\|\partial_t\langle\Gamma\rangle\|\rangle$")]:
        ax.set_xlabel(r"$R$ (\AA)")
        ax.set_ylabel(yl)
        ax.set_title(ttl)
        ax.grid(True, color="0.9", lw=0.5)
    axR.legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(ODIR, "smoothness.pdf"))
    plt.close(fig)
    print("[done] smoothness.pdf")


def main():
    apply_nature_style(usetex=False)
    os.makedirs(ODIR, exist_ok=True)
    freq_recovery()
    smoothness()


if __name__ == "__main__":
    main()
