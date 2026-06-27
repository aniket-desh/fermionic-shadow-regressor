#!/usr/bin/env python3
"""Dipole resource figure: measurement savings from the FSR prior, on the two
centrosymmetric molecules whose dipole is a pure dynamical signal (H4, BeH2).

Left:  samples saved versus the flat prior, S_flat - S_FSR, against R/R_eq. The flat
       prior saturates the 40-sample budget everywhere, so this is the number of
       acquisitions the learned prior removes; the band is +/-1 std over the BO trials.
Right: the FSR prior's zero-sample relative dipole error against R/R_eq, with the 5%
       hybrid-DFT bar (Hait & Head-Gordon) and a stricter 1% bar; filled markers are
       geometries that clear 5% with zero samples.

"Samples" here are actively sampled time points -- a proxy for acquisition cost, not
circuit shots. Per-molecule curves (no cross-molecule average): LiH is heteronuclear
(permanent dipole confounds the metric) and N2 is homonuclear (symmetry-suppressed
dipole), so neither is a valid dipole target. Run from models/:
    python -m scripts.plot_dipole_resource
"""
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, DOUBLE_COL

REG = os.path.join(os.path.dirname(__file__), "..", "results", "fermionic_pipeline", "regression")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "paper_figures", "dipole_resource.pdf")

# key, label, R_eq (CCSD(T)/cc-pVTZ), plotdata path
MOLS = [
    ("h4",   "H$_4$",   0.895, "h4_regress_v13/bo_nonoracle/dipole_bo_plotdata_h4.pkl"),
    ("beh2", "BeH$_2$", 1.332, "beh2_regress_v1/bo/dipole_bo_plotdata_beh2.pkl"),
]


def main():
    apply_nature_style(usetex=False)
    base = plt.get_cmap("viridis")
    color = {"h4": base(0.75), "beh2": base(0.38)}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))
    for key, lab, req, rel in MOLS:
        P = pickle.load(open(os.path.join(REG, rel), "rb"))
        x = np.asarray(P["Rs"]) / req
        saved = np.asarray(P["flat_m"]) - np.asarray(P["fsr_m"])
        sd = np.asarray(P["fsr_e"])               # trial std of S_FSR
        err = np.asarray(P["rel_err"]) * 100.0
        tol = float(P["rel_tol"]) * 100.0
        col = color[key]

        # Left: samples saved vs flat prior -- lollipop (discrete per-geometry, no
        # connecting line: the outcome is censored at 0 or the 40-sample budget).
        dx = -0.012 if key == "h4" else 0.012     # tiny offset so the stems don't overlap
        axA.vlines(x + dx, 0, saved, color=col, lw=0.8, alpha=0.55, zorder=2)
        axA.errorbar(x + dx, saved, yerr=sd, fmt="o", ms=2.6, color=col, ecolor=col,
                     elinewidth=0.7, capsize=1.2, zorder=4, label=lab)

        # Right: zero-sample dipole error; filled = passes 5% at zero samples
        passed = err < tol
        axB.plot(x, err, color=col, lw=1.5, zorder=3)
        axB.plot(x[passed], err[passed], lw=0, marker="o", ms=3.4, color=col, zorder=5)
        axB.plot(x[~passed], err[~passed], lw=0, marker="o", ms=3.4, mfc="white",
                 mec=col, zorder=5)

    for ax in (axA, axB):
        ax.axvline(1.0, color="0.6", ls=":", lw=0.8, zorder=1)
        ax.set_xlabel(r"$R/R_\mathrm{eq}$")
        ax.grid(True, color="0.9", lw=0.5)
        ax.set_xlim(0.5, 2.6)
    axA.axhline(40, color="0.5", ls=(0, (4, 3)), lw=0.8)
    axA.text(0.55, 40.5, "full budget", ha="left", va="bottom", color="0.5", fontsize=5)
    axA.set_ylabel("time-point acquisitions saved")
    axA.set_ylim(-2, 44)
    axA.legend(loc="center right", frameon=False, fontsize=6.5, handletextpad=0.4)
    axB.axhline(5, color="0.35", ls="--", lw=0.9)
    axB.text(0.52, 5.2, "5\\% (hybrid DFT)", color="0.35", fontsize=5, va="bottom")
    axB.axhline(1, color="0.6", ls=":", lw=0.8)
    axB.text(0.52, 1.15, "1\\%", color="0.6", fontsize=5, va="bottom")
    axB.set_yscale("log")
    axB.set_ylabel(r"zero-sample dipole error (\%)")
    fig.text(0.5, 0.005, "samples = actively sampled time points (acquisition-cost proxy); "
             "filled markers clear 5\\% at zero samples",
             ha="center", fontsize=5, color="0.4")

    fig.tight_layout(rect=[0, 0.025, 1, 1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[done] {OUT}")
    for key, lab, req, rel in MOLS:
        P = pickle.load(open(os.path.join(REG, rel), "rb"))
        n = int((np.asarray(P["rel_err"]) < P["rel_tol"]).sum())
        print(f"  {key:5s} zero-sample passes {n}/26  max saved "
              f"{(np.asarray(P['flat_m'])-np.asarray(P['fsr_m'])).max():.0f}")


if __name__ == "__main__":
    main()
