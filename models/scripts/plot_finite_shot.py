#!/usr/bin/env python3
"""Finite-shot robustness + matchgate-library (N_Q) sensitivity, on LiH (peer-review E1/E4).

Left: held-out temporal Pearson of an FSR trained on S Born samples per (Q,R,t), vs S,
    against the exact (S=inf) reference (dashed). Graceful, monotone-in-shots degradation;
    S=1000 recovers exact-quality predictions.
Right: same vs the matchgate-library size N_Q, against the standard N_Q=500 (dotted); recovery
    degrades monotonically below the reference library.

H4 is GRID-limited at the coarse grid this sweep affords (its dense high-frequency spectrum
aliases; S=inf already ~0.02 before any shot noise), so its curves carry no shot/library
information and are reported in the text, not plotted. The omega_op ceiling is taken from the
clean diagonalization pre-flight, never re-derived from noisy shadows (re-deriving it inflates
the bandwidth and collapses training).

Reads the B10 JSONs. Run from models/:
    python -m scripts.plot_finite_shot
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, DOUBLE_COL

B = os.path.join(os.path.dirname(__file__), "..", "results", "fermionic_pipeline",
                 "regression", "revision", "B10_finite_shot_nq")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "paper_figures",
                   "finite_shot.pdf")
LIH = plt.get_cmap("viridis")(0.20)


def main():
    apply_nature_style(usetex=False)
    fs = json.load(open(os.path.join(B, "finite_shot_lih.json")))
    nq = json.load(open(os.path.join(B, "nq_sweep_lih.json")))

    fig, (axS, axN) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.40))

    # --- finite-shot ---
    S = np.array([10, 100, 1000])
    yS = np.array([fs["10"], fs["100"], fs["1000"]])
    axS.axhline(fs["inf"], color="0.5", ls="--", lw=0.9)
    axS.text(11, fs["inf"] + 0.02, "exact ($S=\\infty$)", color="0.45", fontsize=5, va="bottom")
    axS.plot(S, yS, color=LIH, lw=1.8, marker="o", ms=4, label="LiH")
    axS.set_xscale("log")
    axS.set_xlabel("Born samples per $(Q,R,t)$, $S$")
    axS.set_ylabel("held-out temporal Pearson $r$")
    axS.set_title("Shot-noise robustness")
    axS.set_ylim(0, 1.05)

    # --- N_Q sweep ---
    Q = np.array([50, 100, 250, 500, 1000])
    yQ = np.array([nq["50"], nq["100"], nq["250"], nq["500"], nq["1000"]])
    axN.axvline(500, color="0.6", ls=":", lw=0.9)
    axN.text(500, 0.04, " reference", color="0.5", fontsize=5, rotation=90, va="bottom")
    axN.plot(Q, yQ, color=LIH, lw=1.8, marker="o", ms=4, label="LiH")
    axN.set_xscale("log")
    axN.set_xlabel("matchgate library size $N_Q$")
    axN.set_ylabel("held-out temporal Pearson $r$")
    axN.set_title("Library-size sensitivity")
    axN.set_ylim(0, 1.05)

    for ax in (axS, axN):
        ax.grid(True, which="major", color="0.9", lw=0.5)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.text(0.5, 0.005, "LiH; H$_4$ is grid-limited at this coarse grid (reported in text)",
             ha="center", fontsize=5, color="0.4")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[done] {OUT}")
    print(f"  finite-shot LiH: S=10 {fs['10']:.2f}  S=100 {fs['100']:.2f}  "
          f"S=1000 {fs['1000']:.2f}  S=inf {fs['inf']:.2f}")
    print(f"  N_Q LiH: " + "  ".join(f"{k}={nq[k]:.2f}" for k in ["50","100","250","500","1000"]))


if __name__ == "__main__":
    main()
