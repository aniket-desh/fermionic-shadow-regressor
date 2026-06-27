#!/usr/bin/env python3
"""Finite-shot robustness + matchgate-library (N_Q) sensitivity, on LiH (peer-review E1/E4).

Per-geometry MEDIAN held-out temporal Pearson over the held-out set (robust on the small coarse
grid; the mean is dragged by a couple of edge geometries, which is why finite-S can sit above the
exact MEAN but not the exact median).

Left: median held-out r of an FSR trained on S Born samples per (Q,R,t), vs S, against the exact
    (S=inf) reference (dashed, same coarse grid + protocol). Monotone; S=1000 recovers exact-quality.
Right: same vs the matchgate-library size N_Q, against the standard N_Q=500 (which IS the S=inf run).

H4 is grid-limited at the coarse grid this sweep affords (dense spectrum aliased) and is reported
in the text, not plotted. omega_op is taken from the clean pre-flight, never the noisy data.

Reads the B10 per-geometry eval JSONs. Run from models/:
    python -m scripts.plot_finite_shot
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, DOUBLE_COL

REV = os.path.join(os.path.dirname(__file__), "..", "results", "fermionic_pipeline",
                   "regression", "revision")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "paper_figures", "finite_shot.pdf")
LIH = plt.get_cmap("viridis")(0.20)


def median(path):
    d = json.load(open(path))["results"]
    return float(np.median([r["pearson_mean"] for r in d]))


def main():
    apply_nature_style(usetex=False)
    fs_eval = lambda s: os.path.join(REV, "3A_fs", f"lih_S{s}_model", "eval", "regressor_eval.json")
    nq_eval = lambda q: os.path.join(REV, "2D_nq", f"lih_nq{q}_model", "eval", "regressor_eval.json")

    fig, (axS, axN) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.40))

    # --- finite-shot (median) ---
    S = np.array([10, 100, 1000])
    yS = np.array([median(fs_eval(s)) for s in ("10", "100", "1000")])
    y_exact = median(fs_eval("inf"))
    axS.axhline(y_exact, color="0.5", ls="--", lw=0.9)
    axS.text(11, y_exact + 0.02, "exact ($S=\\infty$)", color="0.45", fontsize=5, va="bottom")
    axS.plot(S, yS, color=LIH, lw=1.8, marker="o", ms=4, label="LiH")
    axS.set_xscale("log")
    axS.set_xlabel("Born samples per $(Q,R,t)$, $S$")
    axS.set_ylabel("median held-out Pearson $r$")
    axS.set_ylim(0, 1.05)

    # --- N_Q sweep (median); the N_Q=500 reference is the S=inf run ---
    Q = np.array([50, 100, 250, 500, 1000])
    yQ = np.array([median(nq_eval("50")), median(nq_eval("100")), median(nq_eval("250")),
                   y_exact, median(nq_eval("1000"))])
    axN.axvline(500, color="0.6", ls=":", lw=0.9)
    axN.text(500, 0.04, " reference", color="0.5", fontsize=5, rotation=90, va="bottom")
    axN.plot(Q, yQ, color=LIH, lw=1.8, marker="o", ms=4, label="LiH")
    axN.set_xscale("log")
    axN.set_xlabel("matchgate library size $N_Q$")
    axN.set_ylabel("median held-out Pearson $r$")
    axN.set_ylim(0, 1.05)

    for ax in (axS, axN):
        ax.grid(True, which="major", color="0.9", lw=0.5)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.text(0.5, 0.005, "LiH per-geometry medians; H$_4$ is grid-limited at this coarse grid "
             "(reported in text)", ha="center", fontsize=5, color="0.4")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[done] {OUT}")
    print(f"  finite-shot S (median): S=10 {yS[0]:.2f}  S=100 {yS[1]:.2f}  S=1000 {yS[2]:.2f}  "
          f"S=inf {y_exact:.2f}")
    print(f"  N_Q (median): " + "  ".join(f"{q}={v:.2f}" for q, v in zip(Q, yQ)))


if __name__ == "__main__":
    main()
