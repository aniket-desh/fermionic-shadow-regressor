#!/usr/bin/env python3
"""Cross-molecule "failure anatomy" of held-out accuracy vs R/R_eq (peer-review E5
redesign).

Main figure (two panels): decorrelation 1-r and dynamic-range error |range ratio - 1|,
one curve per molecule, raw per-geometry points with a light rolling-median line, log-log,
molecules labelled at the curve ends, and the pre-flight-congested N2 region shaded. The
prediction MSE is emitted as a separate appendix figure (overlay_mse_appendix.pdf).

R_eq = CCSD(T)/cc-pVTZ (briefing #5). Congested ranges from the line-spectrum pre-flight
(scripts/plot_preflight_diagnostic.py). Run from models/:
    python -m scripts.plot_overlay_summary
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, DOUBLE_COL

HERE = os.path.dirname(__file__)
JDIR = os.path.join(HERE, "..", "results", "paper_figures", "overlay_jsons")
ODIR = os.path.join(HERE, "..", "results", "paper_figures")

R_EQ = {"h4": 0.895, "lih": 1.609, "beh2": 1.332, "n2": 1.114}   # CCSD(T)/cc-pVTZ
MOLS = [("h4", "H$_4$"), ("n2", "N$_2$"), ("beh2", "BeH$_2$"), ("lih", "LiH")]
CONGEST = {"n2": (2.10, 2.40)}      # pre-flight-congested bond range (Angstrom)
SMOOTH_W = 5


def load(key):
    res = sorted(json.load(open(os.path.join(JDIR, f"{key}.json")))["results"],
                 key=lambda d: d["R"])
    R = np.array([d["R"] for d in res])
    return R / R_EQ[key], {
        "1mr": np.clip(1.0 - np.array([d["pearson_mean"] for d in res]), 1e-4, None),
        "absdev": np.clip(np.abs(np.array([d["range_ratio_mean"] for d in res]) - 1.0), 1e-4, None),
        "mse": np.array([d["mse"] for d in res]),
    }


def rolling_median(y, w):
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(y, float), size=w, mode="nearest")


def decade_ticks(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 3]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.5", "1", "2", "3"]))
    ax.xaxis.set_minor_locator(NullLocator())      # no leaking 6x10^-1 minor labels


def draw(ax, raw, color, field, label_end):
    for key, lab in MOLS:
        x, d = raw[key]
        y = d[field]
        ax.plot(x, y, lw=0, marker="o", ms=1.6, color=color[key], alpha=0.30, zorder=2)
        ys = rolling_median(y, SMOOTH_W)
        emph = key == "lih"
        ax.plot(x, ys, color=color[key], lw=2.0 if emph else 1.4, alpha=0.95,
                zorder=6 if emph else 4)
        if label_end:                       # molecule name at the right curve end
            ax.annotate(lab, (x[-1], ys[-1]), xytext=(3, 0), textcoords="offset points",
                        color=color[key], fontsize=6, va="center")
    for key, (lo, hi) in CONGEST.items():
        ax.axvspan(lo / R_EQ[key], hi / R_EQ[key], color="#d62728", alpha=0.10, zorder=0)
    ax.axvline(1.0, color="0.6", ls=":", lw=0.8, zorder=1)
    decade_ticks(ax)
    ax.set_xlim(0.5, 3.0)
    ax.set_xlabel(r"$R/R_\mathrm{eq}$")
    ax.grid(True, which="major", color="0.88", lw=0.5)
    ax.grid(True, which="minor", color="0.95", lw=0.3)


def main():
    apply_nature_style(usetex=False)
    base = plt.get_cmap("viridis")
    color = {m[0]: base(c) for m, c in zip(MOLS, np.linspace(0.75, 0.20, len(MOLS)))}
    raw = {k: load(k) for k, _ in MOLS}

    # main: decorrelation + range-ratio error
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))
    draw(a1, raw, color, "1mr", label_end=True)
    a1.set_ylabel(r"decorrelation $1-r$")
    a1.set_title("Phase fidelity")
    draw(a2, raw, color, "absdev", label_end=True)
    a2.set_ylabel(r"$|\mathrm{range\ ratio}-1|$")
    a2.set_title("Dynamic-range fidelity")
    fig.text(0.5, 0.005, "shaded: pre-flight-congested N$_2$ region (predicted failure)",
             ha="center", fontsize=5.5, color="0.4")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(os.path.join(ODIR, "cross_molecule_overlay.pdf"))
    plt.close(fig)
    print("[done] cross_molecule_overlay.pdf")

    # appendix: MSE alone
    figm, axm = plt.subplots(figsize=(DOUBLE_COL * 0.5, DOUBLE_COL * 0.42))
    draw(axm, raw, color, "mse", label_end=True)
    axm.set_ylabel("prediction MSE")
    axm.set_title("Prediction MSE versus normalized bond length")
    figm.tight_layout()
    figm.savefig(os.path.join(ODIR, "overlay_mse_appendix.pdf"))
    plt.close(figm)
    print("[done] overlay_mse_appendix.pdf")


if __name__ == "__main__":
    main()
