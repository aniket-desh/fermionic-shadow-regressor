#!/usr/bin/env python3
"""Cross-molecule overlay of the 1x3 per-geometry summary on an R/R_eq axis.

For EXPERIMENTAL and COMPUTED R_eq, emits three variants of the 1x3 (held-out
temporal Pearson, MSE, dynamic-range ratio), one curve per molecule {H4,LiH,BeH2,N2},
sequential colour by spectral speed (light=fast), LiH emphasized, legend in a row below:
    linear        - the two bounded metrics on linear y
    loglog        - log-log; bounded metrics shown as loss-like distance-from-ideal
                    (1 - r ; |range ratio - 1|), MSE as-is
    loglog_smooth - loglog with a rolling-median line over faint raw points

Run from models/:  python -m scripts.plot_overlay_summary
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, DOUBLE_COL

HERE = os.path.dirname(__file__)
JDIR = os.path.join(HERE, "..", "results", "paper_figures", "overlay_jsons")
ODIR = os.path.join(HERE, "..", "results", "paper_figures")

R_EQ_SETS = {
    "calculated":   {"h4": 0.895, "lih": 1.609, "beh2": 1.332, "n2": 1.114},
    "experimental": {"h4": 0.90,  "lih": 1.595, "beh2": 1.334, "n2": 1.098},
}
R_EQ_NOTE = {
    "calculated":   r"calculated $R_\mathrm{eq}$ (CCSD(T)/cc-pVTZ)",
    "experimental": r"experimental $R_\mathrm{eq}$ (NIST CCCBDB; H$_4$ symmetric-chain conv.)",
}
MOLS = [("h4", "H$_4$", False), ("n2", "N$_2$", False),
        ("beh2", "BeH$_2$", False), ("lih", "LiH", True)]
CMAP = "viridis"     # CVD-safe sequential; light=fast, dark=slow
SMOOTH_WINDOW = 5

# (field, title, transform-key, logy, extra) ; extra = ("ylim",lo,hi) | ("ref",y,(lo,hi))
PANELS_LINEAR = [
    ("pearson", "Held-out temporal Pearson $r$", None, False, ("ylim", 0.0, 1.02)),
    ("mse",     "Prediction MSE",                None, True,  None),
    ("rr",      "Dynamic-range ratio",           None, False, ("ref", 1.0, (0.8, 1.3))),
]
PANELS_LOGLOG = [
    ("pearson", r"$1-r$ (decorrelation)",       "1mr",    True, None),
    ("mse",     "Prediction MSE",               None,     True, None),
    ("rr",      r"$|\mathrm{range\ ratio}-1|$", "absdev", True, None),
]


def load(key):
    res = sorted(json.load(open(os.path.join(JDIR, f"{key}.json")))["results"],
                 key=lambda d: d["R"])
    R = np.array([d["R"] for d in res])
    return R, {"pearson": np.array([d["pearson_mean"] for d in res]),
               "mse": np.array([d["mse"] for d in res]),
               "rr": np.array([d["range_ratio_mean"] for d in res])}


def transform(vals, kind):
    if kind == "1mr":
        return np.clip(1.0 - vals, 1e-4, None)
    if kind == "absdev":
        return np.clip(np.abs(vals - 1.0), 1e-4, None)
    return vals


def rolling_median(y, w):
    from scipy.ndimage import median_filter
    return median_filter(np.asarray(y, float), size=w, mode="nearest")


def make(mode, loglog, smooth, raw):
    R_eq = R_EQ_SETS[mode]
    base = plt.get_cmap(CMAP)
    cpos = np.linspace(0.75, 0.20, len(MOLS))   # light(fast) -> mid-dark(slow); avoid near-black
    color = {m[0]: base(p) for m, p in zip(MOLS, cpos)}
    panels = PANELS_LOGLOG if loglog else PANELS_LINEAR

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.42))
    for ax, (field, title, tf, logy, extra) in zip(axes, panels):
        ax.set_axisbelow(True)
        ax.grid(True, which="major", color="0.85", lw=0.5, zorder=0)
        if extra and extra[0] == "ref":
            ax.axhline(extra[1], color="0.45", lw=0.8, ls=":", zorder=1)
        for key, label, emph in MOLS:
            R, d = raw[key]
            x = R / R_eq[key]
            y = transform(d[field], tf)
            lab = f"{label} ({R_eq[key]:.2f})"
            if smooth:
                ax.plot(x, y, color=color[key], lw=0, marker="o", ms=1.5,
                        alpha=0.28, zorder=3)
                ax.plot(x, rolling_median(y, SMOOTH_WINDOW), color=color[key],
                        label=lab, lw=2.1 if emph else 1.5, alpha=0.95,
                        zorder=6 if emph else 4)
            else:
                ax.plot(x, y, color=color[key], label=lab,
                        lw=1.8 if emph else 1.4, marker="o",
                        ms=2.5 if emph else 2.0, alpha=0.8, zorder=6 if emph else 4)
        if logy:
            ax.set_yscale("log")
            ax.grid(True, which="minor", color="0.93", lw=0.3, zorder=0)
        if loglog:
            ax.set_xscale("log")
        ax.set_xlabel(r"$R / R_\mathrm{eq}$")
        ax.set_title(title)
        ax.margins(x=0.02)
        if extra and extra[0] == "ylim":
            ax.set_ylim(extra[1], extra[2])
        elif extra and extra[0] == "ref":
            ax.set_ylim(*extra[2])

    tag = ("log-log" + (", smoothed" if smooth else "")) if loglog else "linear"
    fig.suptitle(f"Normalized by {R_EQ_NOTE[mode]}  ({tag})", y=0.99)
    fig.tight_layout(rect=[0, 0.11, 1, 0.93])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=len(MOLS), frameon=False, handlelength=1.6, columnspacing=2.2)

    os.makedirs(ODIR, exist_ok=True)
    suffix = ("_loglog" if loglog else "") + ("_smooth" if smooth else "")
    out = os.path.join(ODIR, f"overlay_{mode}{suffix}.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"[done] {out}")


def main():
    apply_nature_style(usetex=False)
    raw = {k: load(k) for k, _, _ in MOLS}
    for mode in R_EQ_SETS:
        make(mode, loglog=False, smooth=False, raw=raw)   # linear
        make(mode, loglog=True,  smooth=False, raw=raw)   # log-log (raw)
        make(mode, loglog=True,  smooth=True,  raw=raw)   # log-log (smoothed)


if __name__ == "__main__":
    main()
