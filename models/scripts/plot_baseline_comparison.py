#!/usr/bin/env python3
"""Held-out accuracy: FSR vs fair classical surrogates (peer-review E2).

Grouped bars of held-out temporal Pearson, one group per molecule, for the FSR and three
matched baselines: a fixed-frequency-bank linear-harmonic regressor (1C), a parameter-matched
Fourier-feature MLP with the same HF features (2A), and kernel-ridge regression over (R,t) (2B).
All at seed 42, same held-out split, same metric. The FSR's explicit, geometry-conditioned
harmonic head wins decisively on the hard molecules; the honest exception (a fixed bank matches
BeH2, the smoothest system) is visible. Per-geometry classical FFT peak-recovery (2C) is an
oracle-given-data ceiling on a different metric and is reported in the text, not here.

Reads the revision baseline JSONs. Run from models/:
    python -m scripts.plot_baseline_comparison
"""
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, ONEHALF_COL

RR = os.path.join(os.path.dirname(__file__), "..", "results", "fermionic_pipeline",
                  "regression", "revision")
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "paper_figures",
                   "baseline_comparison.pdf")

MOLS = [("h4", "H$_4$"), ("n2", "N$_2$"), ("beh2", "BeH$_2$"), ("lih", "LiH")]
# (label, json template, viridis position); FSR first, darkest -> baselines lighter
METHODS = [
    ("FSR",              None,                                   0.15),
    ("linear-harmonic",  "1C_linharm/baseline_linharm_{}.json",  0.45),
    ("Fourier MLP",      "2A_fourmlp/baseline_fourmlp_{}.json",   0.65),
    ("GP / KRR",         "2B_gpkrr/baseline_gpkrr_{}.json",       0.82),
]


def heldout(path):
    d = json.load(open(path))
    for k in ("heldout_pearson_mean", "pearson_mean", "heldout_pearson"):
        if k in d:
            return float(d[k])
    return np.nan


def main():
    apply_nature_style(usetex=False)
    vals = {}
    for lab, tmpl, _ in METHODS:
        if lab == "FSR":
            vals[lab] = [json.load(open(os.path.join(RR, "1A", "stats", f"stats_{m}.json")))
                         ["per_seed"]["42"]["pearson_mean"] for m, _ in MOLS]
        else:
            vals[lab] = [heldout(os.path.join(RR, tmpl.format(m))) for m, _ in MOLS]

    n_m = len(MOLS)
    nb = len(METHODS)
    x = np.arange(n_m)
    w = 0.8 / nb

    base = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(ONEHALF_COL, ONEHALF_COL * 0.62))
    for i, (lab, _, pos) in enumerate(METHODS):
        off = (i - (nb - 1) / 2) * w
        bars = ax.bar(x + off, vals[lab], w, label=lab, color=base(pos),
                      edgecolor="0.2" if lab == "FSR" else "none", linewidth=0.5, zorder=3)
        if lab == "FSR":
            for b, v in zip(bars, vals[lab]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=5, color="0.1")

    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="0.9", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in MOLS])
    ax.set_ylabel("held-out temporal Pearson $r$")
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=2, frameon=False, fontsize=6, loc="lower center",
              bbox_to_anchor=(0.5, -0.32), columnspacing=1.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {OUT}")
    for lab, _, _ in METHODS:
        print(f"  {lab:16s} " + "  ".join(f"{m}={v:.3f}" for (m, _), v in zip(MOLS, vals[lab])))


if __name__ == "__main__":
    main()
