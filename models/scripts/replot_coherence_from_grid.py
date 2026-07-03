#!/usr/bin/env python3
"""Re-draw a coherence heatmap r(R,t) from a cached ``coherence_grid.npz`` -- NO
model, NO GPU. This lets you iterate on centering/fonts for the coherence-map
figures (paper Figs 4-5 + appendix) without re-running the network.

The authoritative producer is ``fermionic_pipeline.eval.plot_regression`` /
``extrapolation_heatmap`` (which run the model forward, then cache the grid as
``coherence_grid.npz``). This script consumes that cache and reproduces the exact
same drawing. FONTS are governed by ``nature_style`` (shared with every other
figure), so global font edits there propagate here automatically; only the
per-panel centering below (figsize / colorbar / tight_layout) is local to this file.

Run from models/. Examples:
    python -m scripts.replot_coherence_from_grid \
        --grid results/fermionic_pipeline/regression/h4_regress_v1_orb_s42_model/plots_extrap/coherence_grid.npz \
        --out  results/paper_figures/coherence_heatmap.pdf
    # LiH main-text figure:
    python -m scripts.replot_coherence_from_grid \
        --grid results/fermionic_pipeline/regression/lih_regress_v1_orb_s42_model/plots_extrap/coherence_grid.npz \
        --out  results/paper_figures/lih_coherence_heatmap.pdf
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, DOUBLE_COL, quality_cmap


def draw(grid_path, out_path, colorblind=False):
    d = np.load(grid_path, allow_pickle=True)
    corr = d["corr"]
    R = np.asarray(d["R"], dtype=float)
    t = np.asarray(d["t"], dtype=float)
    R_lo, R_hi = (float(x) for x in d["train_R_range"])
    t_lo, t_hi = (float(x) for x in d["train_t_range"])

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
    im = ax.imshow(corr, aspect="auto", cmap=quality_cmap(colorblind), vmin=-0.2, vmax=1.0,
                   extent=[t[0], t[-1], R[-1], R[0]], interpolation="nearest")
    ax.set_xlabel(r"$t$ (a.u.)")
    ax.set_ylabel("$R$ (Å)")
    ax.set_title(r"Windowed Pearson $r(R, t)$ — model vs. exact observables")
    plt.colorbar(im, ax=ax, label=r"Pearson $r$")

    # training-region box: white halo under a bold blue dashed line
    ax.add_patch(Rectangle((t_lo, R_lo), t_hi - t_lo, R_hi - R_lo,
                           fill=False, edgecolor="white", linewidth=3.6, zorder=4))
    ax.add_patch(Rectangle((t_lo, R_lo), t_hi - t_lo, R_hi - R_lo,
                           fill=False, edgecolor="#0b3dff", linestyle="--", linewidth=2.0,
                           zorder=5, label="training region"))

    in_R = (R >= R_lo) & (R <= R_hi)
    in_t = (t >= t_lo) & (t <= t_hi)
    if in_R.any() and in_t.any() and not (in_R.all() and in_t.all()):
        inbox = corr[np.ix_(in_R, in_t)]
        omask = np.ones_like(corr, dtype=bool)
        omask[np.ix_(in_R, in_t)] = False
        ax.text(0.015, 0.03,
                f"in-box  $\\bar{{r}}$ ={np.nanmean(inbox):.2f}     "
                f"out-of-box  $\\bar{{r}}$ ={np.nanmean(corr[omask]):.2f}",
                transform=ax.transAxes, color="#0b3dff",
                bbox=dict(boxstyle="round", fc="white", ec="#0b3dff", alpha=0.85))
    ax.set_xlim(min(t[0], t_lo), max(t[-1], t_hi))
    ax.set_ylim(max(R[-1], R_hi), min(R[0], R_lo))
    ax.legend(loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[done] {out_path}  (in-box r̄={np.nanmean(corr[np.ix_(in_R, in_t)]):.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", required=True, help="path to a coherence_grid.npz")
    ap.add_argument("--out", required=True, help="output pdf path")
    ap.add_argument("--colorblind", action="store_true")
    args = ap.parse_args()
    apply_nature_style(usetex=False, colorblind=args.colorblind)
    draw(args.grid, args.out, args.colorblind)


if __name__ == "__main__":
    main()
