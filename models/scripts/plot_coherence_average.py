#!/usr/bin/env python3
"""Cross-molecule AVERAGED coherence heatmap.

Each molecule's windowed-Pearson grid r(R,t) (dumped by extrapolation_heatmap as
coherence_grid.npz) lives on its own physical axes. We put them on a common,
dimensionless frame -- R normalized by the calculated equilibrium R_eq
(CCSD(T)/cc-pVTZ, briefing #5) and t normalized by that molecule's TRAINING
horizon t_train -- interpolate each onto a shared grid, and average across
molecules. The result is a scaling-law-style summary: where (in units of bond
stretch and training-horizon multiples) the amortized model stays coherent.

The horizontal line t/t_train = 1 is the temporal in-box / out-of-box boundary
(Prop 1 interpolation below it, Prop 2 extrapolation above it), common to all
molecules; R/R_eq = 1 marks equilibrium.

Runs on whatever grids are present (>=2 molecules per cell to average); H4's grid
arrives via RunPod briefing #6. Run from models/:
    python -m scripts.plot_coherence_average
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, ONEHALF_COL, quality_cmap

HERE = os.path.dirname(__file__)
REG = os.path.join(HERE, "..", "results", "fermionic_pipeline", "regression")
OUT = os.path.join(HERE, "..", "results", "paper_figures", "coherence_average.pdf")

R_EQ = {"h4": 0.895, "lih": 1.609, "beh2": 1.332, "n2": 1.114}   # CCSD(T)/cc-pVTZ (briefing #5)
MOLS = [("h4", "H$_4$"), ("n2", "N$_2$"), ("beh2", "BeH$_2$"), ("lih", "LiH")]

# common dimensionless frame
XR = np.linspace(0.55, 2.45, 96)    # R / R_eq
YT = np.linspace(0.02, 1.98, 110)   # t / t_train
MIN_MOLS = 2                        # cells with fewer contributing molecules are masked


def grid_path(key):
    return os.path.join(REG, f"{key}_regress_v1_orb_s42_model", "plots_extrap",
                        "coherence_grid.npz")


def resample(key):
    """Return the molecule's corr resampled onto (XR, YT), or None if absent."""
    p = grid_path(key)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    R, t, corr = np.asarray(d["R"]), np.asarray(d["t"]), np.asarray(d["corr"])
    t_train = float(d["train_t_range"][1])
    x, y = R / R_EQ[key], t / t_train          # native normalized axes (sorted asc)
    interp = RegularGridInterpolator((x, y), corr, bounds_error=False, fill_value=np.nan)
    XX, YY = np.meshgrid(XR, YT, indexing="ij")
    return interp(np.stack([XX.ravel(), YY.ravel()], axis=-1)).reshape(XX.shape)


def main():
    apply_nature_style(usetex=False)
    layers, present = [], []
    for key, _ in MOLS:
        g = resample(key)
        if g is not None:
            layers.append(g)
            present.append(key)
    if not layers:
        sys.exit("no coherence grids found")
    stack = np.stack(layers, axis=0)                      # (n_mol, |XR|, |YT|)
    count = np.sum(np.isfinite(stack), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(stack, axis=0)
    mean = np.where(count >= MIN_MOLS, mean, np.nan)

    fig, ax = plt.subplots(figsize=(ONEHALF_COL, ONEHALF_COL * 0.82))
    im = ax.imshow(mean.T, origin="lower", aspect="auto", cmap=quality_cmap(),
                   vmin=-0.2, vmax=1.0,
                   extent=[XR[0], XR[-1], YT[0], YT[-1]], interpolation="nearest")
    ax.axhline(1.0, color="white", lw=2.6, zorder=3)
    ax.axhline(1.0, color="#0b3dff", lw=1.4, ls="--", zorder=4,
               label=r"training horizon $t/t_\mathrm{train}=1$")
    ax.axvline(1.0, color="0.25", lw=1.0, ls=":", zorder=4)
    ax.text(1.0, YT[-1] * 0.985, " equilibrium", color="0.2", fontsize=5,
            va="top", ha="left", rotation=90)
    ax.set_xlabel(r"$R / R_\mathrm{eq}$")
    ax.set_ylabel(r"$t / t_\mathrm{train}$")
    ax.set_title(f"Averaged coherence $r(R,t)$ over {len(present)} molecules")
    ax.legend(loc="upper right", fontsize=5, framealpha=0.85)
    cb = plt.colorbar(im, ax=ax, label=r"mean windowed Pearson $r$")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[done] {OUT}   molecules={present}")
    # quadrant summary: in-box (t<=t_train) vs temporal extrapolation
    inbox = mean[:, YT <= 1.0]
    outbox = mean[:, YT > 1.0]
    print(f"  in-box  (t/t_train<=1)  mean r = {np.nanmean(inbox):.3f}")
    print(f"  out-box (t/t_train >1)  mean r = {np.nanmean(outbox):.3f}")


if __name__ == "__main__":
    main()
