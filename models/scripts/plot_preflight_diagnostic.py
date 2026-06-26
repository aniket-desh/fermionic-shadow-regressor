#!/usr/bin/env python3
"""Pre-flight diagnostic figure (peer-review W3): the line-spectrum quantities the
sampling grid is designed from, shown explicitly.

Left:  omega_op(R), the 99%-power bandwidth edge, which sets the Nyquist step
       Delta t <= pi / omega_op. Light (fast) to dark (slow) over the four molecules.
Right: delta-omega(R), the spacing of the co-dominant Bohr lines, against each
       molecule's horizon-resolution floor 2*pi / t_max (dashed). Where the spacing
       falls to the floor the lines are unresolvable within the training horizon
       (congestion); this is where the surrogate is expected to fail, and for N2 it
       happens at the avoided crossing near R ~ 2.1-2.35 A.

Cheap exact 8-qubit diagonalization, no checkpoints. Run from models/:
    python -m scripts.plot_preflight_diagnostic
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.eval.nature_style import apply_nature_style, DOUBLE_COL
from fermionic_pipeline.data.line_spectrum_preflight import (
    build_molecule, build_majorana_matrices, line_spectrum, aggregate_lines,
    dominant_stats,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "paper_figures",
                   "preflight_diagnostic.pdf")

# key, label, scan range (extended to expose congestion), training box, t_max
MOLS = [
    ("h4",   "H$_4$",   (0.40, 3.30), (0.5, 3.0), 300),
    ("n2",   "N$_2$",   (0.70, 2.60), (0.9, 2.3), 500),
    ("beh2", "BeH$_2$", (0.80, 3.40), (1.0, 3.0), 500),
    ("lih",  "LiH",    (0.80, 3.60), (1.0, 3.2), 1200),
]
STEP = 0.05


def scan(mol, lo, hi, gammas):
    Rs = np.arange(lo, hi + 1e-9, STEP)
    bw, dw = np.zeros(len(Rs)), np.zeros(len(Rs))
    for i, R in enumerate(Rs):
        H, _nq, ne = build_molecule(mol, float(R))
        ls = line_spectrum(H, ne, gammas)
        fr, w = aggregate_lines(ls["freqs"], ls["weights"], 4)
        st = dominant_stats(fr, w)
        bw[i] = st["bw99"]
        dw[i] = st["dw_strong"]          # inf when <2 co-dominant lines (uncongested)
    return Rs, bw, dw


def main():
    apply_nature_style(usetex=False)
    gammas = build_majorana_matrices(8)   # all molecules share the 8-qubit interface
    base = plt.get_cmap("viridis")
    cpos = np.linspace(0.75, 0.20, len(MOLS))
    color = {m[0]: base(c) for m, c in zip(MOLS, cpos)}

    data = {key: scan(key, lo, hi, gammas) for key, _, (lo, hi), _, _ in MOLS}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.40))
    for key, lab, _, _box, tmax in MOLS:
        col = color[key]
        Rs, bw, dw = data[key]
        axA.plot(Rs, bw, color=col, lw=1.6, label=lab)
        floor = 2 * np.pi / tmax
        dw_plot = np.where(np.isfinite(dw), dw, np.nan)
        axB.plot(Rs, dw_plot, color=col, lw=1.6, marker="o", ms=1.8, label=lab)
        axB.axhline(floor, color=col, ls=":", lw=0.9, alpha=0.8)
        cong = np.isfinite(dw) & (dw < floor)
        if cong.any():
            axB.plot(Rs[cong], dw[cong], color=col, lw=0, marker="v", ms=4.5, zorder=5)

    for ax in (axA, axB):
        ax.set_axisbelow(True)
        ax.grid(True, color="0.9", lw=0.5)
        ax.set_xlabel(r"$R$ (\AA)")
    axA.set_ylabel(r"$\omega_\mathrm{op}(R)$ ($E_\mathrm{h}$)")
    axA.set_title(r"Bandwidth: sets $\Delta t \leq \pi/\omega_\mathrm{op}$")
    axA.legend(frameon=False, ncol=2, fontsize=6, handlelength=1.4)
    axB.set_yscale("log")
    axB.set_ylabel(r"co-dominant spacing $\delta\omega(R)$ ($E_\mathrm{h}$)")
    axB.set_title(r"Resolution: $\delta\omega < 2\pi/t_\mathrm{max}$ (dashed) congests")
    axB.text(0.02, 0.04, "down-markers: congested (predicted failure)",
             transform=axB.transAxes, fontsize=5.5, color="0.3")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"[done] {OUT}")
    for key, _lab, _, _box, tmax in MOLS:
        Rs, bw, dw = data[key]
        floor = 2 * np.pi / tmax
        cong = Rs[np.isfinite(dw) & (dw < floor)]
        tag = f"{cong.min():.2f}-{cong.max():.2f}" if cong.size else "none"
        print(f"  {key:5s} omega_op {bw.max():.3f}->{bw.min():.3f}  floor {floor:.4f}  congested R: {tag}")


if __name__ == "__main__":
    main()
