"""
freq_net diagnostic: learned ω_k(R) vs. true Hamiltonian gaps.

Tests whether the freq_net in a conditioned-frequency ObservableRegressor
can reach the high-ω content needed at short R, or whether it saturates
below the true gap spectrum (the "low-pass filter" failure mode).

Usage:
    python3 -m fermionic_pipeline.eval.freq_net_diagnostic \
      --data_path results/fermionic_pipeline/regression/h4_regress_v10/regression_targets.h5 \
      --checkpoint results/fermionic_pipeline/regression/h4_regress_v10_model/regressor.pt \
      --save_dir  results/fermionic_pipeline/regression/h4_regress_v10_model/diagnostics
"""
from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model


def compute_learned_omega(model, orb_energies: np.ndarray, R_values: np.ndarray, device: str):
    """Evaluate ω_k(R) = ω_base + freq_net(input) for every R in the dataset."""
    cfg = model.config
    with torch.no_grad():
        omega_base = model.omega_base.detach().cpu().numpy()
        if model.freq_net is None:
            return np.broadcast_to(omega_base, (len(R_values), len(omega_base))).copy()

        if cfg.n_orb_features > 0:
            x = torch.tensor(orb_energies, dtype=torch.float32, device=device)
        else:
            x = torch.tensor(R_values[:, None], dtype=torch.float32, device=device)
        shift = model.freq_net(x).detach().cpu().numpy()
        return omega_base[None, :] + shift


def gaps_from_ground(eigvals: np.ndarray) -> np.ndarray:
    """Per-R gaps E_j − E_0, shape (n_R, n_levels − 1)."""
    E0 = eigvals[:, :1]
    return eigvals[:, 1:] - E0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_gap_overlay", type=int, default=32,
                   help="How many lowest E_j−E_0 gaps to overlay on the ω_k(R) panel.")
    p.add_argument("--short_r", type=float, default=1.0)
    p.add_argument("--long_r", type=float, default=1.75)
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    with h5py.File(args.data_path, "r") as f:
        R_values = f["R_values"][...]
        eigvals = f["eigvals"][...]
        orb = f["hf_orbital_energies"][...] if "hf_orbital_energies" in f else None

    model, _ = load_checkpoint_model(args.checkpoint, device=args.device)
    omega = compute_learned_omega(model, orb, R_values, args.device)
    abs_omega = np.abs(omega)

    gaps = gaps_from_ground(eigvals)
    abs_gaps = np.abs(gaps)

    # ─── Panel A: ω_k(R) cloud + true gap envelope ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axes[0]
    for k in range(abs_omega.shape[1]):
        ax.plot(R_values, abs_omega[:, k], color="tab:blue", alpha=0.05, lw=0.6)
    max_omega = abs_omega.max(axis=1)
    ax.plot(R_values, max_omega, color="tab:blue", lw=2.0,
            label=r"max$_k\,|\omega_k(R)|$")

    n_overlay = min(args.n_gap_overlay, abs_gaps.shape[1])
    for j in range(n_overlay):
        ax.plot(R_values, abs_gaps[:, j], color="tab:red", alpha=0.35, lw=0.8)
    ax.plot(R_values, abs_gaps[:, :n_overlay].max(axis=1), color="tab:red", lw=2.0,
            label=f"max of lowest {n_overlay} true gaps $E_j-E_0$")

    for R_mark in (args.short_r, args.long_r):
        ax.axvline(R_mark, ls=":", color="gray", alpha=0.6)
    ax.set_xlabel("R (Å)")
    ax.set_ylabel(r"$|\omega|$  ($E_h$)")
    ax.set_title(r"Learned $\omega_k(R)$ vs. true eigenvalue gaps")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    # ─── Panel B: per-regime histograms ────────────────────────────────
    ax = axes[1]
    short_mask = R_values < args.short_r
    mid_mask = (R_values >= args.short_r) & (R_values < args.long_r)
    long_mask = R_values >= args.long_r
    bins = np.linspace(0.0, max(abs_omega.max(), abs_gaps.max()) * 1.02, 60)

    for mask, label, color in [
        (short_mask, f"R < {args.short_r}", "tab:blue"),
        (mid_mask, f"{args.short_r} ≤ R < {args.long_r}", "tab:green"),
        (long_mask, f"R ≥ {args.long_r}", "tab:orange"),
    ]:
        if not mask.any():
            continue
        ax.hist(abs_omega[mask].ravel(), bins=bins, density=True, histtype="step",
                lw=2.0, color=color, label=f"learned ω, {label}")
        ax.hist(abs_gaps[mask].ravel(), bins=bins, density=True, histtype="step",
                lw=1.2, ls="--", color=color, alpha=0.8,
                label=f"true gaps, {label}")
    ax.set_xlabel(r"$|\omega|$  ($E_h$)")
    ax.set_ylabel("density")
    ax.set_title("Frequency support by R-regime")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # ─── Panel C: ceiling ratio ────────────────────────────────────────
    ax = axes[2]
    true_max = abs_gaps.max(axis=1)
    ax.plot(R_values, max_omega, color="tab:blue", lw=2.0, label=r"max$_k\,|\omega_k|$ (learned)")
    ax.plot(R_values, true_max, color="tab:red", lw=2.0, label=r"max true gap")
    ax.plot(R_values, max_omega / true_max, color="black", lw=1.5, ls="--",
            label="ratio (learned / true)")
    ax.axhline(1.0, color="gray", alpha=0.5, lw=0.8)
    for R_mark in (args.short_r, args.long_r):
        ax.axvline(R_mark, ls=":", color="gray", alpha=0.6)
    ax.set_xlabel("R (Å)")
    ax.set_ylabel(r"$|\omega|$ / ratio")
    ax.set_title("Ceiling comparison")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(args.save_dir, "freq_net_diagnostic.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)

    # Brief stdout summary — per-regime ceiling gap.
    for name, mask in [("short", short_mask), ("mid", mid_mask), ("long", long_mask)]:
        if not mask.any():
            continue
        lm = max_omega[mask].mean()
        tm = true_max[mask].mean()
        print(f"[{name:>5}] <max learned ω>={lm:.3f}  <max true gap>={tm:.3f}  "
              f"ratio={lm/tm:.3f}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
