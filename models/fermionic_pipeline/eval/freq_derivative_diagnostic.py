"""
Per-R derivative test: is freq_net's ω_k(R) jitter the cause of the
bad-ratio cluster at R ≈ 0.65–0.78, or does it just track fast-moving
true frequencies?

Plots three curves vs R:
  1. ‖∂ω_k / ∂R‖_2 (model-side, numerical diff of v10 ω_k(R)).
  2. ‖∂ω_peaks / ∂R‖_2 (target-side, from FFT peaks of y_true(R, t)).
  3. The per-R stall ratio (overlay, secondary y-axis).

Peak tracking across R uses bipartite matching by proximity (Hungarian-
free greedy nearest-neighbor), restricted to peaks above a fraction of
the per-R max amplitude. Pad/truncate to match peak counts so derivatives
don't blow up on identity-swaps.
"""
from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks

from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model
from fermionic_pipeline.eval.fft_amplitude_diagnostic import predict_grid, spectrum
from fermionic_pipeline.eval.freq_alignment_diagnostic import compute_learned_omega


def track_peaks(P, freqs, n_track=12, prominence_frac=0.05):
    """Track top-n true-spectrum peaks across R by greedy proximity match.
    Returns (n_R, n_track) array of peak ω positions (NaN where missing)."""
    n_R = P.shape[0]
    tracked = np.full((n_R, n_track), np.nan)
    # seed from first R
    p0 = P[0] / P[0].max()
    pks0, _ = find_peaks(p0, prominence=prominence_frac)
    pks0 = pks0[np.argsort(p0[pks0])[::-1][:n_track]]
    tracked[0, :len(pks0)] = freqs[pks0]
    for i in range(1, n_R):
        p = P[i] / max(P[i].max(), 1e-30)
        pks, _ = find_peaks(p, prominence=prominence_frac)
        cand = freqs[pks]
        prev = tracked[i - 1]
        for k in range(n_track):
            if np.isnan(prev[k]) or len(cand) == 0:
                continue
            j = int(np.argmin(np.abs(cand - prev[k])))
            tracked[i, k] = cand[j]
            cand = np.delete(cand, j)
    return tracked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_track", type=int, default=16)
    ap.add_argument("--hi_lo", type=float, default=4.0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    with h5py.File(args.data_path, "r") as f:
        R = f["R_values"][...]
        t = f["times"][...]
        y_true = f["expectations"][...]
        orb = f["hf_orbital_energies"][...]

    model, _ = load_checkpoint_model(args.checkpoint, device=args.device)

    omega = compute_learned_omega(model, orb, R, args.device)        # (n_R, K)
    abs_omega = np.abs(omega)
    domega_dR = np.gradient(abs_omega, R, axis=0)                     # (n_R, K)
    model_speed = np.linalg.norm(domega_dR, axis=1)                   # ||∂ω/∂R||₂

    freqs, S_true = spectrum(y_true, t)
    tracked = track_peaks(S_true, freqs, n_track=args.n_track)        # (n_R, n_track)
    dpk_dR = np.gradient(tracked, R, axis=0)
    valid = ~np.isnan(dpk_dR)
    target_speed = np.zeros(len(R))
    for i in range(len(R)):
        v = dpk_dR[i][valid[i]]
        target_speed[i] = np.linalg.norm(v) if len(v) > 0 else np.nan

    # Per-R stall ratio (recompute from cached S_true and y_pred).
    y_pred = predict_grid(model, R, t, orb, args.device, batch_R=8)
    _, S_pred = spectrum(y_pred, t)
    ratio = np.full(len(R), np.nan)
    omega_op = np.zeros(len(R))
    for i in range(len(R)):
        cum = np.cumsum(S_true[i]) / S_true[i].sum()
        op = freqs[min(np.searchsorted(cum, 0.99), len(freqs) - 1)]
        omega_op[i] = op
        if op < args.hi_lo + 0.5:
            continue
        hi = (freqs >= args.hi_lo) & (freqs <= op)
        full = freqs <= op
        mt = S_true[i, hi].sum() / max(S_true[i, full].sum(), 1e-30)
        mp = S_pred[i, hi].sum() / max(S_pred[i, full].sum(), 1e-30)
        if mt > 1e-6:
            ratio[i] = mp / mt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.plot(R, model_speed, color="tab:blue", lw=1.6,
            label=r"$\|\partial \omega_k/\partial R\|_2$ (learned, all $k$)")
    ax.plot(R, target_speed, color="tab:red", lw=1.6,
            label=fr"$\|\partial \omega_{{peak}}/\partial R\|_2$ (top-{args.n_track} true peaks)")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\|\partial\omega/\partial R\|_2$  ($E_h/$Å)")
    ax.set_title("Model-side vs target-side frequency speed")
    for R_mark in (0.69, 0.71, 0.78):
        ax.axvline(R_mark, ls=":", color="black", alpha=0.4)
    ax.grid(alpha=0.3); ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(R, ratio, color="black", lw=1.6, label="stall ratio (pred/true hi-band)")
    ax.axhline(1.0, ls="--", color="gray", alpha=0.6)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel("stall ratio"); ax.set_xlabel("R (Å)")
    for R_mark in (0.69, 0.71, 0.78):
        ax.axvline(R_mark, ls=":", color="black", alpha=0.4)
    ax.grid(alpha=0.3); ax.legend(fontsize=10)

    fig.tight_layout()
    out = os.path.join(args.save_dir, "freq_derivative_diagnostic.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)

    # Numerical summary at the bad cluster.
    print(f"{'R':>6s}  {'model_speed':>11s}  {'target_speed':>12s}  {'ratio':>8s}  {'ω_op':>6s}")
    for R_target in (0.65, 0.67, 0.69, 0.70, 0.71, 0.72, 0.78, 0.85, 1.00):
        i = int(np.argmin(np.abs(R - R_target)))
        print(f"{R[i]:>6.2f}  {model_speed[i]:>11.3f}  {target_speed[i]:>12.3f}  "
              f"{ratio[i]:>8.3f}  {omega_op[i]:>6.2f}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
