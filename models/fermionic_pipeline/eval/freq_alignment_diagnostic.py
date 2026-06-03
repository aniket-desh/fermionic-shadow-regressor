"""
freq_net alignment diagnostic: do learned ω_k(R) sit where the signal lives?

Companion to freq_net_diagnostic.py (which checks range/ceiling). This script
measures positional alignment between learned ω_k and the actual spectral
support of y_true(R, t), not against the raw eigenvalue gap set.

Per-R metrics:
  - operational ceiling ω_op(R): smallest ω capturing 99% of mean-subtracted,
    Hann-windowed |Y_true|^2.
  - D(R) = ∫ |Y_true(R, ω)|^2 · min_k |ω - ω_k(R)|^2 dω, over ω ∈ [0, ω_op(R)].
    Amplitude-weighted miss-distance: small ⇔ learned basis sits where the
    signal sits.
  - W1(R): Wasserstein-1 between {|ω_k(R)|} (uniformly weighted) and the
    discretized |Y_true(R, ·)|^2 distribution on [0, ω_op(R)].

Plots:
  A. ω_op(R) and D(R) vs R, with regime shading.
  B. W1(R) vs R.
  C. Three spike-plot panels (short/mid/long R): |Y_true(ω)| backdrop with
     learned ω_k positions overlaid as vertical sticks.
"""
from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model


def compute_learned_omega(model, orb, R_values, device):
    cfg = model.config
    with torch.no_grad():
        omega_base = model.omega_base.detach().cpu().numpy()
        if model.freq_net is None:
            return np.broadcast_to(omega_base, (len(R_values), len(omega_base))).copy()
        if cfg.n_orb_features > 0:
            x = torch.tensor(orb, dtype=torch.float32, device=device)
        else:
            x = torch.tensor(R_values[:, None], dtype=torch.float32, device=device)
        shift = model.freq_net(x).detach().cpu().numpy()
        return omega_base[None, :] + shift


def true_spectrum(y, t):
    """Return (freqs, |Y|^2) with per-(R,μ) DC removed and Hann window."""
    n = len(t)
    dt = t[1] - t[0]
    freqs = 2 * np.pi * np.fft.rfftfreq(n, d=dt)  # angular ω in E_h
    y0 = y - y.mean(axis=1, keepdims=True)
    hann = np.hanning(n)[None, :, None]
    Y = np.fft.rfft(y0 * hann, axis=1)
    P = (np.abs(Y) ** 2).sum(axis=2)  # (n_R, n_freq) — sum over observables
    return freqs, P


def operational_ceiling(P_R, freqs, frac=0.99):
    cum = np.cumsum(P_R) / P_R.sum()
    idx = np.searchsorted(cum, frac)
    return freqs[min(idx, len(freqs) - 1)]


def miss_distance_D(omega_k, freqs, P_R, omega_op):
    """∫_{0}^{ω_op} |Y|^2 · min_k |ω - ω_k|^2 dω, normalized by ∫|Y|^2 dω."""
    mask = freqs <= omega_op
    f = freqs[mask]
    p = P_R[mask]
    if p.sum() < 1e-30:
        return np.nan
    # min_k |ω - ω_k| for each ω in f
    abs_omega_k = np.sort(np.abs(omega_k))
    idx = np.searchsorted(abs_omega_k, f)
    idx = np.clip(idx, 1, len(abs_omega_k) - 1)
    left = abs_omega_k[idx - 1]
    right = abs_omega_k[idx]
    d = np.minimum(np.abs(f - left), np.abs(f - right))
    # Edge: ω below smallest ω_k
    edge = f < abs_omega_k[0]
    d[edge] = abs_omega_k[0] - f[edge]
    return float((p * d ** 2).sum() / p.sum())


def w1_distance(omega_k, freqs, P_R, omega_op):
    """W1 between empirical {|ω_k|} (within [0, ω_op]) and |Y|^2 distribution."""
    mask = freqs <= omega_op
    f = freqs[mask]
    p = P_R[mask]
    if p.sum() < 1e-30:
        return np.nan
    p = p / p.sum()
    abs_omega_k = np.abs(omega_k)
    abs_omega_k = abs_omega_k[abs_omega_k <= omega_op]
    if len(abs_omega_k) == 0:
        return np.nan
    # CDFs on shared grid
    sort_k = np.sort(abs_omega_k)
    cdf_k = np.searchsorted(sort_k, f, side="right") / len(sort_k)
    cdf_p = np.cumsum(p)
    df = np.diff(f, append=f[-1])
    return float(np.sum(np.abs(cdf_k - cdf_p) * df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frac", type=float, default=0.99)
    ap.add_argument("--short_r", type=float, default=1.0)
    ap.add_argument("--long_r", type=float, default=1.75)
    ap.add_argument("--spike_r", type=float, nargs=3, default=[0.7, 1.3, 2.5])
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    with h5py.File(args.data_path, "r") as f:
        R = f["R_values"][...]
        t = f["times"][...]
        y = f["expectations"][...]
        orb = f["hf_orbital_energies"][...]

    model, _ = load_checkpoint_model(args.checkpoint, device=args.device)
    omega = compute_learned_omega(model, orb, R, args.device)

    freqs, P = true_spectrum(y, t)

    n_R = len(R)
    omega_op = np.array([operational_ceiling(P[i], freqs, args.frac) for i in range(n_R)])
    D = np.array([miss_distance_D(omega[i], freqs, P[i], omega_op[i]) for i in range(n_R)])
    W1 = np.array([w1_distance(omega[i], freqs, P[i], omega_op[i]) for i in range(n_R)])

    # ── Panel A+B: ω_op, D, W1 vs R ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))

    ax = axes[0]
    ax.plot(R, omega_op, color="tab:purple", lw=1.8, label=fr"$\omega_{{op}}(R)$ ({int(args.frac*100)}% energy)")
    ax.plot(R, omega.max(axis=1), color="tab:blue", lw=1.5, alpha=0.7,
            label=r"max$_k\,|\omega_k(R)|$ (learned)")
    ax.set_xlabel("R (Å)"); ax.set_ylabel(r"$\omega$  ($E_h$)")
    ax.set_title("Operational ceiling vs learned range")
    ax.axvline(args.short_r, ls=":", color="gray"); ax.axvline(args.long_r, ls=":", color="gray")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(R, D, color="tab:red", lw=1.8)
    ax.set_xlabel("R (Å)"); ax.set_ylabel(r"$D(R)$ ($E_h^2$)")
    ax.set_title(r"Amplitude-weighted miss: $\int |Y|^2 \min_k |\omega-\omega_k|^2\,d\omega / \int|Y|^2\,d\omega$")
    ax.axvline(args.short_r, ls=":", color="gray"); ax.axvline(args.long_r, ls=":", color="gray")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(R, W1, color="tab:green", lw=1.8)
    ax.set_xlabel("R (Å)"); ax.set_ylabel(r"$W_1$  ($E_h$)")
    ax.set_title(r"$W_1$: learned $|\omega_k|$ vs $|Y|^2$ distribution")
    ax.axvline(args.short_r, ls=":", color="gray"); ax.axvline(args.long_r, ls=":", color="gray")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(args.save_dir, "freq_alignment_diagnostic.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)

    # ── Panel C: spike plots at three R ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5), sharey=False)
    for ax, R_target in zip(axes, args.spike_r):
        i = int(np.argmin(np.abs(R - R_target)))
        op = omega_op[i]
        f_max = max(op * 1.4, 4.0)
        m = freqs <= f_max
        ax.fill_between(freqs[m], 0, P[i, m] / P[i, m].max(),
                        color="tab:red", alpha=0.35, label=fr"$|Y_{{true}}|^2$ (R={R[i]:.2f})")
        ax.axvline(op, ls="--", color="tab:red", alpha=0.7, label=fr"$\omega_{{op}}={op:.2f}$")
        for w in np.abs(omega[i]):
            if w <= f_max:
                ax.axvline(w, ls="-", color="tab:blue", alpha=0.4, lw=0.8)
        ax.axvline(np.nan, ls="-", color="tab:blue", alpha=0.4, lw=0.8, label=r"learned $|\omega_k|$")
        ax.set_xlabel(r"$\omega$  ($E_h$)")
        ax.set_title(fr"R = {R[i]:.2f} Å,  D={D[i]:.3f},  $W_1$={W1[i]:.3f}")
        ax.set_xlim(0, f_max); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("normalized amplitude")
    fig.tight_layout()
    out2 = os.path.join(args.save_dir, "freq_alignment_spikes.pdf")
    fig.savefig(out2); fig.savefig(out2.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)

    short = R < args.short_r
    mid = (R >= args.short_r) & (R < args.long_r)
    long_ = R >= args.long_r
    print(f"{'regime':>14s}  {'<ω_op>':>8s}  {'<D>':>8s}  {'<W1>':>8s}")
    for name, m in [("short R<{:.1f}".format(args.short_r), short),
                    ("mid {:.1f}-{:.1f}".format(args.short_r, args.long_r), mid),
                    ("long R>={:.1f}".format(args.long_r), long_)]:
        if m.any():
            print(f"{name:>14s}  {omega_op[m].mean():8.3f}  {np.nanmean(D[m]):8.3f}  {np.nanmean(W1[m]):8.3f}")
    print(f"[saved] {out}")
    print(f"[saved] {out2}")


if __name__ == "__main__":
    main()
