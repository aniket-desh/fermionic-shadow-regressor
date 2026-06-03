"""
FFT-amplitude diagnostic: stalled vs mistargeted at high ω.

The ObservableRegressor has no separable per-k amplitude head — the trunk
mixes Fourier features and emits all observables jointly. We recover the
effective per-frequency amplitude operationally by sweeping y_pred(t)
over the same t-grid as the training data and FFT'ing.

For each R, with mean removal + Hann window:
  S_pred(R, ω) = Σ_μ |Y_pred,μ(R, ω)|^2
  S_true(R, ω) = Σ_μ |Y_true,μ(R, ω)|^2

Operational ceiling ω_op(R): smallest ω capturing 99% of S_true(R, ω).
Hi-band [ω_hi_lo, ω_op(R)], default ω_hi_lo = 4 E_h.

Decision logic per R-regime:
  spectral mass fraction in hi-band:
    m_pred_hi = ∫_{hi-band} S_pred / ∫ S_pred
    m_true_hi = ∫_{hi-band} S_true / ∫ S_true
  ratio r(R) = m_pred_hi / m_true_hi.

  r ≪ 1 (e.g. < 0.2) → STALLED: model emits no high-ω content, even
                       though truth has it. Loss/SNR shortcut.
  r ≈ 1 but D(R) (alignment) large → MISTARGETED: model has high-ω
                       energy but at wrong frequencies. Basis mismatch.
  r ≈ 1 and D small  → not the bottleneck (look elsewhere).
"""
from __future__ import annotations

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model


def predict_grid(model, R, t, orb, device, batch_R=8):
    """Run the trained regressor on the full (R, t) grid. Returns (n_R, n_t, n_obs)."""
    cfg = model.config
    n_R, n_t = len(R), len(t)
    out = np.empty((n_R, n_t, cfg.n_observables), dtype=np.float32)
    t_t = torch.tensor(t, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i0 in range(0, n_R, batch_R):
            i1 = min(i0 + batch_R, n_R)
            R_chunk = torch.tensor(R[i0:i1], dtype=torch.float32, device=device)
            orb_chunk = torch.tensor(orb[i0:i1], dtype=torch.float32, device=device)
            B = i1 - i0
            R_e = R_chunk[:, None].expand(B, n_t).reshape(-1, 1)
            t_e = t_t[None, :].expand(B, n_t).reshape(-1, 1)
            rt = torch.cat([R_e, t_e], dim=-1)
            if cfg.n_orb_features > 0:
                orb_e = orb_chunk[:, None, :].expand(B, n_t, orb_chunk.shape[1]).reshape(-1, orb_chunk.shape[1])
                y = model(rt, orb_e)
            else:
                y = model(rt)
            out[i0:i1] = y.detach().cpu().numpy().reshape(B, n_t, cfg.n_observables)
    return out


def spectrum(y, t):
    n = len(t); dt = t[1] - t[0]
    freqs = 2 * np.pi * np.fft.rfftfreq(n, d=dt)
    y0 = y - y.mean(axis=1, keepdims=True)
    hann = np.hanning(n)[None, :, None]
    Y = np.fft.rfft(y0 * hann, axis=1)
    return freqs, (np.abs(Y) ** 2).sum(axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frac", type=float, default=0.99)
    ap.add_argument("--hi_lo", type=float, default=4.0,
                    help="Lower edge of high-ω band (E_h).")
    ap.add_argument("--short_r", type=float, default=1.0)
    ap.add_argument("--long_r", type=float, default=1.75)
    ap.add_argument("--batch_R", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    with h5py.File(args.data_path, "r") as f:
        R = f["R_values"][...]
        t = f["times"][...]
        y_true = f["expectations"][...]
        orb = f["hf_orbital_energies"][...]

    model, _ = load_checkpoint_model(args.checkpoint, device=args.device)

    print(f"[predict] regressor over {len(R)} R × {len(t)} t = {len(R)*len(t)} samples")
    y_pred = predict_grid(model, R, t, orb, args.device, batch_R=args.batch_R)
    print(f"[predict] done.  pred shape {y_pred.shape}")

    freqs, S_true = spectrum(y_true, t)
    _, S_pred = spectrum(y_pred, t)

    n_R = len(R)
    omega_op = np.empty(n_R)
    m_true_hi = np.empty(n_R)
    m_pred_hi = np.empty(n_R)
    for i in range(n_R):
        cum = np.cumsum(S_true[i]) / S_true[i].sum()
        omega_op[i] = freqs[min(np.searchsorted(cum, args.frac), len(freqs) - 1)]
        op = omega_op[i]
        hi_mask = (freqs >= args.hi_lo) & (freqs <= op)
        full_mask = freqs <= op
        m_true_hi[i] = S_true[i, hi_mask].sum() / max(S_true[i, full_mask].sum(), 1e-30)
        m_pred_hi[i] = S_pred[i, hi_mask].sum() / max(S_pred[i, full_mask].sum(), 1e-30)
    ratio = m_pred_hi / np.where(m_true_hi > 1e-6, m_true_hi, np.nan)

    # ── Plot ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(R, omega_op, color="tab:purple", lw=1.8, label=r"$\omega_{op}(R)$")
    ax.axhline(args.hi_lo, ls="--", color="gray", alpha=0.7, label=fr"hi-band edge = {args.hi_lo}")
    ax.set_xlabel("R (Å)"); ax.set_ylabel(r"$\omega$  ($E_h$)")
    ax.set_title("Operational band per R")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(R, m_true_hi, color="tab:red", lw=1.8, label=r"true: hi-band fraction")
    ax.plot(R, m_pred_hi, color="tab:blue", lw=1.8, label=r"pred: hi-band fraction")
    ax.set_xlabel("R (Å)"); ax.set_ylabel("fraction of in-band energy")
    ax.set_title(fr"Spectral mass in [{args.hi_lo}, $\omega_{{op}}$]")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(R, ratio, color="black", lw=1.8)
    ax.axhline(1.0, color="gray", ls="--", alpha=0.7)
    ax.axhline(0.2, color="tab:orange", ls=":", alpha=0.7, label="r=0.2 stall threshold")
    ax.set_xlabel("R (Å)"); ax.set_ylabel(r"$m_{pred}^{hi}/m_{true}^{hi}$")
    ax.set_title("Stall ratio (r ≪ 1 → stalled; r ≈ 1 → mistargeted-or-fine)")
    ax.set_yscale("symlog", linthresh=0.1)
    ax.grid(alpha=0.3); ax.legend(fontsize=9)

    ax = axes[1, 1]
    short = R < args.short_r
    mid = (R >= args.short_r) & (R < args.long_r)
    long_ = R >= args.long_r
    for mask, label, color in [(short, "short R<1.0", "tab:blue"),
                               (mid, "mid 1.0–1.75", "tab:green"),
                               (long_, "long R≥1.75", "tab:orange")]:
        if not mask.any():
            continue
        S_t = S_true[mask].mean(axis=0); S_p = S_pred[mask].mean(axis=0)
        S_t = S_t / S_t.max(); S_p = S_p / S_p.max()
        m = freqs <= 10.0
        ax.plot(freqs[m], S_t[m], color=color, lw=1.6, label=f"{label} true")
        ax.plot(freqs[m], S_p[m], color=color, lw=1.0, ls="--", label=f"{label} pred")
    ax.set_xlabel(r"$\omega$  ($E_h$)"); ax.set_ylabel("regime-mean |Y|^2 (normalized)")
    ax.set_title("Average spectra by regime")
    ax.axvline(args.hi_lo, ls=":", color="gray", alpha=0.5)
    ax.set_yscale("log"); ax.set_ylim(1e-4, 2.0)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(args.save_dir, "fft_amplitude_diagnostic.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)

    print(f"\n{'regime':>14s}  {'<ω_op>':>8s}  {'<m_true_hi>':>11s}  {'<m_pred_hi>':>11s}  {'<r>':>8s}")
    for name, m in [("short R<{:.1f}".format(args.short_r), short),
                    ("mid {:.1f}-{:.1f}".format(args.short_r, args.long_r), mid),
                    ("long R>={:.1f}".format(args.long_r), long_)]:
        if m.any():
            print(f"{name:>14s}  {omega_op[m].mean():8.3f}  "
                  f"{m_true_hi[m].mean():11.4f}  {m_pred_hi[m].mean():11.4f}  "
                  f"{np.nanmean(ratio[m]):8.3f}")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
