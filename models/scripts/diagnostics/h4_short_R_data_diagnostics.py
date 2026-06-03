"""
Short-R data-side diagnostics for H4 v10 regression pipeline.

Hypothesis being tested: short-R failure on H4 (R<1.0 pearson ~0.4) is upstream
of the model — the training data itself is undersampled in time (Nyquist
violation), undersampled in R (local Lipschitz too high for grid spacing),
and/or the train/test split has an R-density artifact. Six diagnostics:

  D1  Per-R Nyquist headroom on the training time grid
  D2  Samples per dominant oscillation period across R (uses ω_op)
  D3  Per-observable temporal Lipschitz at one short-R geometry
  D4  Time-series visual with training-grid ticks at multiple R
  D5  Train/test R-density (split-artifact check)
  D6  Learned ω_k(R) vs. data Nyquist ceiling (model-side cross-cut)

D1 was the smoking gun: with dt=0.20 a.u., ω_max(R) reaches 23 E_h at R=0.5,
exceeding ω_Ny=15.7 E_h for all R<0.74 Å. That alone explains v10 H4 short-R
failure.

Usage:
    python3 -m scripts.diagnostics.h4_short_R_data_diagnostics \\
        --data_path  results/fermionic_pipeline/regression/h4_regress_v10/regression_targets.h5 \\
        --checkpoint results/fermionic_pipeline/regression/h4_regress_v10_model/regressor.pt \\
        --eval_path  results/fermionic_pipeline/regression/h4_regress_v10_model/eval/regressor_eval.json \\
        --save_dir   results/diagnostics/h4_short_R \\
        [--diags d1,d2,d3,d4,d5,d6]   (default: all)

All outputs (plots + JSON summaries) land in --save_dir.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _safe_savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def _omega_op_per_R(y: np.ndarray, t: np.ndarray, frac: float = 0.99) -> np.ndarray:
    """Smallest ω capturing `frac` of cumulative |Y|² (mean-subtracted, Hann)."""
    n = len(t)
    dt = t[1] - t[0]
    freqs = 2 * np.pi * np.fft.rfftfreq(n, d=dt)
    y0 = y - y.mean(axis=1, keepdims=True)
    hann = np.hanning(n)[None, :, None]
    Y = np.fft.rfft(y0 * hann, axis=1)
    P = (np.abs(Y) ** 2).sum(axis=2)
    out = np.empty(len(P))
    for i in range(len(P)):
        s = P[i].sum()
        if s < 1e-30:
            out[i] = freqs[-1]; continue
        cum = np.cumsum(P[i]) / s
        out[i] = freqs[min(np.searchsorted(cum, frac), len(freqs) - 1)]
    return out


# ─── D1: Per-R Nyquist headroom ───────────────────────────────────────────────
def diag_d1(R: np.ndarray, eigvals: np.ndarray, t: np.ndarray, save_dir: Path) -> dict:
    print("[D1] Per-R Nyquist headroom on training time grid")
    dt = float(t[1] - t[0])
    omega_ny = np.pi / dt
    omega_max = eigvals.max(axis=1) - eigvals.min(axis=1)
    ratio = omega_max / omega_ny

    aliased_R = R[ratio > 1.0]
    boundary = float(aliased_R.max()) if len(aliased_R) else None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(R, omega_max, lw=1.6, color="tab:blue", label=r"$\omega_{\max}(R) = E_{\max}-E_{\min}$")
    ax.axhline(omega_ny, color="tab:red", ls="--",
               label=fr"$\omega_{{Ny}}=\pi/dt={omega_ny:.2f}\,E_h$")
    ax.fill_between(R, omega_ny, omega_max, where=(omega_max > omega_ny),
                    color="tab:red", alpha=0.18, label="aliased region")
    if boundary is not None:
        ax.axvline(boundary, color="tab:red", ls=":", alpha=0.6,
                   label=fr"aliasing onset $R={boundary:.2f}\,$Å")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel(r"$\omega\;(E_h)$")
    ax.set_title(r"D1: Hamiltonian spectral width vs. time-grid Nyquist ceiling")
    ax.legend(loc="upper right")
    _safe_savefig(fig, save_dir / "d1_nyquist_headroom.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(R, ratio, lw=1.6, color="tab:purple")
    ax.axhline(1.0, color="tab:red", ls="--", label="aliasing threshold")
    ax.axhline(0.5, color="tab:orange", ls=":", label="2× safety factor")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel(r"$\omega_{\max}(R)\,/\,\omega_{Ny}$")
    ax.set_title("D1: Aliasing ratio across the PES (ratio>1 = aliased data)")
    ax.legend()
    _safe_savefig(fig, save_dir / "d1_aliasing_ratio.png")

    return {
        "dt_au": dt,
        "omega_nyquist_Eh": float(omega_ny),
        "omega_max_min": float(omega_max.min()),
        "omega_max_max": float(omega_max.max()),
        "ratio_max": float(ratio.max()),
        "aliasing_boundary_R_Ang": boundary,
        "n_R_aliased": int((ratio > 1.0).sum()),
        "recommended_dt_at_R_min_au": float(np.pi / omega_max.max()),
        "recommended_dt_safety2_au": float(np.pi / (2 * omega_max.max())),
    }


# ─── D2: Samples per dominant oscillation period ──────────────────────────────
def diag_d2(R: np.ndarray, y: np.ndarray, t: np.ndarray, save_dir: Path) -> dict:
    print("[D2] Samples per dominant oscillation period across R")
    dt = float(t[1] - t[0])
    omega_op = _omega_op_per_R(y, t, frac=0.99)
    T_dom = np.where(omega_op > 1e-9, 2 * np.pi / omega_op, np.inf)
    samples_per_period = T_dom / dt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(R, samples_per_period, lw=1.6, color="tab:blue",
            label=r"$N_{samp}(R) = T_{dom}(R)/dt$")
    ax.axhline(8.0, color="tab:red", ls="--", label="rule-of-thumb floor (8/period)")
    ax.axhline(2.0, color="tab:purple", ls=":", label="hard Nyquist (2/period)")
    ax.set_yscale("log")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("samples per dominant period")
    ax.set_title(r"D2: training time-grid resolution vs. dominant period $T_{dom}=2\pi/\omega_{op}$")
    ax.legend()
    _safe_savefig(fig, save_dir / "d2_samples_per_period.png")

    return {
        "dt_au": dt,
        "omega_op_min_Eh": float(omega_op.min()),
        "omega_op_max_Eh": float(omega_op.max()),
        "samples_per_period_min": float(samples_per_period.min()),
        "samples_per_period_max": float(samples_per_period.max()),
        "n_R_below_8_per_period": int((samples_per_period < 8).sum()),
        "n_R_below_2_per_period": int((samples_per_period < 2).sum()),
    }


# ─── D3: Per-observable temporal Lipschitz at short R ─────────────────────────
def diag_d3(R: np.ndarray, y: np.ndarray, t: np.ndarray, save_dir: Path,
            short_R_target: float) -> dict:
    print(f"[D3] Per-observable temporal Lipschitz at R≈{short_R_target}")
    r_idx = int(np.argmin(np.abs(R - short_R_target)))
    actual_R = float(R[r_idx])
    y_r = y[r_idx]                                # (n_t, K)
    dt = float(t[1] - t[0])

    diff = np.abs(np.diff(y_r, axis=0))           # (n_t-1, K)
    rng = y_r.max(axis=0) - y_r.min(axis=0)       # (K,)
    rng_safe = np.where(rng > 1e-12, rng, 1.0)
    rel_jump = diff / rng_safe[None, :]
    max_rel_jump = rel_jump.max(axis=0)
    mean_rel_jump = rel_jump.mean(axis=0)

    K = y_r.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.bar(np.arange(K), max_rel_jump, color="tab:blue", alpha=0.75)
    ax.axhline(1.0, color="tab:red", ls="--", label="full signal range / step")
    ax.axhline(0.5, color="tab:orange", ls=":", label="50% range / step")
    ax.set_xlabel("observable index μ")
    ax.set_ylabel(r"$\max_t\,|\Delta y_\mu|/\mathrm{range}(y_\mu)$")
    ax.set_title(f"D3: max single-step jump / signal range, R={actual_R:.2f} Å")
    ax.legend()

    ax = axes[1]
    worst = int(np.argmax(max_rel_jump))
    ax.plot(t, y_r[:, worst], lw=1.0, color="tab:blue", label=f"obs {worst}")
    ax.scatter(t, y_r[:, worst], s=4, color="tab:blue", alpha=0.4)
    ax.set_xlabel("t (a.u.)")
    ax.set_ylabel("⟨Γ_μ(t)⟩")
    ax.set_title(f"D3: worst observable trace, R={actual_R:.2f} Å (max-jump = {max_rel_jump[worst]:.2f})")
    _safe_savefig(fig, save_dir / "d3_temporal_lipschitz.png")

    return {
        "R_target": short_R_target,
        "R_actual": actual_R,
        "max_rel_jump_overall": float(max_rel_jump.max()),
        "mean_rel_jump_overall": float(mean_rel_jump.mean()),
        "n_observables_with_jump_above_0.5": int((max_rel_jump > 0.5).sum()),
        "n_observables_with_jump_above_1.0": int((max_rel_jump > 1.0).sum()),
        "worst_observable_idx": int(np.argmax(max_rel_jump)),
    }


# ─── D4: Time-series with training-grid ticks at several R ────────────────────
def diag_d4(R: np.ndarray, y: np.ndarray, t: np.ndarray, save_dir: Path,
            R_targets: list[float], n_show_obs: int = 1) -> dict:
    print(f"[D4] Time-series with training-grid markers at R={R_targets}")
    dt = float(t[1] - t[0])
    n_R = len(R_targets)
    fig, axes = plt.subplots(n_R, 1, figsize=(11, 3.0 * n_R), squeeze=False)

    info = []
    for row, R_t in enumerate(R_targets):
        r_idx = int(np.argmin(np.abs(R - R_t)))
        y_r = y[r_idx]
        rng = y_r.max(axis=0) - y_r.min(axis=0)
        worst = int(np.argmax(rng))                 # show observable with biggest range

        ax = axes[row, 0]
        for k in range(min(n_show_obs, y_r.shape[1])):
            obs_idx = worst if k == 0 else k
            ax.plot(t, y_r[:, obs_idx], lw=0.9, alpha=0.85, label=f"obs {obs_idx}")
            ax.scatter(t, y_r[:, obs_idx], s=2, alpha=0.45)

        ymin, ymax = ax.get_ylim()
        ax.vlines(t, ymin, ymin + 0.02 * (ymax - ymin),
                  color="black", alpha=0.25, lw=0.5)
        ax.set_xlim(0, t.max())
        ax.set_xlabel("t (a.u.)")
        ax.set_ylabel("⟨Γ_μ(t)⟩")
        ax.set_title(f"D4: R={R[r_idx]:.2f} Å | dt={dt:.3f} | {len(t)} samples in [0, {t[-1]:.0f}]")
        ax.legend(loc="upper right", fontsize=8)
        info.append({"R_target": R_t, "R_actual": float(R[r_idx]), "worst_obs": worst})
    _safe_savefig(fig, save_dir / "d4_signal_with_grid.png")
    return {"per_R": info, "dt_au": dt, "n_t_samples": int(len(t))}


# ─── D5: Train/test R-density ─────────────────────────────────────────────────
def diag_d5(R: np.ndarray, train_idx: list[int], test_idx: list[int],
            save_dir: Path) -> dict:
    print("[D5] Train/test R-density")
    train_R = R[np.asarray(train_idx, dtype=int)]
    test_R = R[np.asarray(test_idx, dtype=int)]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(train_R, bins=40, range=(R.min(), R.max()), alpha=0.55,
            label=f"train (n={len(train_R)})", color="tab:blue")
    ax.hist(test_R, bins=40, range=(R.min(), R.max()), alpha=0.55,
            label=f"test (n={len(test_R)})", color="tab:red")
    ax.axvline(0.74, color="black", ls=":", alpha=0.6,
               label="aliasing onset (D1: R=0.74)")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel("count")
    ax.set_title("D5: train / test R coverage")
    ax.legend()
    _safe_savefig(fig, save_dir / "d5_train_test_density.png")

    n_train_short = int((train_R < 1.0).sum())
    n_test_short = int((test_R < 1.0).sum())
    return {
        "n_train": int(len(train_R)),
        "n_test": int(len(test_R)),
        "n_train_R_below_1.0": n_train_short,
        "n_test_R_below_1.0": n_test_short,
        "n_train_R_aliased": int((train_R < 0.74).sum()),
        "n_test_R_aliased": int((test_R < 0.74).sum()),
        "train_R_min": float(train_R.min()),
        "test_R_min": float(test_R.min()),
    }


# ─── D6: Learned ω_k(R) vs data Nyquist ───────────────────────────────────────
def diag_d6(data_path: str, checkpoint: str, save_dir: Path,
            device: str = "cpu") -> dict:
    print("[D6] Learned ω_k(R) vs. data Nyquist ceiling")
    import torch
    from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model

    with h5py.File(data_path, "r") as f:
        R = f["R_values"][...]
        eigvals = f["eigvals"][...]
        t = f["times"][...]
        orb = f["hf_orbital_energies"][...] if "hf_orbital_energies" in f else None

    model, _ = load_checkpoint_model(checkpoint, device=device)
    cfg = model.config

    with torch.no_grad():
        if model.freq_net is None:
            omega_base = model.omega_base.detach().cpu().numpy()
            omega = np.broadcast_to(omega_base, (len(R), len(omega_base))).copy()
        else:
            omega_base = model.omega_base.detach().cpu().numpy()
            if cfg.n_orb_features > 0 and orb is not None:
                x = torch.tensor(orb, dtype=torch.float32, device=device)
            else:
                x = torch.tensor(R[:, None], dtype=torch.float32, device=device)
            shift = model.freq_net(x).detach().cpu().numpy()
            omega = omega_base[None, :] + shift
    abs_omega = np.abs(omega)

    dt = float(t[1] - t[0])
    omega_ny = np.pi / dt
    omega_max_data = eigvals.max(axis=1) - eigvals.min(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for k in range(abs_omega.shape[1]):
        ax.plot(R, abs_omega[:, k], color="tab:blue", alpha=0.07, lw=0.55)
    ax.plot(R, abs_omega.max(axis=1), color="navy", lw=1.4, label=r"$\max_k|\omega_k(R)|$")
    ax.plot(R, omega_max_data, color="tab:green", lw=1.4,
            label=r"true $\omega_{\max}(R)$ (data spectrum)")
    ax.axhline(omega_ny, color="tab:red", ls="--",
               label=fr"$\omega_{{Ny}}=\pi/dt={omega_ny:.2f}\,E_h$")
    ax.set_xlabel("R (Å)")
    ax.set_ylabel(r"$|\omega|\;(E_h)$")
    ax.set_title("D6: learned freq_net output ω_k(R) vs. data Nyquist ceiling")
    ax.legend()
    _safe_savefig(fig, save_dir / "d6_learned_omega_vs_nyquist.png")

    n_above_ny = int((abs_omega > omega_ny).sum())
    pct_above_ny = float(100.0 * n_above_ny / abs_omega.size)
    return {
        "n_fourier": int(abs_omega.shape[1]),
        "omega_nyquist_Eh": float(omega_ny),
        "learned_omega_max": float(abs_omega.max()),
        "learned_omega_min": float(abs_omega.min()),
        "n_omega_above_nyquist": n_above_ny,
        "pct_omega_above_nyquist": pct_above_ny,
        "n_R_with_any_omega_above_nyquist": int((abs_omega > omega_ny).any(axis=1).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True,
                    help="HDF5 produced by regression_dataset (must contain eigvals, times, expectations)")
    ap.add_argument("--checkpoint", default=None,
                    help="Optional regressor.pt — required for D6")
    ap.add_argument("--eval_path", default=None,
                    help="Optional regressor_eval.json — used to source train/test indices for D5")
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--diags", default="d1,d2,d3,d4,d5,d6",
                    help="comma-separated subset of d1..d6")
    ap.add_argument("--short_R_target", type=float, default=0.7,
                    help="R for D3 single-geometry plot")
    ap.add_argument("--d4_R_targets", default="0.6,1.0,1.5,2.5",
                    help="comma-separated R values for D4 panels")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    diags = {d.strip().lower() for d in args.diags.split(",")}
    summary = {"data_path": args.data_path, "checkpoint": args.checkpoint, "diagnostics": {}}

    with h5py.File(args.data_path, "r") as f:
        R = f["R_values"][...]
        t = f["times"][...]
        y = f["expectations"][...]
        eigvals = f["eigvals"][...]

    if "d1" in diags:
        summary["diagnostics"]["d1"] = diag_d1(R, eigvals, t, save_dir)
    if "d2" in diags:
        summary["diagnostics"]["d2"] = diag_d2(R, y, t, save_dir)
    if "d3" in diags:
        summary["diagnostics"]["d3"] = diag_d3(R, y, t, save_dir, args.short_R_target)
    if "d4" in diags:
        d4_R = [float(x) for x in args.d4_R_targets.split(",")]
        summary["diagnostics"]["d4"] = diag_d4(R, y, t, save_dir, d4_R)
    if "d5" in diags:
        train_idx, test_idx = None, None
        if args.checkpoint and Path(args.checkpoint).exists():
            import torch
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            train_idx = payload.get("train_r_indices")
            test_idx = payload.get("test_r_indices")
        if (train_idx is None or test_idx is None) and args.eval_path and Path(args.eval_path).exists():
            ev = json.loads(Path(args.eval_path).read_text())
            test_R_set = sorted({r["R"] for r in ev["results"]})
            test_idx = [int(np.argmin(np.abs(R - rv))) for rv in test_R_set]
            train_idx = [i for i in range(len(R)) if i not in set(test_idx)]
        if train_idx is not None and test_idx is not None:
            summary["diagnostics"]["d5"] = diag_d5(R, train_idx, test_idx, save_dir)
        else:
            print("[D5] skipped — no checkpoint or eval JSON to source train/test indices from")
    if "d6" in diags:
        if args.checkpoint and Path(args.checkpoint).exists():
            summary["diagnostics"]["d6"] = diag_d6(args.data_path, args.checkpoint,
                                                  save_dir, device=args.device)
        else:
            print("[D6] skipped — needs --checkpoint")

    summary_path = save_dir / "diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
