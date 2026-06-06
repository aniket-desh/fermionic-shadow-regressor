"""Dipole sample-efficiency study (H4) — the chemist-facing BO figure.

Reconstruct the dipole trace <mu_x(t)> (whose FFT is the polarizability) from a
few sampled time points, using the shipped v18-orb FSR as the GP prior mean, and
compare against a flat prior: how many quantum samples (time points) are needed
to reach a target RMSE. With a good prior the answer is "fewer".

  <mu_x(R,t)> = c_x(R) . D(R,t)            (D = degree-2 Majorana signal matrix)
  target  : c_x . D_exact   (h5 shadow signal, + optional shot noise on samples)
  FSR prior: c_x . D_pred   (v18-orb predictions)

All inputs are local and pyscf-free: dipole_coeffs.npz, regression_targets.h5,
v18-orb regressor.pt. Produces dipole_bo_h4.pdf (per-R fits + sample-count bars).
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model
from fermionic_pipeline.eval.plot_regression import (
    predict_signal_matrix, _get_orb_energies, _get_omega_op,
)
from scipy.linalg import solve_triangular

from fermionic_pipeline.bo import FunctionPriorGP


def _interp_fn(t_grid, vals):
    def f(x):
        return np.interp(np.asarray(x, float).reshape(-1), t_grid, vals)
    return f


def _post_var_diag(gp, x_cand, signal_variance):
    """Posterior variance at each candidate — DIAGONAL only, O(N*n). Avoids the
    full N×N covariance that FunctionPriorGP.predict(return_std=True) would build
    (the BO acquisition only needs per-point variance). For the RBF, k(x,x)=sv."""
    if gp.x_train is None or gp._cho_factor is None:
        return np.full(len(x_cand), signal_variance)
    k_ct = gp._kernel(x_cand, gp.x_train)                       # (N, n)
    L, lower = gp._cho_factor
    v = solve_triangular(L, k_ct.T, lower=lower, check_finite=False)  # (n, N)
    return np.maximum(signal_variance - np.sum(v ** 2, axis=0), 0.0)


def _run_to_threshold(prior_fn, x_cand, y_true, *, max_samples, target_rmse,
                      length_scale, signal_variance, noise_variance, shot_sigma,
                      rng, n_init=1):
    """Active learning by max posterior variance until posterior-mean RMSE (vs the
    noiseless target on the candidate grid) <= target_rmse, or max_samples hit.
    Returns (n_samples, final_gp, sampled_idx)."""
    gp = FunctionPriorGP(prior_mean=prior_fn, length_scale=length_scale,
                         signal_variance=signal_variance, noise_variance=noise_variance)

    def observe(idx):
        return y_true[idx] + (shot_sigma * rng.standard_normal(len(idx)) if shot_sigma > 0 else 0.0)

    used = []
    n = 0
    while True:
        # n=0 evaluates the PRIOR alone (count 0 if the FSR trajectory is already
        # within tolerance — the paper's convention); then sample to correct it.
        rmse = float(np.sqrt(np.mean((gp.predict(x_cand) - y_true) ** 2)))
        if rmse <= target_rmse or n >= max_samples:
            return n, gp, used
        if n == 0:
            j = int(rng.integers(len(x_cand)))
        else:
            var = _post_var_diag(gp, x_cand, signal_variance)
            var[used] = -np.inf
            j = int(np.argmax(var))
        used.append(j)
        gp.update(x_cand[j:j + 1], observe(np.array([j])))
        n += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="results/fermionic_pipeline/regression/h4_regress_v13/dipole_coeffs.npz")
    ap.add_argument("--data_h5", default="results/fermionic_pipeline/regression/h4_regress_v13/regression_targets.h5")
    ap.add_argument("--checkpoint", default="results/fermionic_pipeline/regression/h4_regress_v18_v18_orb_s42_model/regressor.pt")
    ap.add_argument("--save_dir", default="results/fermionic_pipeline/regression/h4_regress_v13/bo")
    ap.add_argument("--R", type=float, nargs="+", default=[0.65, 1.49, 2.27, 2.58])
    ap.add_argument("--rel_rmse", type=float, default=0.4,
                    help="target RMSE as a fraction of each trace's std (geometry-adaptive)")
    ap.add_argument("--abs_rmse", type=float, default=2e-3, help="floor on the target RMSE")
    ap.add_argument("--target_rmse", type=float, default=0.0,
                    help="absolute RMSE override; 0 = use rel_rmse*std (recommended)")
    ap.add_argument("--max_samples", type=int, default=40)
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--n_cand", type=int, default=1500, help="candidate time points (subsample of the t grid)")
    ap.add_argument("--length_scale", type=float, default=0.0,
                    help="RBF length scale (a.u. of time); 0 = auto per-R from the dominant frequency")
    ap.add_argument("--noise_sigma", type=float, default=4e-3,
                    help="observation noise on a sampled dipole point (kept below target_rmse)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    d = np.load(args.npz)
    R_npz, cx = d["R_values"], d["c_x"]
    handle = RegressionDatasetHandle(args.data_h5)
    model, _ = load_checkpoint_model(args.checkpoint, device=device)
    t = handle.times
    stride = max(1, len(t) // args.n_cand)
    x_cand = t[::stride].reshape(-1, 1)
    shot_sigma = args.noise_sigma
    dt = float(t[1] - t[0])
    fft_freq = np.fft.rfftfreq(len(t), d=dt) * 2 * np.pi  # angular freq grid

    cases = []
    for Rtarget in args.R:
        ri = int(np.argmin(np.abs(handle.R_values - Rtarget)))
        R = float(handle.R_values[ri])
        ci = int(np.argmin(np.abs(R_npz - R)))
        c = cx[ci]
        mu_exact = handle.expectations[ri] @ c                      # (n_t,)
        D_pred = predict_signal_matrix(model, R, t, device,
                    orb_energies=_get_orb_energies(handle, ri),
                    omega_op=_get_omega_op(handle, ri)).T            # (n_t, 120)
        mu_fsr = D_pred @ c
        y_true = mu_exact[::stride]
        prior_fsr = _interp_fn(t, mu_fsr)
        prior_flat = lambda x: np.zeros(len(np.asarray(x, float).reshape(-1)))
        # residual variance sets the GP signal amplitude (per prior)
        sv_fsr = max(np.var(y_true - prior_fsr(x_cand)), 1e-6)
        sv_flat = max(np.var(y_true), 1e-6)
        # per-R length scale: ~quarter-period of the dominant dipole frequency,
        # so the GP residual can actually track the oscillation (flat) / the
        # FSR's residual error (FSR). Clipped to a sane band.
        spec = np.abs(np.fft.rfft(mu_exact - mu_exact.mean()))
        E_peak = float(fft_freq[1:][np.argmax(spec[1:])])
        ls = args.length_scale if args.length_scale > 0 else float(np.clip(np.pi / (2 * E_peak), 0.3, 12.0))
        # geometry-adaptive target: reconstruct each trace to rel_rmse * its std
        tol = args.target_rmse if args.target_rmse > 0 else max(args.rel_rmse * float(np.std(y_true)), args.abs_rmse)

        fsr_counts, flat_counts = [], []
        gp_fsr_disp = gp_flat_disp = used_fsr_disp = None
        for trial in range(args.n_trials):
            rng = np.random.default_rng(1000 + trial)
            nf, gpf, uf = _run_to_threshold(prior_fsr, x_cand, y_true,
                max_samples=args.max_samples, target_rmse=tol,
                length_scale=ls, signal_variance=sv_fsr,
                noise_variance=shot_sigma**2 + 1e-8, shot_sigma=shot_sigma, rng=rng)
            rng = np.random.default_rng(1000 + trial)
            n0, gp0, _ = _run_to_threshold(prior_flat, x_cand, y_true,
                max_samples=args.max_samples, target_rmse=tol,
                length_scale=ls, signal_variance=sv_flat,
                noise_variance=shot_sigma**2 + 1e-8, shot_sigma=shot_sigma, rng=rng)
            fsr_counts.append(nf); flat_counts.append(n0)
            if trial == 0:
                gp_fsr_disp, gp_flat_disp, used_fsr_disp = gpf, gp0, uf
        cases.append(dict(R=R, x=x_cand[:, 0], y=y_true, prior=prior_fsr(x_cand),
                          fsr_mean=gp_fsr_disp.predict(x_cand), flat_mean=gp_flat_disp.predict(x_cand),
                          sampled=x_cand[used_fsr_disp, 0], sampled_y=y_true[used_fsr_disp],
                          fsr_counts=np.array(fsr_counts), flat_counts=np.array(flat_counts)))
        print(f"R={R:.2f}  FSR samples {np.mean(fsr_counts):.1f}±{np.std(fsr_counts):.1f}  "
              f"flat {np.mean(flat_counts):.1f}±{np.std(flat_counts):.1f}  "
              f"(saved {np.mean(np.array(flat_counts)-np.array(fsr_counts)):.1f})  [E_peak={E_peak:.2f} ls={ls:.2f} tol={tol:.4f}]")

    # ── figure: top row = per-R fits, bottom = sample-count bars ──
    nR = len(cases)
    fig = plt.figure(figsize=(3.4 * nR, 6.2))
    gs = fig.add_gridspec(2, nR, height_ratios=[2.2, 1.0], hspace=0.32, wspace=0.28)
    for k, cse in enumerate(cases):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(cse["x"], cse["y"], color="k", lw=1.0, label="target", zorder=1)
        ax.plot(cse["x"], cse["prior"], "--", color="#16a34a", lw=1.0, alpha=0.8, label="FSR prior")
        ax.plot(cse["x"], cse["fsr_mean"], color="#16a34a", lw=1.3, label="FSR GP")
        ax.plot(cse["x"], cse["flat_mean"], color="#6b7280", lw=1.1, alpha=0.9, label="flat GP")
        ax.scatter(cse["sampled"], cse["sampled_y"], s=12, color="#16a34a", zorder=3)
        ax.set_title(f"H4, R = {cse['R']:.2f} Å", fontsize=10)
        ax.set_xlabel("t (a.u.)", fontsize=9)
        if k == 0:
            ax.set_ylabel(r"$\langle\mu_x(t)\rangle$", fontsize=10)
            ax.legend(fontsize=7, loc="best", framealpha=0.85)
    for k, cse in enumerate(cases):
        ax = fig.add_subplot(gs[1, k])
        m = [cse["flat_counts"].mean(), cse["fsr_counts"].mean()]
        e = [1.96 * cse["flat_counts"].std() / np.sqrt(len(cse["flat_counts"])),
             1.96 * cse["fsr_counts"].std() / np.sqrt(len(cse["fsr_counts"]))]
        ax.bar([0, 1], m, yerr=e, capsize=4, color=["#6b7280", "#16a34a"],
               edgecolor="#111827", width=0.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["flat", "FSR"], fontsize=8)
        ax.set_ylim(0, args.max_samples * 1.05)
        if k == 0:
            ax.set_ylabel("sampled\ntime points", fontsize=9)
    fig.suptitle("Dipole reconstruction sample efficiency — FSR prior vs flat (H4, v18-orb)", fontsize=11)
    path = os.path.join(args.save_dir, "dipole_bo_h4.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
