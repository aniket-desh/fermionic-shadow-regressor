"""Dipole sample-efficiency study (H4) — the chemist-facing BO figure.

Reconstruct the dipole trace <mu_x(t)> (whose FFT is the polarizability) from a
few sampled time points, using the shipped v18-orb FSR as the GP prior mean, and
compare against a flat prior: how many quantum samples (time points) are needed
to reach a target RMSE. With a good prior the answer is "fewer".

  <mu_x(R,t)> = c_x(R) . D(R,t)            (D = degree-2 Majorana signal matrix)
  target   : c_x . D_exact   (h5 shadow signal, + optional shot noise on samples)
  FSR prior: c_x . D_pred    (v18-orb predictions)

Main panel: samples-vs-R sweep (FSR vs flat) across the dissociation curve, which
shows directly the regimes where the FSR prior helps (mid/long bond) and where it
does not (short bond). Plus a few fit-example panels. All inputs local/pyscf-free.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import solve_triangular

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model
from fermionic_pipeline.eval.plot_regression import (
    predict_signal_matrix, _get_orb_energies, _get_omega_op,
)
from fermionic_pipeline.bo import FunctionPriorGP


def _interp_fn(t_grid, vals):
    def f(x):
        return np.interp(np.asarray(x, float).reshape(-1), t_grid, vals)
    return f


def _post_var_diag(gp, x_cand, signal_variance):
    """Posterior variance at each candidate — DIAGONAL only, O(N*n). Avoids the
    full N×N covariance that FunctionPriorGP.predict(return_std=True) would build
    (the acquisition only needs per-point variance). For the RBF, k(x,x)=sv."""
    if gp.x_train is None or gp._cho_factor is None:
        return np.full(len(x_cand), signal_variance)
    k_ct = gp._kernel(x_cand, gp.x_train)
    L, lower = gp._cho_factor
    v = solve_triangular(L, k_ct.T, lower=lower, check_finite=False)
    return np.maximum(signal_variance - np.sum(v ** 2, axis=0), 0.0)


def _run_to_threshold(prior_fn, x_cand, y_true, *, max_samples, target_rmse,
                      length_scale, signal_variance, noise_variance, shot_sigma, rng):
    """Active learning by max posterior variance until posterior-mean RMSE (vs the
    noiseless target on the candidate grid) <= target_rmse, or max_samples hit.
    n=0 evaluates the prior alone (count 0 if already within tol). Returns
    (n_samples, gp, sampled_idx)."""
    gp = FunctionPriorGP(prior_mean=prior_fn, length_scale=length_scale,
                         signal_variance=signal_variance, noise_variance=noise_variance)

    def observe(idx):
        return y_true[idx] + (shot_sigma * rng.standard_normal(len(idx)) if shot_sigma > 0 else 0.0)

    used, n = [], 0
    while True:
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


def _bo_one_geometry(handle, model, R_npz, cx, t, x_cand, stride, fft_freq, args, Rtarget, want_fit):
    """Run the FSR-prior and flat-prior active learning at the nearest geometry to
    Rtarget over n_trials. Returns a dict with sample counts (+ fit arrays)."""
    ri = int(np.argmin(np.abs(handle.R_values - Rtarget)))
    R = float(handle.R_values[ri])
    c = cx[int(np.argmin(np.abs(R_npz - R)))]
    mu_exact = handle.expectations[ri] @ c
    D_pred = predict_signal_matrix(model, R, t, torch.device(args.device),
                orb_energies=_get_orb_energies(handle, ri),
                omega_op=_get_omega_op(handle, ri)).T
    mu_fsr = D_pred @ c
    y_true = mu_exact[::stride]
    prior_fsr = _interp_fn(t, mu_fsr)
    prior_flat = lambda x: np.zeros(len(np.asarray(x, float).reshape(-1)))
    resid = y_true - prior_fsr(x_cand)
    sv_fsr = max(np.var(resid), 1e-6)
    sv_flat = max(np.var(y_true), 1e-6)
    sig_std = max(float(np.std(y_true)), 1e-9)
    fsr_rel_err = float(np.sqrt(np.mean(resid ** 2)) / sig_std)  # 0-sample FSR error, in units of std
    spec = np.abs(np.fft.rfft(mu_exact - mu_exact.mean()))
    E_peak = float(fft_freq[1:][np.argmax(spec[1:])])
    ls = args.length_scale if args.length_scale > 0 else float(np.clip(np.pi / (2 * E_peak), 0.3, 12.0))
    tol = args.target_rmse if args.target_rmse > 0 else max(args.rel_rmse * float(np.std(y_true)), args.abs_rmse)
    shot_sigma = args.noise_sigma

    fsr_c, flat_c = [], []
    disp = None
    for trial in range(args.n_trials):
        rng = np.random.default_rng(1000 + trial)
        nf, gpf, uf = _run_to_threshold(prior_fsr, x_cand, y_true, max_samples=args.max_samples,
            target_rmse=tol, length_scale=ls, signal_variance=sv_fsr,
            noise_variance=shot_sigma**2 + 1e-8, shot_sigma=shot_sigma, rng=rng)
        rng = np.random.default_rng(1000 + trial)
        n0, gp0, _ = _run_to_threshold(prior_flat, x_cand, y_true, max_samples=args.max_samples,
            target_rmse=tol, length_scale=ls, signal_variance=sv_flat,
            noise_variance=shot_sigma**2 + 1e-8, shot_sigma=shot_sigma, rng=rng)
        fsr_c.append(nf); flat_c.append(n0)
        if want_fit and trial == 0:
            disp = dict(x=x_cand[:, 0], y=y_true, prior=prior_fsr(x_cand),
                        fsr_mean=gpf.predict(x_cand), flat_mean=gp0.predict(x_cand),
                        sampled=x_cand[uf, 0], sampled_y=y_true[uf])
    return dict(R=R, fsr=np.array(fsr_c), flat=np.array(flat_c), fit=disp,
                fsr_rel_err=fsr_rel_err, rel_tol=args.rel_rmse)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="results/fermionic_pipeline/regression/h4_regress_v13/dipole_coeffs.npz")
    ap.add_argument("--data_h5", default="results/fermionic_pipeline/regression/h4_regress_v13/regression_targets.h5")
    ap.add_argument("--checkpoint", default="results/fermionic_pipeline/regression/h4_regress_v18_v18_orb_s42_model/regressor.pt")
    ap.add_argument("--save_dir", default="results/fermionic_pipeline/regression/h4_regress_v13/bo")
    ap.add_argument("--R", type=float, nargs="+", default=[0.65, 1.50],
                    help="bond lengths for the fit-example panels (one weak, one strong)")
    ap.add_argument("--n_sweep", type=int, default=26, help="geometries in the samples-vs-R sweep")
    ap.add_argument("--sweep_lo", type=float, default=0.5)
    ap.add_argument("--sweep_hi", type=float, default=3.0)
    ap.add_argument("--rel_rmse", type=float, default=0.4,
                    help="target RMSE as a fraction of each trace's std (geometry-adaptive)")
    ap.add_argument("--abs_rmse", type=float, default=2e-3)
    ap.add_argument("--target_rmse", type=float, default=0.0, help="absolute RMSE override; 0 = rel_rmse*std")
    ap.add_argument("--max_samples", type=int, default=40)
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--n_cand", type=int, default=1200)
    ap.add_argument("--length_scale", type=float, default=0.0, help="0 = auto per-R from dominant freq")
    ap.add_argument("--noise_sigma", type=float, default=4e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    d = np.load(args.npz)
    R_npz, cx = d["R_values"], d["c_x"]
    handle = RegressionDatasetHandle(args.data_h5)
    model, _ = load_checkpoint_model(args.checkpoint, device=torch.device(args.device))
    t = handle.times
    stride = max(1, len(t) // args.n_cand)
    x_cand = t[::stride].reshape(-1, 1)
    fft_freq = np.fft.rfftfreq(len(t), d=float(t[1] - t[0])) * 2 * np.pi

    def run(Rt, want_fit):
        return _bo_one_geometry(handle, model, R_npz, cx, t, x_cand, stride, fft_freq, args, Rt, want_fit)

    print("=== samples-vs-R sweep ===")
    sweep = []
    for Rt in np.round(np.linspace(args.sweep_lo, args.sweep_hi, args.n_sweep), 3):
        r = run(float(Rt), want_fit=False)
        sweep.append(r)
        print(f"  R={r['R']:.2f}  FSR {r['fsr'].mean():4.1f}  flat {r['flat'].mean():4.1f}")
    examples = [run(float(Rt), want_fit=True) for Rt in args.R]

    Rs = np.array([s["R"] for s in sweep])
    fsr_m = np.array([s["fsr"].mean() for s in sweep])
    flat_m = np.array([s["flat"].mean() for s in sweep])
    fsr_e = np.array([1.96 * s["fsr"].std() / np.sqrt(len(s["fsr"])) for s in sweep])
    rel_err = np.array([s["fsr_rel_err"] for s in sweep])
    rel_tol = float(sweep[0]["rel_tol"])

    # persist sweep arrays so the figure can be replotted/retuned without re-running
    import json
    with open(os.path.join(args.save_dir, "dipole_bo_sweep.json"), "w") as f:
        json.dump(dict(R=Rs.tolist(), fsr=fsr_m.tolist(), flat=flat_m.tolist(),
                       fsr_ci=fsr_e.tolist(), fsr_rel_err=rel_err.tolist(),
                       rel_tol=rel_tol, max_samples=args.max_samples), f, indent=2)

    nEx = len(examples)
    fig = plt.figure(figsize=(4.6 * nEx, 9.6))
    gs = fig.add_gridspec(3, nEx, height_ratios=[1.2, 1.2, 1.2], hspace=0.40, wspace=0.26)
    for k, ex in enumerate(examples):
        ax = fig.add_subplot(gs[0, k]); c = ex["fit"]
        ax.plot(c["x"], c["y"], color="k", lw=0.9, label="target")
        ax.plot(c["x"], c["prior"], "--", color="#16a34a", lw=1.0, alpha=0.85, label="FSR prior")
        ax.plot(c["x"], c["flat_mean"], color="#9ca3af", lw=1.0, label="flat GP")
        ax.scatter(c["sampled"], c["sampled_y"], s=10, color="#16a34a", zorder=3)
        ax.set_title(f"H4, R = {ex['R']:.2f} Å   (FSR {ex['fsr'].mean():.0f} vs flat {ex['flat'].mean():.0f} samples)", fontsize=9)
        ax.set_xlabel("t (a.u.)", fontsize=9)
        if k == 0:
            ax.set_ylabel(r"$\langle\mu_x(t)\rangle$", fontsize=10); ax.legend(fontsize=7, loc="best")
    # row 1: samples-vs-R
    axs = fig.add_subplot(gs[1, :])
    axs.axhline(args.max_samples, color="#9ca3af", ls=":", lw=1, alpha=0.8)
    axs.plot(Rs, flat_m, "-o", color="#6b7280", ms=4, label="flat prior")
    axs.fill_between(Rs, fsr_m - fsr_e, fsr_m + fsr_e, color="#16a34a", alpha=0.2)
    axs.plot(Rs, fsr_m, "-o", color="#16a34a", ms=4, label="FSR prior (v18-orb)")
    axs.set_ylabel("quantum samples\nto reach tolerance", fontsize=10)
    axs.set_title("Dipole reconstruction sample efficiency vs bond length (H4, v18-orb prior)", fontsize=10)
    axs.set_ylim(-1, args.max_samples + 3); axs.legend(fontsize=9, loc="center right")
    # row 2: the smooth quantity that explains it — FSR 0-sample error vs R
    axe = fig.add_subplot(gs[2, :], sharex=axs)
    axe.plot(Rs, rel_err, "-o", color="#16a34a", ms=4, label="FSR prior error (0 samples)")
    axe.axhline(rel_tol, color="#b91c1c", ls="--", lw=1.2, label=f"tolerance ({rel_tol:g}×std)")
    axe.fill_between(Rs, 0, rel_tol, color="#16a34a", alpha=0.08)
    axe.set_xlabel("R (Å)", fontsize=11)
    axe.set_ylabel("FSR error / signal std", fontsize=10)
    axe.set_title("Why: FSR prior accuracy vs bond length (below the line → 0 samples needed)", fontsize=10)
    axe.legend(fontsize=9, loc="upper center")
    for ax in (axs, axe):
        for ex in examples:
            ax.axvline(ex["R"], color="k", ls="--", lw=0.6, alpha=0.35)
    path = os.path.join(args.save_dir, "dipole_bo_sweep_h4.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
