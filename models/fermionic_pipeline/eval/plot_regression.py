"""
Plot spectral comparison for the observable regressor.

Loads model + dataset, recomputes spectra, and produces:
  1. Per-geometry spectral comparison (model vs exact vs exact gaps)
  2. Summary: Pearson and MSE vs R
  3. Example observable time series
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.inference.spectral_analysis import extract_peaks, spectral_analysis
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model


def _get_orb_energies(handle, r_idx):
    if handle.hf_orbital_energies is not None:
        return handle.hf_orbital_energies[r_idx]
    return None


# Module-level omega_op source override. When set (via set_omega_source), all
# helpers in this module — and the scripts that import them (extrapolation
# heatmap, dipole experiment) — resolve omega_op through it instead of reading
# the evaluated geometry's own dataset value. See eval/omega_source.py for why
# the dataset value is an oracle input at held-out geometries.
OMEGA_SOURCE = None


def set_omega_source(source):
    global OMEGA_SOURCE
    OMEGA_SOURCE = source


def _get_omega_op(handle, r_idx):
    if OMEGA_SOURCE is not None:
        return OMEGA_SOURCE.value(r_idx=r_idx)
    if handle.omega_op is not None:
        return float(handle.omega_op[r_idx])
    return None


@torch.no_grad()
def predict_signal_matrix(model, R, times, device, orb_energies=None, omega_op=None):
    N_t = len(times)
    rt = np.stack([np.full(N_t, R), times], axis=1).astype(np.float32)
    rt_tensor = torch.from_numpy(rt).to(device)
    orb_e = None
    if orb_energies is not None:
        orb_e = torch.from_numpy(
            np.tile(orb_energies.astype(np.float32), (N_t, 1))
        ).to(device)
    omega_op_t = None
    if omega_op is not None:
        omega_op_t = torch.full((N_t,), float(omega_op), dtype=torch.float32, device=device)
    pred = model(rt_tensor, orb_energies=orb_e, omega_op=omega_op_t).cpu().numpy()
    return pred.T  # (K, N_t)


def plot_spectra(handle, model, test_r_indices, device, save_dir, ljung_box_p=0.06):
    """Per-geometry spectral comparison: model (red) vs exact (blue) vs exact gaps (green)."""
    n_test = len(test_r_indices)
    n_cols = min(4, n_test)
    n_rows = (n_test + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for panel_idx, r_idx in enumerate(test_r_indices):
        row, col = divmod(panel_idx, n_cols)
        ax = axes[row, col]
        R = float(handle.R_values[r_idx])

        D_model = predict_signal_matrix(model, R, handle.times, device, orb_energies=_get_orb_energies(handle, r_idx), omega_op=_get_omega_op(handle, r_idx))
        D_exact = handle.expectations[r_idx].T

        omega_m, spec_m, _ = spectral_analysis(D_model, handle.times, ljung_box_p=ljung_box_p)
        omega_e, spec_e, _ = spectral_analysis(D_exact, handle.times, ljung_box_p=ljung_box_p)

        # Normalize for visual comparison
        spec_e_n = spec_e / max(spec_e.max(), 1e-12)
        spec_m_n = spec_m / max(spec_m.max(), 1e-12)

        ax.plot(omega_e, spec_e_n, "b-", alpha=0.8, label="Exact", linewidth=1)
        ax.plot(omega_m, spec_m_n, "r-", alpha=0.7, label="Model", linewidth=1)

        # Exact energy gaps as vertical lines
        eigvals = handle.eigvals[r_idx]
        gaps = eigvals[1:] - eigvals[0]
        omega_max = min(10.0, omega_e[-1])
        for g in gaps:
            if 0 < g < omega_max:
                ax.axvline(g, color="green", alpha=0.25, linewidth=0.7)

        # Pearson annotation
        pearsons = []
        for i in range(D_exact.shape[0]):
            se, sm = np.std(D_exact[i]), np.std(D_model[i])
            if se > 1e-12 and sm > 1e-12:
                pearsons.append(np.corrcoef(D_exact[i], D_model[i])[0, 1])
        mean_r = np.nanmean(pearsons) if pearsons else 0.0
        mse = np.mean((D_model - D_exact) ** 2)

        ax.set_title(f"R = {R:.2f} Å   (r = {mean_r:.2f}, MSE = {mse:.1e})", fontsize=9)
        ax.set_xlabel("ω (Eₕ)", fontsize=8)
        ax.set_ylabel("I(ω) [normalized]", fontsize=8)
        ax.set_xlim(0, omega_max)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)

    for panel_idx in range(n_test, n_rows * n_cols):
        row, col = divmod(panel_idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Spectral Comparison: Model (red) vs Exact (blue), gaps (green)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "spectral_comparison.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[done] {path}")


def plot_summary(handle, model, test_r_indices, device, save_dir):
    """Summary: Pearson correlation and MSE vs bond length R."""
    Rs, mses, pearsons, range_ratios = [], [], [], []

    for r_idx in test_r_indices:
        R = float(handle.R_values[r_idx])
        D_model = predict_signal_matrix(model, R, handle.times, device, orb_energies=_get_orb_energies(handle, r_idx), omega_op=_get_omega_op(handle, r_idx))
        D_exact = handle.expectations[r_idx].T
        Rs.append(R)
        mses.append(np.mean((D_model - D_exact) ** 2))

        ps, rrs = [], []
        for i in range(D_exact.shape[0]):
            se, sm = np.std(D_exact[i]), np.std(D_model[i])
            if se > 1e-12 and sm > 1e-12:
                ps.append(np.corrcoef(D_exact[i], D_model[i])[0, 1])
                rrs.append(sm / se)
        pearsons.append(np.nanmean(ps) if ps else 0.0)
        range_ratios.append(np.nanmean(rrs) if rrs else 0.0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    ax1.plot(Rs, pearsons, "o-", color="tab:blue", markersize=5)
    ax1.set_xlabel("R (Å)")
    ax1.set_ylabel("Mean Pearson r")
    ax1.set_title("Observable Correlation vs R")
    ax1.set_ylim(-0.2, 1.05)
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(Rs, mses, "o-", color="tab:red", markersize=5)
    ax2.set_xlabel("R (Å)")
    ax2.set_ylabel("MSE")
    ax2.set_title("Prediction Error vs R")
    ax2.grid(True, alpha=0.3)

    ax3.plot(Rs, range_ratios, "o-", color="tab:green", markersize=5)
    ax3.set_xlabel("R (Å)")
    ax3.set_ylabel("Range Ratio (model/exact)")
    ax3.set_title("Amplitude Ratio vs R")
    ax3.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax3.set_ylim(0, 2.0)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, "regression_summary.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[done] {path}")


def plot_time_series(handle, model, test_r_indices, device, save_dir, n_obs=4, n_geom=7):
    """Example observable time series across geometries."""
    # Pick n_geom geometries evenly spanning the test set
    n_test = len(test_r_indices)
    if n_test <= n_geom:
        indices = list(test_r_indices)
    else:
        step = (n_test - 1) / (n_geom - 1)
        indices = [test_r_indices[int(round(i * step))] for i in range(n_geom)]

    fig, axes = plt.subplots(len(indices), n_obs, figsize=(4 * n_obs, 3 * len(indices)), squeeze=False)

    for row, r_idx in enumerate(indices):
        R = float(handle.R_values[r_idx])
        D_model = predict_signal_matrix(model, R, handle.times, device, orb_energies=_get_orb_energies(handle, r_idx), omega_op=_get_omega_op(handle, r_idx))
        D_exact = handle.expectations[r_idx].T  # (K, N_t)

        # Pick observables with largest exact signal variance
        variances = np.var(D_exact, axis=1)
        top_obs = np.argsort(variances)[::-1][:n_obs]

        for col, obs_idx in enumerate(top_obs):
            ax = axes[row, col]
            t = handle.times
            ax.plot(t, D_exact[obs_idx], "b-", alpha=0.7, linewidth=0.8, label="Exact")
            ax.plot(t, D_model[obs_idx], "r-", alpha=0.6, linewidth=0.8, label="Model")
            key = handle.observable_keys[obs_idx]
            ax.set_title(f"R={R:.2f}, obs {key}", fontsize=8)
            ax.tick_params(labelsize=6)
            if row == len(indices) - 1:
                ax.set_xlabel("t (a.u.)", fontsize=7)
            if col == 0:
                ax.set_ylabel("⟨Γ⟩", fontsize=7)
                ax.legend(fontsize=6)

    fig.suptitle("Observable Time Series: Model (red) vs Exact (blue)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(save_dir, "time_series.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[done] {path}")


def _standardize_rows(D):
    """Standardize each row of D (zero mean, unit variance)."""
    mu = D.mean(axis=1, keepdims=True)
    sigma = D.std(axis=1, ddof=1, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    return (D - mu) / sigma


def plot_chan_pipeline(handle, model, test_r_indices, device, save_dir, ljung_box_p=0.06):
    """Chan et al. Fig 2-style plots: D matrix, covariance C, and spectrum.

    For each test geometry, produces a 2×3 figure:
      Row 1 (exact):  D^T heatmap | C = D^T D heatmap | I(E) spectrum
      Row 2 (model):  D^T heatmap | C = D^T D heatmap | I(E) spectrum
    """
    from fermionic_pipeline.inference.spectral_analysis import _ljung_box_screen

    for r_idx in test_r_indices:
        R = float(handle.R_values[r_idx])
        D_model = predict_signal_matrix(model, R, handle.times, device, orb_energies=_get_orb_energies(handle, r_idx), omega_op=_get_omega_op(handle, r_idx))
        D_exact = handle.expectations[r_idx].T  # (K, N_t)

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))

        for row, (D, label) in enumerate([(D_exact, "Exact"), (D_model, "Model")]):
            D_std = _standardize_rows(D)

            # Ljung-Box screening
            if ljung_box_p is not None:
                D_screened, kept = _ljung_box_screen(D_std, p_threshold=ljung_box_p)
            else:
                D_screened = D_std

            # (a) Data matrix D^T
            ax_d = axes[row, 0]
            vmax = np.percentile(np.abs(D_std), 99)
            im = ax_d.imshow(
                D_std, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                extent=[0, len(handle.times), D_std.shape[0], 0],
            )
            ax_d.set_xlabel("time index $n$", fontsize=9)
            ax_d.set_ylabel("observable index $k$", fontsize=9)
            ax_d.set_title(f"{label}: data matrix $\\mathbf{{D}}$", fontsize=10)
            plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)

            # (b) Covariance C = D^T D
            ax_c = axes[row, 1]
            C = D_screened.T @ D_screened
            vmax_c = np.percentile(np.abs(C), 99)
            im_c = ax_c.imshow(
                C, aspect="auto", cmap="RdBu_r", vmin=-vmax_c, vmax=vmax_c,
                extent=[0, len(handle.times), len(handle.times), 0],
            )
            ax_c.set_xlabel("time index $n$", fontsize=9)
            ax_c.set_ylabel("time index $n$", fontsize=9)
            n_kept = D_screened.shape[0]
            ax_c.set_title(
                f"{label}: $\\mathbf{{C}} = \\mathbf{{D}}^T\\mathbf{{D}}$ "
                f"({n_kept}/{D_std.shape[0]} obs)",
                fontsize=10,
            )
            plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)

            # (c) Shadow spectrum I(E)
            ax_s = axes[row, 2]
            N_T = D_screened.shape[1]
            r_eig = min(10, N_T // 2)
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            idx = np.argsort(eigvals_C)[::-1][:r_eig]
            V = eigvecs_C[:, idx]

            dt = handle.times[1] - handle.times[0] if len(handle.times) > 1 else 1.0
            w = np.hanning(N_T)
            Y = V.T * w
            F = np.fft.rfft(Y, axis=1)
            omega = 2 * np.pi * np.fft.rfftfreq(N_T, d=dt)
            spectrum = np.sum(np.abs(F) ** 2, axis=0).real

            # Normalize
            spectrum_n = spectrum / max(spectrum.max(), 1e-12)
            ax_s.plot(omega, spectrum_n, color="tab:blue" if row == 0 else "tab:red", linewidth=1.2)
            ax_s.set_xlabel("$E$ ($E_h$)", fontsize=9)
            ax_s.set_ylabel("$I(E)$", fontsize=9)
            ax_s.set_title(f"{label}: shadow spectrum", fontsize=10)

            # Exact energy gaps as dashed lines
            eigvals = handle.eigvals[r_idx]
            gaps = eigvals[1:] - eigvals[0]
            omega_max = min(10.0, omega[-1])
            for g in gaps:
                if 0 < g < omega_max:
                    ax_s.axvline(g, color="green", alpha=0.4, linewidth=0.7, linestyle="--")
            ax_s.set_xlim(0, omega_max)

        fig.suptitle(
            f"Chan et al. pipeline — R = {R:.2f} Å  (H4, 8 qubits)",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(save_dir, f"chan_pipeline_R{R:.2f}.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"[done] {path}")


@torch.no_grad()
def plot_coherence_heatmap(handle, model, test_r_indices, device, save_dir, window=20,
                           train_R_range=None, train_t_range=None):
    """Heatmap of windowed Pearson r as a function of (R, t).

    Shows model coherence time — at large R the model tracks exact
    observables across the full t range; at short R it loses phase quickly.

    A dashed rectangle marks the training-data region in (R, t). When the eval
    grid is extended to extrapolated geometries/times, the box delimits in-box
    (interpolation, Prop 1) from out-of-box (extrapolation, Prop 2) regimes.
    ``train_R_range`` / ``train_t_range`` default to the full dataset extent.
    """
    times = handle.times
    N_t = len(times)
    dt = times[1] - times[0] if N_t > 1 else 1.0

    # Sort test geometries by R for clean y-axis
    sorted_pairs = sorted(
        [(float(handle.R_values[ri]), ri) for ri in test_r_indices]
    )
    Rs = [p[0] for p in sorted_pairs]
    sorted_indices = [p[1] for p in sorted_pairs]

    stride = max(1, window // 2)
    t_starts = list(range(0, N_t - window, stride))
    t_centers = np.array([(s + window / 2) * dt for s in t_starts])

    corr_map = np.zeros((len(Rs), len(t_starts)))

    for i, r_idx in enumerate(sorted_indices):
        R = Rs[i]
        D_model = predict_signal_matrix(model, R, times, device, orb_energies=_get_orb_energies(handle, r_idx), omega_op=_get_omega_op(handle, r_idx))  # (K, N_t)
        D_exact = handle.expectations[r_idx].T  # (K, N_t)

        for j, t_start in enumerate(t_starts):
            t_end = t_start + window
            ps = []
            for k in range(D_exact.shape[0]):
                se = np.std(D_exact[k, t_start:t_end])
                sm = np.std(D_model[k, t_start:t_end])
                if se > 1e-12 and sm > 1e-12:
                    r = np.corrcoef(
                        D_exact[k, t_start:t_end],
                        D_model[k, t_start:t_end],
                    )[0, 1]
                    if np.isfinite(r):
                        ps.append(r)
            corr_map[i, j] = np.mean(ps) if ps else 0.0

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        corr_map, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0,
        extent=[t_centers[0], t_centers[-1], Rs[-1], Rs[0]],
        interpolation="nearest",
    )
    ax.set_xlabel("t (a.u.)", fontsize=11)
    ax.set_ylabel("R (Å)", fontsize=11)
    ax.set_title("Windowed Pearson r(R, t) — model vs exact observables", fontsize=12)
    plt.colorbar(im, ax=ax, label="Pearson r")

    # Training-data bounding box. Defaults to the full dataset (R, t) extent; the
    # held-out test geometries here all sit inside it, so for the in-box eval the
    # box nearly fills the panel. Once extrapolated R/t are evaluated, the box
    # stays put and the surrounding region is the extrapolation regime.
    from matplotlib.patches import Rectangle

    if train_R_range is None:
        train_R_range = (float(np.min(handle.R_values)), float(np.max(handle.R_values)))
    if train_t_range is None:
        train_t_range = (float(times[0]), float(times[-1]))
    R_lo, R_hi = train_R_range
    t_lo, t_hi = train_t_range
    # High-contrast box: a white halo under a bold blue dashed line reads clearly
    # over the whole RdYlGn range (black vanishes in red, white in yellow).
    ax.add_patch(Rectangle((t_lo, R_lo), t_hi - t_lo, R_hi - R_lo,
                           fill=False, edgecolor="white", linewidth=3.6, zorder=4))
    ax.add_patch(Rectangle((t_lo, R_lo), t_hi - t_lo, R_hi - R_lo,
                           fill=False, edgecolor="#0b3dff", linestyle="--", linewidth=2.0,
                           zorder=5, label="training region"))
    # Quantify the interpolation-vs-extrapolation contrast (Prop 1/2).
    Rs_arr = np.array(Rs)
    in_R = (Rs_arr >= R_lo) & (Rs_arr <= R_hi)
    in_t = (t_centers >= t_lo) & (t_centers <= t_hi)
    if in_R.any() and in_t.any() and not (in_R.all() and in_t.all()):
        inbox = corr_map[np.ix_(in_R, in_t)]
        omask = np.ones_like(corr_map, dtype=bool)
        omask[np.ix_(in_R, in_t)] = False
        ax.text(0.015, 0.03,
                f"in-box  $\\bar r$={np.nanmean(inbox):.2f}     "
                f"out-of-box  $\\bar r$={np.nanmean(corr_map[omask]):.2f}",
                transform=ax.transAxes, fontsize=10, color="#0b3dff",
                bbox=dict(boxstyle="round", fc="white", ec="#0b3dff", alpha=0.85))
    # Expand limits so the box is fully visible even when eval ⊆ training (now)
    # or eval ⊋ training (after the extrapolation runs). y-axis is inverted.
    ax.set_xlim(min(t_centers[0], t_lo), max(t_centers[-1], t_hi))
    ax.set_ylim(max(Rs[-1], R_hi), min(Rs[0], R_lo))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(save_dir, "coherence_heatmap.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[done] {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot regression spectral results")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ljung_box_p", type=float, default=0.06)
    parser.add_argument("--omega_op_source", type=str, default="dataset",
                        choices=["dataset", "train-interp"],
                        help="train-interp: non-oracle omega_op interpolated from "
                             "the checkpoint's training geometries only.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    handle = RegressionDatasetHandle(args.data_path)
    model, payload = load_checkpoint_model(args.checkpoint, device=device)
    test_r_indices = payload.get("test_r_indices", list(range(len(handle.R_values))))

    if args.omega_op_source == "train-interp":
        from fermionic_pipeline.eval.omega_source import OmegaOpSource
        set_omega_source(OmegaOpSource("train-interp", handle=handle, payload=payload))
        print("[info] omega_op source: train-interp (non-oracle)")

    print(f"[info] {len(test_r_indices)} test geometries, K={handle.n_observables}")

    plot_spectra(handle, model, test_r_indices, device, args.save_dir, args.ljung_box_p)
    plot_summary(handle, model, test_r_indices, device, args.save_dir)
    plot_time_series(handle, model, test_r_indices, device, args.save_dir)
    plot_chan_pipeline(handle, model, test_r_indices, device, args.save_dir, args.ljung_box_p)
    plot_coherence_heatmap(handle, model, test_r_indices, device, args.save_dir)


if __name__ == "__main__":
    main()
