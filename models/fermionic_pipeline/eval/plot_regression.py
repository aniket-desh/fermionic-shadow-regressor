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
from fermionic_pipeline.eval.regressor_eval import initial_values_for, predict_signal_matrix
from fermionic_pipeline.eval.nature_style import (
    apply_nature_style,
    grid_figsize,
    tex_escape,
    diverging_cmap,
    quality_cmap,
    DOUBLE_COL,
)
from fermionic_pipeline.inference.spectral_analysis import extract_peaks, spectral_analysis
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model

# Render every figure in this module (and the scripts that reuse its plotting
# helpers) with the Nature-optimised LaTeX style.
apply_nature_style()


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


def _predict_at(handle, model, r_idx, times, device):
    d0 = initial_values_for(handle, model)
    return predict_signal_matrix(
        model, float(handle.R_values[r_idx]), times, device,
        orb_energies=_get_orb_energies(handle, r_idx),
        omega_op=_get_omega_op(handle, r_idx), initial_values=d0,
    )


def plot_spectra(handle, model, test_r_indices, device, save_dir, ljung_box_p=0.06):
    """Per-geometry spectral comparison: model (red) vs exact (blue) vs exact gaps (green)."""
    n_test = len(test_r_indices)
    n_cols = min(4, n_test)
    n_rows = (n_test + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=grid_figsize(n_cols, n_rows, aspect=0.82),
                             squeeze=False)

    for panel_idx, r_idx in enumerate(test_r_indices):
        row, col = divmod(panel_idx, n_cols)
        ax = axes[row, col]
        R = float(handle.R_values[r_idx])

        D_model = _predict_at(handle, model, r_idx, handle.times, device)
        D_exact = handle.expectations[r_idx].T

        omega_m, spec_m, _ = spectral_analysis(D_model, handle.times, ljung_box_p=ljung_box_p)
        omega_e, spec_e, _ = spectral_analysis(D_exact, handle.times, ljung_box_p=ljung_box_p)

        # Normalize for visual comparison
        spec_e_n = spec_e / max(spec_e.max(), 1e-12)
        spec_m_n = spec_m / max(spec_m.max(), 1e-12)

        ax.plot(omega_e, spec_e_n, "b-", alpha=0.8, label="Exact", linewidth=0.9)
        ax.plot(omega_m, spec_m_n, "r-", alpha=0.7, label="Model", linewidth=0.9)

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

        ax.set_title(rf"$R = {R:.2f}$\,\AA \quad ($r = {mean_r:.2f}$, MSE\,$=\,${mse:.1e})")
        ax.set_xlabel(r"$\omega$ ($E_h$)")
        ax.set_ylabel(r"$I(\omega)$ [normalized]")
        ax.set_xlim(0, omega_max)
        ax.legend(loc="upper right")

    for panel_idx in range(n_test, n_rows * n_cols):
        row, col = divmod(panel_idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Spectral comparison: Model (red) vs Exact (blue), gaps (green)")
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
        D_model = _predict_at(handle, model, r_idx, handle.times, device)
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

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 3.4))

    ax1.plot(Rs, pearsons, "o-", color="tab:blue")
    ax1.set_xlabel("$R$ (Å)")
    ax1.set_ylabel(r"Mean Pearson $r$")
    ax1.set_ylim(-0.2, 1.05)
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(Rs, mses, "o-", color="tab:red")
    ax2.set_xlabel("$R$ (Å)")
    ax2.set_ylabel("MSE")
    ax2.grid(True, alpha=0.3)

    ax3.plot(Rs, range_ratios, "o-", color="tab:green")
    ax3.set_xlabel("$R$ (Å)")
    ax3.set_ylabel("Range ratio (model/exact)")
    ax3.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax3.set_ylim(0, 2.0)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, "regression_summary.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"[done] {path}")


def plot_time_series(handle, model, test_r_indices, device, save_dir, n_obs=3, n_geom=3):
    """Curated observable time series: n_geom geometries (compressed / near-equilibrium /
    stretched) x n_obs channels spanning the signal-variance range (loud -> quiet, an
    easy -> hard proxy). Defaults give a readable 3x3 grid rather than a dense notebook dump."""
    # Pick n_geom geometries evenly spanning the test set
    n_test = len(test_r_indices)
    if n_test <= n_geom:
        indices = list(test_r_indices)
    else:
        step = (n_test - 1) / (n_geom - 1)
        indices = [test_r_indices[int(round(i * step))] for i in range(n_geom)]

    fig, axes = plt.subplots(len(indices), n_obs,
                             figsize=grid_figsize(n_obs, len(indices), aspect=0.72),
                             squeeze=False)

    for row, r_idx in enumerate(indices):
        R = float(handle.R_values[r_idx])
        D_model = _predict_at(handle, model, r_idx, handle.times, device)
        D_exact = handle.expectations[r_idx].T  # (K, N_t)

        # Pick channels spanning the signal-variance range (loud -> quiet), an easy -> hard
        # proxy, after dropping dead channels -- not the n_obs loudest, which look alike.
        variances = np.var(D_exact, axis=1)
        order = np.argsort(variances)[::-1]
        order = order[variances[order] > 1e-6 * variances.max()]
        if len(order) >= n_obs:
            pick = np.linspace(0, len(order) - 1, n_obs).round().astype(int)
            top_obs = order[pick]
        else:
            top_obs = order[:n_obs]

        for col, obs_idx in enumerate(top_obs):
            ax = axes[row, col]
            t = handle.times
            ax.plot(t, D_exact[obs_idx], "b-", alpha=0.7, linewidth=0.8, label="Exact")
            ax.plot(t, D_model[obs_idx], "r-", alpha=0.6, linewidth=0.8, label="Model")
            key = handle.observable_keys[int(obs_idx)]
            if hasattr(key, "item"):       # unwrap numpy scalars so titles read "obs 13", not "np.int32(13)"
                key = key.item()
            ax.set_title(rf"$R={R:.2f}$, obs {tex_escape(key)}")
            if row == len(indices) - 1:
                ax.set_xlabel(r"$t$ (a.u.)")
            if col == 0:
                ax.set_ylabel(r"$\langle\Gamma\rangle$")
                ax.legend()

    fig.tight_layout()
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


def plot_chan_pipeline(handle, model, test_r_indices, device, save_dir, ljung_box_p=0.06,
                       colorblind=False):
    """Chan et al. Fig 2-style plots: D matrix, covariance C, and spectrum.

    For each test geometry, produces a 2×3 figure:
      Row 1 (exact):  D^T heatmap | C = D^T D heatmap | I(E) spectrum
      Row 2 (model):  D^T heatmap | C = D^T D heatmap | I(E) spectrum
    """
    from fermionic_pipeline.inference.spectral_analysis import _ljung_box_screen

    for r_idx in test_r_indices:
        R = float(handle.R_values[r_idx])
        D_model = _predict_at(handle, model, r_idx, handle.times, device)
        D_exact = handle.expectations[r_idx].T  # (K, N_t)

        fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.52))

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
                D_std, aspect="auto", cmap=diverging_cmap(colorblind), vmin=-vmax, vmax=vmax,
                extent=[0, len(handle.times), D_std.shape[0], 0],
            )
            ax_d.set_xlabel(r"time index $n$")
            ax_d.set_ylabel(r"observable index $k$")
            ax_d.set_title(rf"{label}: data matrix $\mathbf{{D}}$")
            plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)

            # (b) Covariance C = D^T D
            ax_c = axes[row, 1]
            C = D_screened.T @ D_screened
            vmax_c = np.percentile(np.abs(C), 99)
            im_c = ax_c.imshow(
                C, aspect="auto", cmap=diverging_cmap(colorblind), vmin=-vmax_c, vmax=vmax_c,
                extent=[0, len(handle.times), len(handle.times), 0],
            )
            ax_c.set_xlabel(r"time index $n$")
            ax_c.set_ylabel(r"time index $n$")
            n_kept = D_screened.shape[0]
            ax_c.set_title(
                rf"{label}: $\mathbf{{C}} = \mathbf{{D}}^T\mathbf{{D}}$ "
                rf"({n_kept}/{D_std.shape[0]} obs)"
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
            ax_s.set_xlabel(r"$E$ ($E_h$)")
            ax_s.set_ylabel(r"$I(E)$")
            ax_s.set_title(rf"{label}: shadow spectrum")

            # Exact energy gaps as dashed lines
            eigvals = handle.eigvals[r_idx]
            gaps = eigvals[1:] - eigvals[0]
            omega_max = min(10.0, omega[-1])
            for g in gaps:
                if 0 < g < omega_max:
                    ax_s.axvline(g, color="green", alpha=0.4, linewidth=0.7, linestyle="--")
            ax_s.set_xlim(0, omega_max)

        fig.suptitle(
            rf"Chan et al.\ pipeline --- $R = {R:.2f}$\,\AA\ (H4, 8 qubits)",
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = os.path.join(save_dir, f"chan_pipeline_R{R:.2f}.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"[done] {path}")


@torch.no_grad()
def plot_coherence_heatmap(handle, model, test_r_indices, device, save_dir, window=20,
                           train_R_range=None, train_t_range=None, colorblind=False):
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
        D_model = _predict_at(handle, model, r_idx, times, device)  # (K, N_t)
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

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, DOUBLE_COL * 0.5))
    im = ax.imshow(
        corr_map, aspect="auto", cmap=quality_cmap(colorblind), vmin=-0.2, vmax=1.0,
        extent=[t_centers[0], t_centers[-1], Rs[-1], Rs[0]],
        interpolation="nearest",
    )
    ax.set_xlabel(r"$t$ (a.u.)")
    ax.set_ylabel("$R$ (Å)")
    ax.set_title(r"Windowed Pearson $r(R, t)$ — model vs. exact observables")
    plt.colorbar(im, ax=ax, label=r"Pearson $r$")

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
                f"in-box  $\\bar{{r}}$ ={np.nanmean(inbox):.2f}     "
                f"out-of-box  $\\bar{{r}}$ ={np.nanmean(corr_map[omask]):.2f}",
                transform=ax.transAxes, color="#0b3dff",
                bbox=dict(boxstyle="round", fc="white", ec="#0b3dff", alpha=0.85))
    # Expand limits so the box is fully visible even when eval ⊆ training (now)
    # or eval ⊋ training (after the extrapolation runs). y-axis is inverted.
    ax.set_xlim(min(t_centers[0], t_lo), max(t_centers[-1], t_hi))
    ax.set_ylim(max(Rs[-1], R_hi), min(Rs[0], R_lo))
    ax.legend(loc="upper right")

    fig.tight_layout()
    path = os.path.join(save_dir, "coherence_heatmap.pdf")
    fig.savefig(path)
    plt.close(fig)
    # Also dump the raw r(R,t) grid + axes, for downstream use such as the
    # cross-molecule averaged coherence heatmap (which normalizes R by R_eq and
    # t by the horizon, then averages the grids).
    np.savez(
        os.path.join(save_dir, "coherence_grid.npz"),
        corr=corr_map, R=np.array(Rs), t=t_centers,
        train_R_range=np.array(train_R_range), train_t_range=np.array(train_t_range),
    )
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
    parser.add_argument("--colorblind", action="store_true",
                        help="use colourblind-safe palette + colour maps "
                             "(Okabe-Ito lines; cividis instead of red-green).")
    args = parser.parse_args()

    # Re-apply the style with the requested colour mode (module import applied
    # the default); the cmap helpers read this when each figure is drawn.
    apply_nature_style(colorblind=args.colorblind)

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
    plot_chan_pipeline(handle, model, test_r_indices, device, args.save_dir, args.ljung_box_p,
                       colorblind=args.colorblind)
    plot_coherence_heatmap(handle, model, test_r_indices, device, args.save_dir,
                           colorblind=args.colorblind)


if __name__ == "__main__":
    main()
