"""
Spectral evaluation for the observable regressor.

Builds the signal matrix D directly from model predictions (no shadow
sampling), then runs Chan et al. spectral analysis. Also compares
against exact D matrix entries from the dataset.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.inference.spectral_analysis import extract_peaks, spectral_analysis
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model


@torch.no_grad()
def predict_signal_matrix(model, R, times, device, orb_energies=None, omega_op=None):
    """Build D matrix directly from model predictions."""
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
    pred = model(rt_tensor, orb_energies=orb_e, omega_op=omega_op_t).cpu().numpy()  # (N_t, K)
    return pred.T  # (K, N_t)


def evaluate_geometry(model, handle, r_idx, ljung_box_p, n_peaks, device):
    R = float(handle.R_values[r_idx])
    orb_e = handle.hf_orbital_energies[r_idx] if handle.hf_orbital_energies is not None else None
    omega_op = float(handle.omega_op[r_idx]) if handle.omega_op is not None else None

    D_model = predict_signal_matrix(model, R, handle.times, device, orb_energies=orb_e, omega_op=omega_op)
    D_exact = handle.expectations[r_idx].T  # (K, N_t)

    # Per-observable metrics
    pearsons, range_ratios = [], []
    for i in range(D_exact.shape[0]):
        std_e, std_m = np.std(D_exact[i]), np.std(D_model[i])
        if std_e < 1e-12 or std_m < 1e-12:
            pearsons.append(np.nan)
            range_ratios.append(np.nan)
            continue
        pearsons.append(float(np.corrcoef(D_exact[i], D_model[i])[0, 1]))
        range_ratios.append(float(std_m / std_e))
    pearsons = np.array(pearsons)
    range_ratios = np.array(range_ratios)

    # Spectral analysis: model predictions
    omega_m, spec_m, _ = spectral_analysis(D_model, handle.times, ljung_box_p=ljung_box_p)
    peaks_m, heights_m = extract_peaks(omega_m, spec_m, n_peaks=n_peaks)

    # Spectral analysis: exact targets (upper bound)
    omega_e, spec_e, _ = spectral_analysis(D_exact, handle.times, ljung_box_p=ljung_box_p)
    peaks_e, heights_e = extract_peaks(omega_e, spec_e, n_peaks=n_peaks)

    eigvals = handle.eigvals[r_idx]
    gaps = (eigvals[1:] - eigvals[0]).tolist()

    return {
        "R": R,
        "mse": float(np.mean((D_model - D_exact) ** 2)),
        "pearson_mean": float(np.nanmean(pearsons)),
        "pearson_median": float(np.nanmedian(pearsons)),
        "range_ratio_mean": float(np.nanmean(range_ratios)),
        "range_ratio_median": float(np.nanmedian(range_ratios)),
        "per_observable_pearson": pearsons.tolist(),
        "per_observable_range_ratio": range_ratios.tolist(),
        "model_peaks": peaks_m.tolist(),
        "model_heights": heights_m.tolist(),
        "exact_peaks": peaks_e.tolist(),
        "exact_heights": heights_e.tolist(),
        "exact_gaps": gaps,
        "n_observables": int(D_exact.shape[0]),
        "n_times": int(D_exact.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Spectral eval for observable regressor")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ljung_box_p", type=float, default=0.06)
    parser.add_argument("--n_peaks", type=int, default=10)
    parser.add_argument("--test_r_indices", type=int, nargs="+", default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    handle = RegressionDatasetHandle(args.data_path)
    model, payload = load_checkpoint_model(args.checkpoint, device=device)

    if args.test_r_indices is not None:
        test_r_indices = args.test_r_indices
    else:
        test_r_indices = payload.get("test_r_indices", list(range(len(handle.R_values))))

    results = []
    for r_idx in test_r_indices:
        print(f"[eval] R={handle.R_values[r_idx]:.4f}", flush=True)
        result = evaluate_geometry(
            model, handle, r_idx, args.ljung_box_p, args.n_peaks, device,
        )
        results.append(result)
        print(
            f"  mse={result['mse']:.6f} pearson={result['pearson_mean']:.4f} "
            f"model_peaks={result['model_peaks'][:5]} "
            f"exact_peaks={result['exact_peaks'][:5]}",
            flush=True,
        )

    summary = {"checkpoint": args.checkpoint, "data_path": args.data_path, "results": results}
    out_path = os.path.join(args.save_dir, "regressor_eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] regressor evaluation -> {out_path}")


if __name__ == "__main__":
    main()
