"""End-to-end spectrum metrics for FSR checkpoints or reconstructed traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.eval.omega_source import OmegaOpSource
from fermionic_pipeline.eval.regressor_eval import initial_values_for, predict_signal_matrix
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model
from fermionic_pipeline.experiments.common import temporal_pearsons, write_json


def normalized_spectrum(D, times):
    # Chan's eigendecomposition of C=D^T D is exactly the right-singular
    # subspace of standardized D.  SVD avoids materializing an N_T x N_T
    # covariance (6001^2 entries on the H4 grid).
    centered = D - D.mean(axis=1, keepdims=True)
    scale = D.std(axis=1, ddof=1, keepdims=True)
    good = scale[:, 0] > 1e-12
    standardized = centered[good] / scale[good]
    if not len(standardized):
        omega = 2 * np.pi * np.fft.rfftfreq(len(times), d=times[1] - times[0])
        return omega, np.zeros_like(omega)
    _, _, vh = np.linalg.svd(standardized, full_matrices=False)
    r = min(10, len(vh))
    F = np.fft.rfft(vh[:r] * np.hanning(len(times)), axis=1)
    omega = 2 * np.pi * np.fft.rfftfreq(len(times), d=times[1] - times[0])
    power = np.sum(np.abs(F) ** 2, axis=0).real
    power = np.maximum(power, 0)
    return omega, power / max(float(power.sum()), 1e-30)


def peak_table(omega, power, min_height_frac=0.01, max_peaks=30):
    idx, props = find_peaks(power, height=min_height_frac * power.max())
    if not len(idx):
        return np.array([]), np.array([])
    order = np.argsort(props["peak_heights"])[::-1][:max_peaks]
    heights = props["peak_heights"][order]
    heights = heights / max(float(heights.sum()), 1e-30)
    return omega[idx[order]], heights


def match_peaks(ref_w, ref_h, pred_w, pred_h, tolerance):
    if not len(ref_w) or not len(pred_w):
        return [], 0.0, 0.0
    cost = np.abs(ref_w[:, None] - pred_w[None, :])
    ri, pi = linear_sum_assignment(cost)
    matches = []
    for r, p in zip(ri, pi):
        if cost[r, p] <= tolerance:
            matches.append((int(r), int(p), float(cost[r, p])))
    recall = len(matches) / len(ref_w)
    precision = len(matches) / len(pred_w)
    return matches, precision, recall


def compare_spectra(D_pred, D_ref, times, tolerance=0.05, dominant_fraction=0.10):
    wp, pp = normalized_spectrum(D_pred, times)
    wr, pr = normalized_spectrum(D_ref, times)
    rw, rh = peak_table(wr, pr)
    pw, ph = peak_table(wp, pp)
    matches, precision, recall = match_peaks(rw, rh, pw, ph, tolerance)
    dominant = rh >= dominant_fraction * rh.max() if len(rh) else np.array([], dtype=bool)
    dom_matches = [m for m in matches if dominant[m[0]]]
    weak_matches = [m for m in matches if not dominant[m[0]]]
    intensity_err = [abs(ph[p] - rh[r]) for r, p, _ in matches]
    return {
        "dominant_peak_error": float(abs(pw[0] - rw[0])) if len(pw) and len(rw) else None,
        "precision": precision, "recall": recall,
        "dominant_recall": len(dom_matches) / max(int(dominant.sum()), 1),
        "weak_recall": len(weak_matches) / max(int((~dominant).sum()), 1) if len(dominant) else 0.0,
        "matched_energy_mae": float(np.mean([m[2] for m in matches])) if matches else None,
        "matched_intensity_mae": float(np.mean(intensity_err)) if intensity_err else None,
        "spectral_wasserstein": float(wasserstein_distance(wr, wp, u_weights=pr, v_weights=pp)),
        "n_reference_peaks": len(rw), "n_predicted_peaks": len(pw),
        "n_matches": len(matches), "tolerance": tolerance,
    }


def aggregate(rows):
    keys = [
        "temporal_pearson", "mse", "dominant_peak_error", "precision", "recall",
        "dominant_recall", "weak_recall", "matched_energy_mae",
        "matched_intensity_mae", "spectral_wasserstein",
    ]
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "median": float(np.median(vals))}
    return out


def evaluate_checkpoint(data_path, checkpoint, device, indices=None, tolerance=0.05,
                        omega_mode="train-interp"):
    h = RegressionDatasetHandle(data_path)
    model, payload = load_checkpoint_model(checkpoint, device=device)
    if indices is None:
        indices = payload.get("test_r_indices", range(len(h.R_values)))
    src = OmegaOpSource(omega_mode, handle=h, payload=payload)
    rows = []
    for i in indices:
        R = float(h.R_values[i])
        orb = h.hf_orbital_energies[i] if h.hf_orbital_energies is not None else None
        d0 = initial_values_for(h, model)
        Dp = predict_signal_matrix(model, R, h.times, device, orb, src.value(r_idx=i), d0)
        Dr = h.expectations[i].T
        row = compare_spectra(Dp, Dr, h.times, tolerance=tolerance)
        row.update({"R": R, "r_idx": int(i),
                    "temporal_pearson": float(np.nanmean(temporal_pearsons(Dp, Dr))),
                    "mse": float(np.mean((Dp - Dr) ** 2))})
        rows.append(row)
        print(f"[R={R:.4f}] r={row['temporal_pearson']:.3f} "
              f"recall={row['recall']:.3f} W1={row['spectral_wasserstein']:.4g}", flush=True)
    return {"data_path": data_path, "checkpoint": checkpoint,
            "omega_op_source": omega_mode, "results": rows, "aggregate": aggregate(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--all_geometries", action="store_true")
    ap.add_argument("--omega_op_source", choices=["dataset", "train-interp"], default="train-interp")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    indices = range(len(RegressionDatasetHandle(args.data_path).R_values)) if args.all_geometries else None
    result = evaluate_checkpoint(args.data_path, args.checkpoint, device, indices,
                                 args.tolerance, args.omega_op_source)
    write_json(args.output, result)
    print(f"[done] spectral metrics -> {args.output}")


if __name__ == "__main__":
    main()
