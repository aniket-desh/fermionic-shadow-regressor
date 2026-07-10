"""Explicit shared-line extraction followed by geometry interpolation.

At each training geometry, dominant lines are extracted from the aggregate
multi-channel spectrum, all channels are fit by linear least squares on that
shared line bank, and frequencies/amplitudes are interpolated to held-out R.
This is the structured competitor closest to the FSR's intended mechanism.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.experiments.common import load_split, temporal_pearsons, write_json
from fermionic_pipeline.experiments.spectral_metrics import compare_spectra, aggregate


def extract_lines(D, times, n_lines):
    y = D - D.mean(axis=1, keepdims=True)
    Y = np.fft.rfft(y * np.hanning(len(times)), axis=1)
    power = np.sum(np.abs(Y) ** 2, axis=0)
    omega = 2 * np.pi * np.fft.rfftfreq(len(times), d=times[1] - times[0])
    peaks, _ = find_peaks(power)
    peaks = peaks[peaks > 0]
    order = peaks[np.argsort(power[peaks])[::-1][:n_lines]] if len(peaks) else np.array([], int)
    lines = np.sort(omega[order])
    if len(lines) < n_lines:
        candidates = np.argsort(power[1:])[::-1] + 1
        fill = [omega[i] for i in candidates if np.all(np.abs(omega[i] - lines) > 1e-10)]
        lines = np.sort(np.r_[lines, fill[:n_lines - len(lines)]])
    return lines[:n_lines]


def design(times, omega):
    return np.column_stack([np.ones(len(times)),
                            *[f(omega[None, :] * times[:, None])[:, k]
                              for f in (np.cos, np.sin) for k in range(len(omega))]])


def fit_coefficients(D, times, omega, ridge):
    X = design(times, omega)
    reg = ridge * np.eye(X.shape[1]); reg[0, 0] = 0
    return np.linalg.solve(X.T @ X + reg, X.T @ D.T)  # (1+2L, K)


def align_line_banks(banks, coeffs):
    """Track line identities sequentially by minimum-frequency assignment."""
    out_w = [banks[0]]; out_c = [coeffs[0]]
    L = len(banks[0])
    for w, c in zip(banks[1:], coeffs[1:]):
        row, col = linear_sum_assignment(np.abs(out_w[-1][:, None] - w[None, :]))
        perm = col[np.argsort(row)]
        w = w[perm]
        # coefficient layout is [dc, cos_0..cos_L-1, sin_0..sin_L-1]
        c = np.r_[c[0:1], c[1:1 + L][perm], c[1 + L:1 + 2 * L][perm]]
        out_w.append(w); out_c.append(c)
    return np.asarray(out_w), np.asarray(out_c)


def interpolate_model(R_train, banks, coeffs, R):
    w = np.array([np.interp(R, R_train, banks[:, k]) for k in range(banks.shape[1])])
    flat = coeffs.reshape(len(R_train), -1)
    c = np.array([np.interp(R, R_train, flat[:, j]) for j in range(flat.shape[1])])
    return w, c.reshape(coeffs.shape[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n_lines", type=int, default=32)
    ap.add_argument("--ridge", type=float, default=1e-6)
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()
    h = RegressionDatasetHandle(args.data_path)
    train, test, split_meta = load_split(args.split)
    order = np.argsort(h.R_values[train]); train = train[order]
    banks, coeffs = [], []
    for i in train:
        D = h.expectations[i].T
        w = extract_lines(D, h.times, args.n_lines)
        banks.append(w); coeffs.append(fit_coefficients(D, h.times, w, args.ridge))
    banks, coeffs = align_line_banks(banks, coeffs)
    rows = []
    for i in test:
        R = float(h.R_values[i])
        w, c = interpolate_model(h.R_values[train], banks, coeffs, R)
        pred = (design(h.times, w) @ c).T
        ref = h.expectations[i].T
        row = compare_spectra(pred, ref, h.times, tolerance=args.tolerance)
        row.update({"R": R, "r_idx": int(i),
                    "temporal_pearson": float(np.nanmean(temporal_pearsons(pred, ref))),
                    "mse": float(np.mean((pred - ref) ** 2))})
        rows.append(row)
        print(f"[R={R:.4f}] r={row['temporal_pearson']:.3f} recall={row['recall']:.3f}")
    result = {"method": "shared-line-extraction-plus-interpolation", "data_path": args.data_path,
              "split": split_meta, "n_lines": args.n_lines, "ridge": args.ridge,
              "results": rows, "aggregate": aggregate(rows)}
    write_json(args.output, result)
    print(f"[done] structured baseline -> {args.output}")


if __name__ == "__main__":
    main()
