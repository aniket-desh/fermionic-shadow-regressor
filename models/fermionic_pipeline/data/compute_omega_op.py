"""
Add operational ω_op(R) to an existing regression HDF5 in-place.

ω_op(R) is defined as the smallest ω capturing `frac` (default 0.99) of the
mean-subtracted, Hann-windowed cumulative spectral energy of y_true(R, t)
summed across observables. Used by the adaptive_bandwidth mode of
ObservableRegressor.

Usage:
    python3 -m fermionic_pipeline.data.compute_omega_op \
      --data_path results/fermionic_pipeline/regression/h4_regress_v10/regression_targets.h5
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np


def compute_omega_op(y, t, frac=0.99):
    """y: (n_R, n_t, K). Returns omega_op: (n_R,) in E_h."""
    n = len(t)
    dt = t[1] - t[0]
    freqs = 2 * np.pi * np.fft.rfftfreq(n, d=dt)  # angular frequency
    y0 = y - y.mean(axis=1, keepdims=True)
    hann = np.hanning(n)[None, :, None]
    Y = np.fft.rfft(y0 * hann, axis=1)
    P = (np.abs(Y) ** 2).sum(axis=2)               # (n_R, n_freq)
    omega_op = np.empty(len(P), dtype=np.float64)
    for i in range(len(P)):
        s = P[i].sum()
        if s < 1e-30:
            omega_op[i] = freqs[-1]
            continue
        cum = np.cumsum(P[i]) / s
        omega_op[i] = freqs[min(np.searchsorted(cum, frac), len(freqs) - 1)]
    return omega_op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--frac", type=float, default=0.99)
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace omega_op even if already present.")
    args = ap.parse_args()

    with h5py.File(args.data_path, "a") as f:
        if "omega_op" in f and not args.overwrite:
            print(f"[skip] omega_op already present in {args.data_path}; pass --overwrite to replace.")
            return
        if "omega_op" in f:
            del f["omega_op"]
        y = f["expectations"][...]
        t = f["times"][...]
        omega_op = compute_omega_op(y, t, frac=args.frac)
        f.create_dataset("omega_op", data=omega_op)
        f["omega_op"].attrs["frac"] = float(args.frac)
        f["omega_op"].attrs["definition"] = (
            "smallest omega capturing `frac` of cumulative |Y_true|^2 "
            "(mean-subtracted, Hann-windowed, summed over observables)"
        )

    R = f["R_values"] if False else None  # silence linter
    print(f"[saved] omega_op({len(omega_op)}) -> {args.data_path}")
    print(f"  min={omega_op.min():.3f}  max={omega_op.max():.3f}  mean={omega_op.mean():.3f}  E_h")


if __name__ == "__main__":
    main()
