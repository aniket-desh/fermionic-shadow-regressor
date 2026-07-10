"""Construct a time-grid screen using training geometries only."""
from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy.signal import find_peaks

from fermionic_pipeline.data.compute_omega_op import compute_omega_op
from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.experiments.common import write_json


def codominant_spacing(y, times, height_fraction=0.10):
    y0 = y - y.mean(axis=1, keepdims=True)
    Y = np.fft.rfft(y0 * np.hanning(len(times))[None, :, None], axis=1)
    P = np.sum(np.abs(Y) ** 2, axis=2)
    omega = 2 * np.pi * np.fft.rfftfreq(len(times), d=times[1] - times[0])
    spacings = []
    for row in P:
        peaks, props = find_peaks(row, height=height_fraction * row.max())
        w = np.sort(omega[peaks])
        if len(w) > 1:
            spacings.append(float(np.min(np.diff(w))))
    return min(spacings) if spacings else 2 * np.pi / (times[-1] - times[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint", required=True, help="Supplies training geometry indices only")
    ap.add_argument("--output", required=True)
    ap.add_argument("--safety", type=float, default=1.10)
    args = ap.parse_args()
    h = RegressionDatasetHandle(args.data_path)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train = np.asarray(payload["train_r_indices"], dtype=int)
    test = np.asarray(payload["test_r_indices"], dtype=int)
    y_train = h.expectations[train]
    ceilings = compute_omega_op(y_train, h.times)
    omega_global = args.safety * float(np.max(ceilings))
    dt_safe = np.pi / omega_global
    dt_data = float(h.times[1] - h.times[0])
    stride = max(1, int(np.floor(dt_safe / dt_data)))
    spacing = codominant_spacing(y_train, h.times)
    horizon = 2 * np.pi / spacing
    used_horizon = min(float(h.times[-1]), horizon)
    # Audit whether any held-out exact target exceeds the train-only ceiling.
    heldout_ceiling = compute_omega_op(h.expectations[test], h.times)
    result = {
        "data_path": args.data_path, "checkpoint": args.checkpoint,
        "n_train": len(train), "n_test": len(test), "safety": args.safety,
        "train_omega_max": float(np.max(ceilings)), "global_ceiling": omega_global,
        "recommended_dt": dt_safe, "source_dt": dt_data, "train_t_stride": stride,
        "codominant_spacing": spacing, "recommended_horizon": horizon,
        "used_horizon": used_horizon,
        "horizon_clamped_to_available": bool(horizon > h.times[-1]),
        "heldout_ceiling_max": float(np.max(heldout_ceiling)),
        "heldout_bandwidth_violations": int(np.sum(heldout_ceiling > omega_global)),
    }
    write_json(args.output, result)
    print(f"[done] train-only screen -> {args.output}")


if __name__ == "__main__":
    main()
