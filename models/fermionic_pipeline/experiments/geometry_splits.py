"""Generate pre-registered sparse, coarse, and blocked geometry splits.

The manifest files are consumed directly by ``regressor_trainer`` via
``--train_r_indices_file``.  No test geometry influences split construction.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.experiments.common import write_json


def evenly_spaced_indices(n, count):
    if count < 2 or count >= n:
        raise ValueError(f"count must be in [2, {n - 1}]")
    idx = np.rint(np.linspace(0, n - 1, count)).astype(int)
    return np.unique(idx)


def make_manifests(R, counts, spacings, blocks, block_centers):
    R = np.asarray(R, dtype=float)
    all_idx = np.arange(len(R))
    out = []
    for count in counts:
        train = evenly_spaced_indices(len(R), count)
        out.append((f"count_{len(train):03d}", train, np.setdiff1d(all_idx, train), {
            "kind": "learning_curve", "n_train": int(len(train)),
        }))
    base_step = float(np.median(np.diff(R)))
    for spacing in spacings:
        stride = max(1, int(round(spacing / base_step)))
        train = np.arange(0, len(R), stride, dtype=int)
        if train[-1] != len(R) - 1:
            train = np.r_[train, len(R) - 1]
        out.append((f"coarse_{spacing:.3f}".replace(".", "p"), train,
                    np.setdiff1d(all_idx, train), {
            "kind": "coarse_to_fine", "requested_spacing": spacing,
            "effective_spacing": stride * base_step,
        }))
    if not block_centers:
        block_centers = [float(np.median(R))]
    for center in block_centers:
        for width in blocks:
            test = np.flatnonzero(np.abs(R - center) <= width / 2 + 1e-12)
            train = np.setdiff1d(all_idx, test)
            if len(test) == 0 or len(train) < 2:
                continue
            tag = f"block_c{center:.3f}_w{width:.3f}".replace(".", "p")
            out.append((tag, train, test, {
                "kind": "blocked_holdout", "center": center, "width": width,
                "actual_interval": [float(R[test].min()), float(R[test].max())],
            }))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--counts", type=int, nargs="+", default=[5, 10, 20, 40, 80])
    ap.add_argument("--spacings", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--block_widths", type=float, nargs="+", default=[0.10, 0.20, 0.30])
    ap.add_argument("--block_centers", type=float, nargs="*", default=[])
    args = ap.parse_args()
    h = RegressionDatasetHandle(args.data_path)
    root = Path(args.output_dir)
    index = []
    for tag, train, test, meta in make_manifests(
        h.R_values, args.counts, args.spacings, args.block_widths, args.block_centers,
    ):
        payload = {
            "tag": tag, "data_path": args.data_path,
            "train": train.tolist(), "test": test.tolist(),
            "train_R": h.R_values[train].tolist(), "test_R": h.R_values[test].tolist(),
            **meta,
        }
        path = root / f"{tag}.json"
        write_json(path, payload)
        index.append({"tag": tag, "path": str(path), **meta,
                      "n_train": len(train), "n_test": len(test)})
    write_json(root / "index.json", index)
    print(f"[done] {len(index)} geometry splits -> {root}")


if __name__ == "__main__":
    main()
