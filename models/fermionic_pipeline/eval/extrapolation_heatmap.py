"""Coherence heatmap on an EXTRAPOLATION grid.

Plots windowed Pearson r(R, t) over a dataset whose (R, t) extent exceeds the
training extent, and draws the training-data bounding box at the sub-region the
model was actually trained on. Inside the box is interpolation (Prop 1); outside
is extrapolation, where Prop 2's beyond-horizon / high-frequency limits predict
degradation. Pair with a dataset built by slurm/regression_extrap.sh.

Usage:
    python3 -m fermionic_pipeline.eval.extrapolation_heatmap \
        --data_path <extrap.h5> --checkpoint <v18-orb>/regressor.pt \
        --save_dir <dir> --train_r_range 0.5 3.0 --train_t_range 0 300 --device cuda
"""
from __future__ import annotations

import argparse
import os

import torch

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model
from fermionic_pipeline.eval.plot_regression import plot_coherence_heatmap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--train_r_range", type=float, nargs=2, default=[0.5, 3.0],
                    help="R bounds the model was trained on (box edges)")
    ap.add_argument("--train_t_range", type=float, nargs=2, default=[0.0, 300.0],
                    help="t bounds the model was trained on (box edges)")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--omega_op_source", type=str, default="dataset",
                    choices=["dataset", "train-interp"],
                    help="train-interp: non-oracle omega_op interpolated from the "
                         "TRAINING dataset's training geometries (requires "
                         "--train_data_path; out-of-range R is clamped to edge values).")
    ap.add_argument("--train_data_path", type=str, default=None,
                    help="The dataset the checkpoint was trained on; required for "
                         "--omega_op_source train-interp on an extended grid.")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    handle = RegressionDatasetHandle(args.data_path)
    model, payload = load_checkpoint_model(args.checkpoint, device=device)

    if args.omega_op_source == "train-interp":
        if args.train_data_path is None:
            raise SystemExit("--omega_op_source train-interp requires --train_data_path")
        from fermionic_pipeline.eval.omega_source import OmegaOpSource
        from fermionic_pipeline.eval import plot_regression as _pr
        train_handle = RegressionDatasetHandle(args.train_data_path)
        _pr.set_omega_source(OmegaOpSource(
            "train-interp", handle=handle, payload=payload, train_handle=train_handle))
        print("[info] omega_op source: train-interp (non-oracle, from training dataset)")
    # plot EVERY geometry in the (extended) grid, not a held-out test split
    all_idx = list(range(len(handle.R_values)))
    print(f"[info] extrapolation heatmap: {len(all_idx)} geometries, "
          f"R {float(handle.R_values.min()):.2f}-{float(handle.R_values.max()):.2f}, "
          f"t {float(handle.times.min()):.0f}-{float(handle.times.max()):.0f}; "
          f"train box R{tuple(args.train_r_range)} t{tuple(args.train_t_range)}")
    plot_coherence_heatmap(
        handle, model, all_idx, device, args.save_dir, window=args.window,
        train_R_range=tuple(args.train_r_range),
        train_t_range=tuple(args.train_t_range),
    )


if __name__ == "__main__":
    main()
