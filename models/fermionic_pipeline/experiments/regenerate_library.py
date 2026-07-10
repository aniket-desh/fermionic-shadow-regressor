"""Regenerate a finite-Q target HDF5 on exactly the grid of a reference file."""
from __future__ import annotations

import argparse
import json

import h5py

from fermionic_pipeline.data.regression_dataset import (
    RegressionDatasetConfig, generate_regression_dataset,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_q", type=int, default=None)
    ap.add_argument("--n_workers", type=int, default=None)
    args = ap.parse_args()
    with h5py.File(args.reference, "r") as f:
        cfg = json.loads(f.attrs["config_json"])
        n_q = args.n_q or int(cfg.get("n_q", 500))
    fields = RegressionDatasetConfig.__dataclass_fields__
    cfg = {k: v for k, v in cfg.items() if k in fields}
    cfg.update(seed=args.seed, n_q=n_q)
    config = RegressionDatasetConfig(**cfg)
    generate_regression_dataset(args.output, config, n_workers=args.n_workers)
    print(f"[done] independent Q library seed={args.seed} -> {args.output}")


if __name__ == "__main__":
    main()
