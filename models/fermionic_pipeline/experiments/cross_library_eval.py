"""Evaluate one QA-trained FSR against QA, independent QB, and exact targets."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.eval.omega_source import OmegaOpSource
from fermionic_pipeline.eval.regressor_eval import initial_values_for, predict_signal_matrix
from fermionic_pipeline.experiments.common import temporal_pearsons, write_json
from fermionic_pipeline.experiments.spectral_metrics import compare_spectra, aggregate
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model


def index_by_R(handle):
    return {round(float(R), 8): i for i, R in enumerate(handle.R_values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--qb", required=True)
    ap.add_argument("--exact", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    handles = {k: RegressionDatasetHandle(v) for k, v in
               {"qa": args.qa, "qb": args.qb, "exact": args.exact}.items()}
    qa = handles["qa"]
    maps = {k: index_by_R(h) for k, h in handles.items()}
    model, payload = load_checkpoint_model(args.checkpoint, device=device)
    src = OmegaOpSource("train-interp", handle=qa, payload=payload)
    test = payload.get("test_r_indices", range(len(qa.R_values)))
    rows = {k: [] for k in handles}
    target_pairwise = []
    for i in test:
        R = float(qa.R_values[i]); key = round(R, 8)
        orb = qa.hf_orbital_energies[i] if qa.hf_orbital_energies is not None else None
        d0 = initial_values_for(qa, model)
        pred = predict_signal_matrix(model, R, qa.times, device, orb, src.value(r_idx=i), d0)
        refs = {}
        for name, h in handles.items():
            j = maps[name][key]
            if not np.allclose(h.times, qa.times):
                raise ValueError(f"time grid mismatch for {name}")
            ref = h.expectations[j].T; refs[name] = ref
            row = compare_spectra(pred, ref, qa.times, tolerance=args.tolerance)
            row.update({"R": R, "temporal_pearson": float(np.nanmean(temporal_pearsons(pred, ref))),
                        "mse": float(np.mean((pred - ref) ** 2))})
            rows[name].append(row)
        target_pairwise.append({
            "R": R,
            "qa_qb_mse": float(np.mean((refs["qa"] - refs["qb"]) ** 2)),
            "qa_exact_mse": float(np.mean((refs["qa"] - refs["exact"]) ** 2)),
            "qb_exact_mse": float(np.mean((refs["qb"] - refs["exact"]) ** 2)),
        })
        print(f"[R={R:.4f}] QA={rows['qa'][-1]['temporal_pearson']:.3f} "
              f"QB={rows['qb'][-1]['temporal_pearson']:.3f} "
              f"exact={rows['exact'][-1]['temporal_pearson']:.3f}", flush=True)
    result = {"checkpoint": args.checkpoint, "datasets": {"qa": args.qa, "qb": args.qb,
              "exact": args.exact}, "results": rows,
              "aggregate": {k: aggregate(v) for k, v in rows.items()},
              "target_pairwise": target_pairwise}
    write_json(args.output, result)
    print(f"[done] cross-library evaluation -> {args.output}")


if __name__ == "__main__":
    main()
