"""
Per-observable composition diagnostic — distinguishes bandwidth failure from
composition failure at the level of the 120 observable trajectories.

Bandwidth failure: aggregate spectrum is truncated below where the signal lives.
Composition failure: aggregate spectrum is right, but per-observable phases /
mixing weights are scrambled across the 120 outputs.

Per-R metrics:
  envelope_pearson — pearson of Σ_obs |Y(ω)|² model vs exact.
                     High (≥0.95) ⇒ aggregate spectral support is correct.
  phase_err_mean   — amplitude-weighted mean |arg(Y_m[k, ω*]) − arg(Y_e[k, ω*])|
                     at each observable's dominant frequency, averaged over k.
                     High ⇒ per-observable phase scrambled even when peaks are right.
  amp_ratio_std    — std of |Y_m[k, ω*]| / |Y_e[k, ω*]| across observables.
                     High ⇒ inconsistent per-observable mixing weights.

Reading rule:
  envelope high + phase_err high + time pearson low → composition failure
  envelope low                                       → bandwidth failure
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.eval.regressor_eval import predict_signal_matrix
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model


ZONES = [
    (0.0, 0.74, "aliased <0.74"),
    (0.74, 1.0, "borderline [0.74, 1.0)"),
    (1.0, 1.5, "[1.0, 1.5)"),
    (1.5, 2.0, "[1.5, 2.0)"),
    (2.0, 3.1, "[2.0, 3.0]"),
]


def composition_metrics(D_model: np.ndarray, D_exact: np.ndarray, times: np.ndarray,
                        amp_floor_frac: float = 0.05) -> dict:
    """Compute the three composition metrics for one geometry.

    D_model, D_exact: (K, N_t) arrays; times: (N_t,)
    amp_floor_frac: ignore observables whose dominant peak amplitude is below
                    this fraction of the bank's max — they carry no signal.
    """
    K, N_t = D_exact.shape

    # Mean-subtract per observable, then Hann-window
    D_m = D_model - D_model.mean(axis=1, keepdims=True)
    D_e = D_exact - D_exact.mean(axis=1, keepdims=True)
    w = np.hanning(N_t)
    Y_m = np.fft.rfft(D_m * w, axis=1)  # (K, F)
    Y_e = np.fft.rfft(D_e * w, axis=1)

    # ---- envelope_pearson: aggregate spectral mass ----
    env_m = (np.abs(Y_m) ** 2).sum(axis=0)  # (F,)
    env_e = (np.abs(Y_e) ** 2).sum(axis=0)
    if env_m.std() > 1e-12 and env_e.std() > 1e-12:
        envelope_pearson = float(np.corrcoef(env_m, env_e)[0, 1])
    else:
        envelope_pearson = float("nan")

    # ---- per-observable dominant-frequency analysis ----
    # For each observable, find the bin with max |Y_e| (skip DC bin 0).
    abs_Ye = np.abs(Y_e)
    abs_Ye[:, 0] = 0  # zero out DC
    w_dom = np.argmax(abs_Ye, axis=1)              # (K,)
    e_dom_amp = abs_Ye[np.arange(K), w_dom]        # (K,)

    # Amplitude floor: observables whose dominant peak is too small are noise-dominated.
    threshold = amp_floor_frac * abs_Ye.max()
    keep = e_dom_amp > threshold
    n_kept = int(keep.sum())

    if n_kept == 0:
        return {
            "envelope_pearson": envelope_pearson,
            "phase_err_mean": float("nan"),
            "amp_ratio_std": float("nan"),
            "n_observables_kept": 0,
            "n_observables_total": K,
        }

    # Phase error at each observable's dominant frequency
    Y_m_dom = Y_m[np.arange(K), w_dom][keep]
    Y_e_dom = Y_e[np.arange(K), w_dom][keep]
    phase_diff = np.angle(Y_m_dom * np.conj(Y_e_dom))  # (-π, π]
    abs_phase_err = np.abs(phase_diff)                 # (kept,)
    weights = np.abs(Y_e_dom)
    phase_err_mean = float(np.average(abs_phase_err, weights=weights))

    # Amplitude ratio scatter
    m_dom_amp = np.abs(Y_m_dom)
    e_dom_amp_kept = np.abs(Y_e_dom)
    amp_ratio = m_dom_amp / np.maximum(e_dom_amp_kept, 1e-12)
    amp_ratio_std = float(np.std(amp_ratio))

    return {
        "envelope_pearson": envelope_pearson,
        "phase_err_mean": phase_err_mean,         # radians
        "amp_ratio_std": amp_ratio_std,
        "n_observables_kept": n_kept,
        "n_observables_total": K,
    }


def stratify(results: list, key: str) -> list:
    out = []
    Rs = np.array([r["R"] for r in results])
    vals = np.array([r[key] for r in results], dtype=float)
    for lo, hi, name in ZONES:
        m = (Rs >= lo) & (Rs < hi)
        if m.sum() == 0:
            continue
        finite = m & np.isfinite(vals)
        if finite.sum() == 0:
            continue
        out.append({
            "zone": name,
            "n": int(finite.sum()),
            "mean": float(np.mean(vals[finite])),
            "median": float(np.median(vals[finite])),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description="Per-observable composition diagnostic")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test_r_indices", type=int, nargs="+", default=None,
                        help="Override; default uses payload's test_r_indices.")
    parser.add_argument("--amp_floor_frac", type=float, default=0.05,
                        help="Ignore observables whose dominant peak < frac × bank-max.")
    parser.add_argument("--omega_op_source", type=str, default="dataset",
                        choices=["dataset", "train-interp"],
                        help="train-interp: non-oracle omega_op interpolated from "
                             "the checkpoint's training geometries only.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    handle = RegressionDatasetHandle(args.data_path)
    model, payload = load_checkpoint_model(args.checkpoint, device=device)

    omega_src = None
    if args.omega_op_source == "train-interp":
        from fermionic_pipeline.eval.omega_source import OmegaOpSource
        omega_src = OmegaOpSource("train-interp", handle=handle, payload=payload)
        print("[info] omega_op source: train-interp (non-oracle)")

    if args.test_r_indices is not None:
        test_r_indices = args.test_r_indices
    else:
        test_r_indices = payload.get("test_r_indices", list(range(len(handle.R_values))))

    results = []
    for r_idx in test_r_indices:
        R = float(handle.R_values[r_idx])
        orb_e = handle.hf_orbital_energies[r_idx] if handle.hf_orbital_energies is not None else None
        if omega_src is not None:
            omega_op = omega_src.value(r_idx=r_idx)
        else:
            omega_op = float(handle.omega_op[r_idx]) if handle.omega_op is not None else None

        D_model = predict_signal_matrix(model, R, handle.times, device,
                                        orb_energies=orb_e, omega_op=omega_op)
        D_exact = handle.expectations[r_idx].T  # (K, N_t)

        m = composition_metrics(D_model, D_exact, handle.times, amp_floor_frac=args.amp_floor_frac)
        m["R"] = R
        results.append(m)
        print(
            f"[R={R:.3f}] envelope_pearson={m['envelope_pearson']:.3f}  "
            f"phase_err={m['phase_err_mean']:.3f} rad  "
            f"amp_ratio_std={m['amp_ratio_std']:.3f}  "
            f"(kept {m['n_observables_kept']}/{m['n_observables_total']})",
            flush=True,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "results": results,
        "stratified": {
            "envelope_pearson":  stratify(results, "envelope_pearson"),
            "phase_err_mean":    stratify(results, "phase_err_mean"),
            "amp_ratio_std":     stratify(results, "amp_ratio_std"),
        },
    }
    out_path = os.path.join(args.save_dir, "composition_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] composition diagnostic -> {out_path}")

    print("\n=== Per-zone summary ===")
    for metric, units in [("envelope_pearson", ""), ("phase_err_mean", " rad"), ("amp_ratio_std", "")]:
        print(f"\n  {metric}:")
        for row in summary["stratified"][metric]:
            print(f"    {row['zone']:<24} n={row['n']:2d}  mean={row['mean']:.3f}{units}  med={row['median']:.3f}{units}")


if __name__ == "__main__":
    main()
