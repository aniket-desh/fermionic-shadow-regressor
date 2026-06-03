"""
Standalone spectral evaluation for conditional distribution models.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

from fermionic_pipeline.data.exact_conditional_dataset import (
    ExactConditionalDatasetHandle,
    bitstrings_from_class_indices,
)
from fermionic_pipeline.inference.spectral_analysis import (
    build_signal_matrix,
    extract_peaks,
    spectral_analysis,
)
from fermionic_pipeline.training.classifier_trainer import load_checkpoint_model


def _sample_outcomes_from_probs(probabilities, permutation, signs, n_qubits, n_samples, rng):
    probabilities = probabilities / probabilities.sum()
    class_indices = rng.choice(len(probabilities), size=n_samples, p=probabilities)
    bitstrings = bitstrings_from_class_indices(class_indices, n_qubits)
    perm_list = permutation.tolist()
    sign_list = signs.tolist()
    return [[perm_list, bitstrings[i].tolist(), sign_list] for i in range(n_samples)]


def _sample_direct_outcomes(handle, r_idx, n_samples_per_q, rng):
    direct = {}
    if h5py is None:
        raise ImportError("h5py is required for spectral_eval direct sampling.")
    with h5py.File(handle.path, "r") as f:
        probs_ds = f["probabilities"]
        for t_idx, t in enumerate(handle.times):
            snapshots = []
            for q_idx in range(handle.n_q):
                probs = probs_ds[r_idx, t_idx, q_idx].astype(np.float64)
                snapshots.extend(
                    _sample_outcomes_from_probs(
                        probabilities=probs,
                        permutation=handle.permutations[q_idx],
                        signs=handle.signs[q_idx],
                        n_qubits=handle.n_qubits,
                        n_samples=n_samples_per_q,
                        rng=rng,
                    )
                )
            direct[float(t)] = snapshots
    return direct


@torch.no_grad()
def _sample_model_outcomes(model, handle, r_idx, n_samples_per_q, rng, device):
    synth = {}
    q_feat = torch.tensor(handle.q_features, dtype=torch.float32, device=device)
    for t_idx, t in enumerate(handle.times):
        rt = np.tile(np.array([[handle.R_values[r_idx], t]], dtype=np.float32), (handle.n_q, 1))
        rt = torch.tensor(rt, dtype=torch.float32, device=device)
        probs, _ = model.predict_distribution(q_feat, rt)
        probs = probs.detach().cpu().numpy()

        snapshots = []
        for q_idx in range(handle.n_q):
            snapshots.extend(
                _sample_outcomes_from_probs(
                    probabilities=probs[q_idx],
                    permutation=handle.permutations[q_idx],
                    signs=handle.signs[q_idx],
                    n_qubits=handle.n_qubits,
                    n_samples=n_samples_per_q,
                    rng=rng,
                )
            )
        synth[float(t)] = snapshots
    return synth


def _rowwise_metrics(D_direct, D_synth):
    pearsons = []
    range_ratios = []
    for i in range(D_direct.shape[0]):
        direct_row = D_direct[i]
        synth_row = D_synth[i]
        std_direct = np.std(direct_row)
        std_synth = np.std(synth_row)
        if std_direct < 1e-12 or std_synth < 1e-12:
            pearsons.append(np.nan)
            range_ratios.append(np.nan)
            continue
        pearsons.append(float(np.corrcoef(direct_row, synth_row)[0, 1]))
        range_ratios.append(float(std_synth / std_direct))
    return np.array(pearsons), np.array(range_ratios)


def evaluate_geometry(
    model,
    handle: ExactConditionalDatasetHandle,
    r_idx: int,
    n_samples_per_q: int,
    ljung_box_p: float | None,
    n_peaks: int,
    rng: np.random.Generator,
    device: torch.device,
):
    direct_outcomes = _sample_direct_outcomes(handle, r_idx, n_samples_per_q, rng)
    synth_outcomes = _sample_model_outcomes(model, handle, r_idx, n_samples_per_q, rng, device)

    D_direct, obs_keys = build_signal_matrix(direct_outcomes, handle.times, k=1)
    D_synth, _ = build_signal_matrix(
        synth_outcomes,
        handle.times,
        observable_keys=obs_keys,
        k=1,
    )

    pearsons, range_ratios = _rowwise_metrics(D_direct, D_synth)

    omega_direct, spec_direct, _ = spectral_analysis(
        D_direct, handle.times, ljung_box_p=ljung_box_p
    )
    omega_synth, spec_synth, _ = spectral_analysis(
        D_synth, handle.times, ljung_box_p=ljung_box_p
    )
    peaks_direct, heights_direct = extract_peaks(omega_direct, spec_direct, n_peaks=n_peaks)
    peaks_synth, heights_synth = extract_peaks(omega_synth, spec_synth, n_peaks=n_peaks)

    eigvals = handle.eigvals[r_idx]
    gaps = (eigvals[1:] - eigvals[0]).tolist()

    return {
        "R": float(handle.R_values[r_idx]),
        "pearson_mean": float(np.nanmean(pearsons)),
        "pearson_median": float(np.nanmedian(pearsons)),
        "range_ratio_mean": float(np.nanmean(range_ratios)),
        "range_ratio_median": float(np.nanmedian(range_ratios)),
        "per_observable_pearson": pearsons.tolist(),
        "per_observable_range_ratio": range_ratios.tolist(),
        "direct_peaks": peaks_direct.tolist(),
        "direct_heights": heights_direct.tolist(),
        "synthetic_peaks": peaks_synth.tolist(),
        "synthetic_heights": heights_synth.tolist(),
        "exact_gaps": gaps,
        "n_observables": int(D_direct.shape[0]),
        "n_times": int(D_direct.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Chan-style spectral evaluation")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--samples_per_q", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--ljung_box_p", type=float, default=0.06)
    parser.add_argument("--n_peaks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_r_indices", type=int, nargs="+", default=None)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(args.seed)

    handle = ExactConditionalDatasetHandle(args.data_path)
    model, payload = load_checkpoint_model(args.checkpoint, device=device)

    if args.test_r_indices is not None:
        test_r_indices = args.test_r_indices
    else:
        test_r_indices = payload.get("test_r_indices", list(range(len(handle.R_values))))

    results = []
    for r_idx in test_r_indices:
        print(f"[eval] R={handle.R_values[r_idx]:.4f} | samples_per_q={args.samples_per_q}", flush=True)
        result = evaluate_geometry(
            model=model,
            handle=handle,
            r_idx=int(r_idx),
            n_samples_per_q=args.samples_per_q,
            ljung_box_p=args.ljung_box_p,
            n_peaks=args.n_peaks,
            rng=rng,
            device=device,
        )
        results.append(result)
        print(
            f"  pearson_mean={result['pearson_mean']:.4f} "
            f"range_ratio_mean={result['range_ratio_mean']:.4f} "
            f"direct_peaks={result['direct_peaks'][:5]} "
            f"synthetic_peaks={result['synthetic_peaks'][:5]}",
            flush=True,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "data_path": args.data_path,
        "samples_per_q": args.samples_per_q,
        "results": results,
    }
    out_path = os.path.join(args.save_dir, "spectral_eval.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] spectral evaluation -> {out_path}")


if __name__ == "__main__":
    main()
