"""
Evaluation script: generate synthetic shadows at test geometries and
compare spectral peaks against exact energy gaps.

Usage:
    python -m fermionic_pipeline.scripts.evaluate \
        --config fermionic_pipeline/configs/h4.yaml \
        --checkpoint results/fermionic_pipeline/default/H4/checkpoint.pt \
        --data_path results/fermionic_pipeline/default/H4/shadow_data.h5
"""

import os
import sys
import argparse
import json

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fermionic_pipeline.models.film_transformer import init_film_transformer, init_crossattn_transformer
from fermionic_pipeline.inference.sample import generate_synthetic_shadows
from fermionic_pipeline.inference.spectral_analysis import (
    build_signal_matrix,
    spectral_analysis,
    extract_peaks,
)
from fermionic_pipeline.data.generate_shadows import (
    build_hydrogen_chain_hamiltonian,
    prepare_initial_state,
    time_evolve,
    sample_fermionic_shadows_statevector,
)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _sample_shadows_worker(args):
    """Sample shadows for one time point. Designed for multiprocessing."""
    statevector, n_qubits, n_shadows, seed = args
    rng = np.random.default_rng(seed)
    return sample_fermionic_shadows_statevector(statevector, n_qubits, n_shadows, rng=rng)


def _sample_shadows_all_times(states, times, n_qubits, n_shadows, rng, n_workers=None):
    """Sample direct shadows across all time points, optionally in parallel.

    Each time point gets an independent seed derived from the parent rng,
    so results are deterministic for a given parent seed regardless of
    n_workers. The physics (matchgate application + Born sampling) is
    identical to the serial path.
    """
    # Pre-generate one seed per time point from the parent rng
    seeds = rng.integers(0, 2**63, size=len(times)).tolist()

    if n_workers is not None and n_workers > 1:
        from multiprocessing import Pool

        worker_args = [
            (states[t], n_qubits, n_shadows, seeds[i])
            for i, t in enumerate(times)
        ]
        direct_outcomes = {}
        with Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap(_sample_shadows_worker, worker_args)):
                direct_outcomes[times[i]] = result
                if (i + 1) % 500 == 0:
                    print(f"    direct shadows: {i + 1}/{len(times)} time points", flush=True)
    else:
        direct_outcomes = {}
        for i, t in enumerate(times):
            worker_rng = np.random.default_rng(seeds[i])
            direct_outcomes[t] = sample_fermionic_shadows_statevector(
                states[t], n_qubits, n_shadows, rng=worker_rng
            )
            if (i + 1) % 500 == 0:
                print(f"    direct shadows: {i + 1}/{len(times)} time points", flush=True)

    return direct_outcomes


def evaluate_geometry(
    model,
    R,
    times,
    n_shadows,
    n_qubits,
    H_sparse,
    eigvals_exact,
    rng,
    n_electrons=None,
    batch_size=1000,
    n_workers=None,
    ljung_box_p=None,
):
    """Compare synthetic vs direct shadows at one geometry.

    Returns dict with spectral analysis results for both methods.
    """
    n_modes = 2 * n_qubits

    # Exact energy gaps
    gaps_exact = []
    E0 = eigvals_exact[0]
    for E in eigvals_exact[1:]:
        gaps_exact.append(E - E0)

    # Prepare time-evolved states for direct shadows
    print(f"  preparing HF initial state (n_electrons={n_electrons})...", flush=True)
    psi_0, _ = prepare_initial_state(H_sparse, n_qubits, n_electrons=n_electrons)
    states = time_evolve(H_sparse, psi_0, times)
    print(f"  time evolution done ({len(times)} time points)", flush=True)

    # --- Direct shadows (ground truth baseline) ---
    print(f"  sampling {n_shadows} direct shadows per time point...", flush=True)
    direct_outcomes = _sample_shadows_all_times(
        states, times, n_qubits, n_shadows, rng, n_workers=n_workers
    )

    print(f"  building signal matrix + spectral analysis (direct)...", flush=True)
    D_direct, obs_keys = build_signal_matrix(direct_outcomes, times, k=1, n_workers=n_workers)
    omega_direct, spec_direct, _ = spectral_analysis(D_direct, times, ljung_box_p=ljung_box_p)
    peaks_direct, heights_direct = extract_peaks(omega_direct, spec_direct)
    print(f"  D_direct shape: {D_direct.shape}, {len(peaks_direct)} peaks found", flush=True)

    # --- Synthetic shadows from model ---
    print(f"  generating {n_shadows} synthetic shadows per time point...", flush=True)
    synth_outcomes = generate_synthetic_shadows(
        model,
        R,
        times,
        n_shadows,
        n_qubits,
        batch_size=batch_size,
        rng=rng,
    )

    print(f"  building signal matrix + spectral analysis (synth)...", flush=True)
    D_synth, _ = build_signal_matrix(
        synth_outcomes, times, observable_keys=obs_keys, k=1, n_workers=n_workers
    )
    omega_synth, spec_synth, _ = spectral_analysis(D_synth, times, ljung_box_p=ljung_box_p)
    peaks_synth, heights_synth = extract_peaks(omega_synth, spec_synth)
    print(f"  D_synth shape: {D_synth.shape}, {len(peaks_synth)} peaks found", flush=True)

    return {
        "R": R,
        "gaps_exact": gaps_exact,
        "omega_direct": omega_direct,
        "spectrum_direct": spec_direct,
        "peaks_direct": peaks_direct,
        "omega_synth": omega_synth,
        "spectrum_synth": spec_synth,
        "peaks_synth": peaks_synth,
        "D_direct": D_direct,
        "D_synth": D_synth,
    }


def plot_spectra(results_list, save_dir):
    """Plot spectral comparison for each test geometry in a grid layout."""
    import math

    n = len(results_list)
    if n <= 3:
        nrows, ncols = 1, n
    else:
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, res in enumerate(results_list):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        ax.plot(
            res["omega_direct"],
            res["spectrum_direct"],
            label="Direct Shadow",
            alpha=0.8,
        )
        ax.plot(
            res["omega_synth"],
            res["spectrum_synth"],
            label="Generative Model",
            alpha=0.8,
        )

        # Mark exact gaps
        for gap in res["gaps_exact"]:
            ax.axvline(gap, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

        ax.set_xlabel(r"$\omega$ (energy)")
        ax.set_ylabel(r"$I(\omega)$")
        ax.set_title(f"R = {res['R']:.2f} A")
        ax.legend(fontsize=8)
        ax.set_xlim(0, max(res["omega_direct"].max(), 5))

    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "spectral_comparison.png"), dpi=150)
    print("Saved spectral_comparison.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--test_R",
        type=float,
        nargs="+",
        default=None,
        help="Test geometries (default: evenly spaced holdouts)",
    )
    parser.add_argument("--n_eval_shadows", type=int, default=None)
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Parallel workers for signal matrix construction")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]
    mc = cfg["model"]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(dc["seed"])

    n_atoms = dc["n_atoms"]
    param_dim = 1

    # Determine n_qubits from the molecular Hamiltonian (STO-3G gives 2*n_atoms)
    R_probe = dc["r_start"]
    _, n_qubits = build_hydrogen_chain_hamiltonian(n_atoms, R_probe)
    print(f"H{n_atoms}: {n_qubits} qubits, {2 * n_qubits} Majorana modes")

    # Load model
    time_embed_kwargs = dict(
        time_embedding=mc.get("time_embedding", "fourier"),
        n_freq=mc.get("n_freq", 64),
        max_freq=mc.get("max_freq", 2.0),
        param_range=tuple(mc.get("param_range", (0.5, 3.0))),
    )

    arch = mc.get("architecture", "film")
    if arch == "crossattn":
        model = init_crossattn_transformer(
            n_qubits=n_qubits,
            param_dim=param_dim,
            d_model=mc["d_model"],
            n_layers=mc["n_layers"],
            n_heads=mc["n_heads"],
            d_ff=mc["d_ff"],
            dropout=mc["dropout"],
            hidden_dim=mc["hidden_dim"],
            **time_embed_kwargs,
        )
    else:
        model = init_film_transformer(
            n_qubits=n_qubits,
            param_dim=param_dim,
            d_model=mc["d_model"],
            n_layers=mc["n_layers"],
            n_heads=mc["n_heads"],
            d_ff=mc["d_ff"],
            dropout=mc["dropout"],
            hidden_dim=mc["hidden_dim"],
            inject_every_layer=mc.get("inject_every_layer", False),
            **time_embed_kwargs,
        )
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")

    # Test geometries
    R_values = np.round(
        np.arange(dc["r_start"], dc["r_end"] + dc["r_step"] / 2, dc["r_step"]), 2
    )
    if args.test_R:
        test_R = args.test_R
    else:
        # Evenly spaced holdouts
        test_idx = np.linspace(0, len(R_values) - 1, dc["n_test_geom"], dtype=int)
        test_R = R_values[test_idx].tolist()

    times = np.linspace(0, dc["t_max"], dc["n_times"])
    n_shadows = args.n_eval_shadows or cfg["eval"]["n_eval_shadows"]

    save_dir = args.save_dir or os.path.dirname(args.checkpoint)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Evaluating at R = {test_R}")
    print(f"  {n_shadows} shadows/time, {len(times)} time points")

    results_list = []
    for R in test_R:
        print(f"\n--- R = {R:.2f} A ---")
        from scipy.sparse.linalg import eigsh

        H_sparse, nq = build_hydrogen_chain_hamiltonian(n_atoms, R)
        eigvals, _ = eigsh(H_sparse.tocsc(), k=min(5, 2**nq), which="SA")
        eigvals = np.sort(eigvals)

        res = evaluate_geometry(
            model,
            R,
            times,
            n_shadows,
            n_qubits,
            H_sparse,
            eigvals,
            rng,
            n_electrons=n_atoms,
            n_workers=args.n_workers,
            ljung_box_p=cfg.get("spectroscopy", {}).get("ljung_box_p_threshold"),
        )
        results_list.append(res)

        print(f"  Exact gaps: {[f'{g:.4f}' for g in res['gaps_exact'][:3]]}")
        print(f"  Direct peaks: {[f'{p:.4f}' for p in res['peaks_direct'][:3]]}")
        print(f"  Synth peaks:  {[f'{p:.4f}' for p in res['peaks_synth'][:3]]}")

    plot_spectra(results_list, save_dir)

    # Save numerical results
    summary = {}
    for res in results_list:
        summary[f"R={res['R']:.2f}"] = {
            "gaps_exact": [float(g) for g in res["gaps_exact"][:5]],
            "peaks_direct": [float(p) for p in res["peaks_direct"][:5]],
            "peaks_synth": [float(p) for p in res["peaks_synth"][:5]],
        }
    with open(os.path.join(save_dir, "eval_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {save_dir}")


if __name__ == "__main__":
    main()
