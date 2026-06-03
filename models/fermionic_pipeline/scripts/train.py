"""
End-to-end training script for the fermionic shadow generative model.

Usage:
    python -m fermionic_pipeline.scripts.train --config fermionic_pipeline/configs/h4.yaml
    python -m fermionic_pipeline.scripts.train --config fermionic_pipeline/configs/h4.yaml --skip_datagen --data_path data/fermionic_shadows/H4_shadows.h5
"""

import os
import sys
import argparse
import json

import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root is on path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from fermionic_pipeline.models.film_transformer import init_film_transformer, init_crossattn_transformer
from fermionic_pipeline.training.fermionic_trainer import FermionicTrainer
from fermionic_pipeline.data.dataset import FermionicShadowDataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def generate_data(cfg, rng, n_workers=None):
    """Run data generation pipeline."""
    from fermionic_pipeline.data.generate_shadows import generate_chain_data

    dc = cfg["data"]
    R_values = np.round(
        np.arange(dc["r_start"], dc["r_end"] + dc["r_step"] / 2, dc["r_step"]), 2
    )

    times = np.linspace(0, dc["t_max"], dc["n_times"])

    outcomes_dict, metadata = generate_chain_data(
        n_atoms=dc["n_atoms"],
        R_values=R_values,
        times=times,
        n_shadows=dc["n_shadows"],
        rng=rng,
        n_workers=n_workers,
    )
    return outcomes_dict, metadata, R_values, times


def split_data(outcomes_dict, R_values, times, n_test_geom, n_qubits, rng):
    """Split into train/test by holding out geometries."""
    test_idx = np.sort(rng.choice(len(R_values), size=n_test_geom, replace=False))
    test_R = set(R_values[test_idx].tolist())
    train_R = set(R_values.tolist()) - test_R

    train_dict = {k: v for k, v in outcomes_dict.items() if k[0] in train_R}
    test_dict = {k: v for k, v in outcomes_dict.items() if k[0] in test_R}

    train_ds = FermionicShadowDataset.from_outcomes_dict(train_dict, n_qubits)
    test_ds = FermionicShadowDataset.from_outcomes_dict(test_dict, n_qubits)

    print(f"[info] train: {len(train_ds)} samples ({len(train_R)} geometries)")
    print(f"[info] test:  {len(test_ds)} samples ({len(test_R)} geometries)")
    print(f"[info] test R: {sorted(test_R)}")

    return train_ds, test_ds, sorted(test_R)


def plot_losses(history, save_dir):
    """Plot train/val loss and learning rate curves."""
    steps = history["step"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    lr = history["lr"]

    has_val = any(v is not None for v in val_loss)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax = axes[0]
    ax.plot(steps, train_loss, label="Train loss")
    if has_val:
        ax.plot(steps, [v for v in val_loss], label="Val loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("FiLM Transformer Loss (conditioning + decoder)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1]
    ax.plot(steps, lr)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(save_dir, "loss_curves.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[done] saved loss curves -> {fig_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip_datagen", action="store_true")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Parallel workers for data generation (default: serial)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg["data"]
    mc = cfg["model"]
    tc = cfg["training"]

    rng = np.random.default_rng(dc["seed"])
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    n_atoms = dc["n_atoms"]
    n_qubits = n_atoms  # STO-3G active space = n_atoms orbitals = n_atoms qubits
    n_modes = 2 * n_qubits
    param_dim = 1  # bond length

    # Output directory
    save_dir = os.path.join(
        REPO_ROOT, "results", "fermionic_pipeline", args.tag, f"H{n_atoms}"
    )
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Data
    if args.skip_datagen and args.data_path:
        print(f"[info] loading cached data from {args.data_path}")
        train_ds = FermionicShadowDataset.from_hdf5(args.data_path, split="train")
        test_ds = FermionicShadowDataset.from_hdf5(args.data_path, split="test")
        n_modes = train_ds.n_modes
        n_qubits = n_modes // 2
        test_R = []
    else:
        print("[info] generating shadow data...")
        outcomes_dict, metadata, R_values, times = generate_data(cfg, rng, n_workers=args.n_workers)
        n_qubits = metadata["n_qubits"]
        n_modes = 2 * n_qubits

        train_ds, test_ds, test_R = split_data(
            outcomes_dict, R_values, times, dc["n_test_geom"], n_qubits, rng
        )

        # Save datasets
        data_path = os.path.join(save_dir, "shadow_data.h5")
        train_ds.save_hdf5(data_path, split="train")
        test_ds.save_hdf5(data_path, split="test")
        print(f"[done] saved shadow data -> {data_path}")

    # Model
    # Fourier time embedding config
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
        print(f"[info] architecture: cross-attention (Q per-element embedding)")
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
        print(f"[info] architecture: FiLM conditioning")
    print(f"[info] time embedding: {time_embed_kwargs['time_embedding']} (n_freq={time_embed_kwargs['n_freq']}, max_freq={time_embed_kwargs['max_freq']})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model: {n_params:,} parameters | n_qubits={n_qubits} | n_modes={n_modes} | Q_flat_dim={n_modes**2}")

    # Train
    trainer = FermionicTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=test_ds,
        iterations=tc["iterations"],
        lr=tc["lr"],
        final_lr=tc["final_lr"],
        warmup_frac=tc["warmup_frac"],
        weight_decay=tc["weight_decay"],
        batch_size=tc["batch_size"],
        smoothing=tc["smoothing"],
        eval_every=tc["eval_every"],
        device=torch.device(device),
        aux_loss=tc.get("aux_loss"),
        aux_weight=tc.get("aux_weight", 0.1),
        obs_loss_every=tc.get("obs_loss_every", 1),
    )

    aux_desc = tc.get("aux_loss", "none")
    obs_every = tc.get("obs_loss_every", 1)
    print(f"[info] training on {device} for {tc['iterations']} iterations | aux_loss={aux_desc} (every {obs_every})")
    trained_model = trainer.train()

    # Save checkpoint
    ckpt_path = os.path.join(save_dir, "checkpoint.pt")
    torch.save(trained_model.state_dict(), ckpt_path)
    print(f"[done] saved checkpoint -> {ckpt_path}")

    # Save and plot loss history
    history = trainer.history
    history_path = os.path.join(save_dir, "loss_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] saved loss history -> {history_path}")

    plot_losses(history, save_dir)


if __name__ == "__main__":
    main()
