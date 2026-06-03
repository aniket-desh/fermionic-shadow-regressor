"""
Self-contained loader for the molecular-shadows observable regressor.

Usage:
    from inference import MolecularShadowsRegressor
    m = MolecularShadowsRegressor.from_local(".")           # after cloning the HF repo
    # or
    m = MolecularShadowsRegressor.from_hub("aniketdesh/molecular-shadows-h2-v10",
                                           revision="v10",  # tag, branch, or commit
                                           token="hf_...")  # only for private repos

    # Predict 120 (or 28 for H2) observable expectations at (R, t):
    y = m.predict(R=1.4, t=12.5)            # scalar -> (n_observables,)
    y = m.predict(R=[1.0, 1.4], t=[5, 10])  # batched -> (B, n_observables)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from observable_regressor import (
    ObservableRegressor,
    ObservableRegressorConfig,
    init_observable_regressor,
)


class MolecularShadowsRegressor:
    def __init__(self, model: ObservableRegressor, payload: dict, orb_grid: np.ndarray,
                 orb_table: np.ndarray, omega_op_table: np.ndarray | None = None,
                 device: str = "cpu"):
        self.model = model.to(device).eval()
        self.payload = payload
        self.orb_grid = orb_grid                # (n_R,)
        self.orb_table = orb_table              # (n_R, n_orb)
        self.omega_op_table = omega_op_table    # (n_R,) or None
        self.device = device

    @property
    def n_observables(self) -> int:
        return self.payload["model_config"]["n_observables"]

    @property
    def observable_keys(self):
        return self.payload["observable_keys"]

    @property
    def R_range(self):
        return float(self.orb_grid.min()), float(self.orb_grid.max())

    @classmethod
    def from_local(cls, repo_dir: str | os.PathLike, device: str = "cpu") -> "MolecularShadowsRegressor":
        repo_dir = Path(repo_dir)
        payload = torch.load(repo_dir / "regressor.pt", map_location=device, weights_only=False)
        config = ObservableRegressorConfig(**payload["model_config"])
        model = init_observable_regressor(**config.to_dict())
        model.load_state_dict(payload["state_dict"])

        orb_npz = np.load(repo_dir / "orbital_energies.npz")
        orb_grid = orb_npz["R_grid"]
        orb_table = orb_npz["orbital_energies"]
        omega_op_table = orb_npz["omega_op"] if "omega_op" in orb_npz.files else None

        return cls(model, payload, orb_grid, orb_table, omega_op_table, device=device)

    @classmethod
    def from_hub(cls, repo_id: str, revision: str | None = None,
                 token: str | None = None, device: str = "cpu",
                 cache_dir: str | None = None) -> "MolecularShadowsRegressor":
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=repo_id, revision=revision, token=token, cache_dir=cache_dir)
        return cls.from_local(local, device=device)

    def _interp_orb(self, R: np.ndarray) -> np.ndarray:
        # Linear per-orbital interpolation on the bundled R-grid
        out = np.empty((R.shape[0], self.orb_table.shape[1]), dtype=np.float32)
        for k in range(self.orb_table.shape[1]):
            out[:, k] = np.interp(R, self.orb_grid, self.orb_table[:, k])
        return out

    def _interp_omega_op(self, R: np.ndarray) -> np.ndarray:
        if self.omega_op_table is None:
            return None
        return np.interp(R, self.orb_grid, self.omega_op_table).astype(np.float32)

    @torch.no_grad()
    def predict(self, R, t):
        R_arr = np.atleast_1d(np.asarray(R, dtype=np.float32))
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float32))
        if R_arr.shape != t_arr.shape:
            R_arr, t_arr = np.broadcast_arrays(R_arr, t_arr)
            R_arr = np.ascontiguousarray(R_arr)
            t_arr = np.ascontiguousarray(t_arr)

        rt = torch.from_numpy(np.stack([R_arr, t_arr], axis=-1)).to(self.device)

        kwargs = {}
        if self.payload["model_config"].get("n_orb_features", 0) > 0:
            orb = self._interp_orb(R_arr)
            kwargs["orb_energies"] = torch.from_numpy(orb).to(self.device)
        if self.payload["model_config"].get("adaptive_bandwidth", False):
            omega_op = self._interp_omega_op(R_arr)
            if omega_op is None:
                raise ValueError("This checkpoint requires omega_op but none is bundled.")
            kwargs["omega_op"] = torch.from_numpy(omega_op).to(self.device)

        y = self.model(rt, **kwargs).cpu().numpy()
        return y[0] if np.isscalar(R) and np.isscalar(t) else y

    @torch.no_grad()
    def predict_trajectory(self, R: float, t_grid: np.ndarray):
        """Convenience: full time series at fixed R. Returns (len(t_grid), n_observables)."""
        t_grid = np.asarray(t_grid, dtype=np.float32)
        return self.predict(np.full_like(t_grid, R, dtype=np.float32), t_grid)


def _print_metadata_summary(m: MolecularShadowsRegressor):
    print(f"  n_observables: {m.n_observables}")
    print(f"  R range:       {m.R_range[0]:.3f} -> {m.R_range[1]:.3f} A")
    print(f"  config:        {json.dumps(m.payload['model_config'])}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_dir", default=".", help="Local checkout of the HF repo")
    ap.add_argument("--R", type=float, default=1.4)
    ap.add_argument("--t", type=float, default=10.0)
    args = ap.parse_args()

    m = MolecularShadowsRegressor.from_local(args.repo_dir)
    _print_metadata_summary(m)
    y = m.predict(args.R, args.t)
    print(f"  predict(R={args.R}, t={args.t}) -> shape {y.shape}, "
          f"min={y.min():.3e}, max={y.max():.3e}, mean={y.mean():.3e}")
