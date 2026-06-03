"""
Trainer for the direct observable regressor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _mod in ["src.training", "src.data.loading"]:
    if _mod not in sys.modules:
        _pkg = types.ModuleType(_mod)
        _pkg.__path__ = [os.path.join(_REPO_ROOT, *_mod.split("."))]
        _pkg.__package__ = _mod
        sys.modules[_mod] = _pkg

from src.training.utils import AverageMeter, warm_up_cosine_lr_scheduler

from fermionic_pipeline.data.exact_conditional_dataset import split_r_indices
from fermionic_pipeline.data.regression_dataset import (
    RegressionDatasetHandle,
    RegressionTorchDataset,
)
from fermionic_pipeline.models.observable_regressor import (
    ObservableRegressorConfig,
    init_observable_regressor,
)


@dataclass
class RegressorTrainConfig:
    steps: int = 50000
    batch_size: int = 256
    lr: float = 1e-3
    final_lr: float = 1e-7
    warmup_frac: float = 0.05
    weight_decay: float = 1e-4
    eval_every: int = 1000
    test_fraction: float = 0.2
    seed: int = 42
    alpha_corr: float = 0.0
    alpha_spec: float = 0.0
    spec_loss_every: int = 10
    spec_n_geom: int = 2
    alpha_temporal_corr: float = 0.0
    temporal_corr_every: int = 10
    temporal_corr_n_geom: int = 2
    grad_clip: float = 0.0  # max grad norm; 0 disables. Guards the unbounded v16 residual trunk.
    residual_grad_clip: float = 0.0  # if >0, clip residual trunk separately at this norm and
    # apply grad_clip only to the rest (decoupled clipping, v17). 0 = old behavior (grad_clip
    # clips ALL params jointly). Decouples so the explicit branch's composition learning is not
    # throttled by the residual trunk's large gradients (the v16 5/27 borderline-regression cause).


def _pearson_corr(pred, target, eps=1e-8):
    """Cross-observable Pearson at fixed (R, t).

    NOTE: this is the across-K-observables correlation per sample, then
    averaged over the batch. It is NOT the same as the eval-time per-observable
    temporal Pearson reported in regressor_eval.py — that one correlates each
    observable's full time-trajectory against the exact target. Use
    `_temporal_corr_loss` for the eval-aligned objective.
    """
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    num = (p * t).sum(dim=-1)
    denom = (p.pow(2).sum(dim=-1).clamp_min(eps) * t.pow(2).sum(dim=-1).clamp_min(eps)).sqrt()
    return (num / denom.clamp_min(eps)).mean()


class RegressorTrainer:
    def __init__(self, model, train_dataset, val_dataset, config, device,
                 handle=None, train_r_indices=None):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )

        # Decoupled-clipping param groups (v17). Residual trunk = self.model.net when the
        # model has the explicit+residual architecture; everything else is the explicit branch
        # (amp_net / dc_net / freq_net / omega_base). Precomputed once so the train loop is cheap.
        mcfg = self.model.config
        trunk = getattr(self.model, "net", None)
        if (getattr(mcfg, "explicit_amplitude", False)
                and getattr(mcfg, "with_residual", False)
                and trunk is not None):
            self._residual_params = list(trunk.parameters())
            _res_ids = {id(p) for p in self._residual_params}
            self._non_residual_params = [
                p for p in self.model.parameters() if id(p) not in _res_ids
            ]
        else:
            self._residual_params = []
            self._non_residual_params = list(self.model.parameters())
        self.scheduler = warm_up_cosine_lr_scheduler(
            self.optimizer,
            epochs=config.steps,
            warm_up_epochs=int(config.warmup_frac * config.steps),
            eta_min=config.final_lr,
        )
        self.history = {
            "step": [], "train_mse": [], "val_mse": [],
            "train_corr": [], "val_corr": [], "lr": [],
            "spec_loss": [], "temporal_corr_loss": [],
        }
        self.mse_meter = AverageMeter()
        self.corr_meter = AverageMeter()
        self.spec_meter = AverageMeter()

        # Per-R full-time-grid setup (used by spectral and temporal-corr losses)
        self.train_r_indices = train_r_indices if train_r_indices is not None else []
        self._rng = np.random.default_rng(config.seed + 1)
        needs_full_grid = (
            (config.alpha_spec > 0 or config.alpha_temporal_corr > 0)
            and handle is not None
            and len(self.train_r_indices) > 0
        )
        if needs_full_grid:
            N_t = len(handle.times)
            self._times_tensor = torch.from_numpy(handle.times.astype(np.float32)).to(device)
            self._hann_window = torch.hann_window(N_t, device=device)
            self._exact_D = {}
            self._exact_P = {}
            for r_idx in self.train_r_indices:
                D = torch.from_numpy(
                    handle.expectations[r_idx].T.copy().astype(np.float32)
                ).to(device)  # (K, N_t)
                self._exact_D[r_idx] = D
                D_w = D * self._hann_window
                P = torch.abs(torch.fft.rfft(D_w, dim=1)) ** 2
                self._exact_P[r_idx] = P / (P.sum(dim=1, keepdim=True) + 1e-10)
            self._R_values = {r_idx: float(handle.R_values[r_idx]) for r_idx in self.train_r_indices}
            if handle.hf_orbital_energies is not None:
                self._orb_energies = {
                    r_idx: torch.from_numpy(
                        handle.hf_orbital_energies[r_idx].astype(np.float32)
                    ).to(device)
                    for r_idx in self.train_r_indices
                }
            else:
                self._orb_energies = None
            if handle.omega_op is not None:
                self._omega_op = {
                    r_idx: torch.tensor(
                        float(handle.omega_op[r_idx]), dtype=torch.float32, device=device
                    )
                    for r_idx in self.train_r_indices
                }
            else:
                self._omega_op = None
        self.tcorr_meter = AverageMeter()

    def _loader(self, dataset, shuffle=True):
        return DataLoader(
            dataset, batch_size=self.config.batch_size,
            shuffle=shuffle, drop_last=shuffle,
        )

    def _compute_loss(self, batch):
        if len(batch) == 4:
            rt, orb_e, omega_op, targets = batch
            orb_e = orb_e.to(self.device)
            omega_op = omega_op.to(self.device)
        elif len(batch) == 3:
            rt, orb_e, targets = batch
            orb_e = orb_e.to(self.device)
            omega_op = None
        else:
            rt, targets = batch
            orb_e = None
            omega_op = None
        rt = rt.to(self.device)
        targets = targets.to(self.device)
        pred = self.model(rt, orb_energies=orb_e, omega_op=omega_op)
        mse = F.mse_loss(pred, targets)
        corr = _pearson_corr(pred, targets)
        loss = mse
        if self.config.alpha_corr > 0:
            loss = loss + self.config.alpha_corr * (1.0 - corr)
        return loss, mse, corr

    def _spectral_loss(self, r_idx):
        """Phase-invariant spectral matching loss for one geometry.

        Evaluates the model at ALL time points, FFTs predicted vs exact,
        compares normalized power spectra. Provides frequency-direction
        gradients even when prediction is completely out of phase.
        """
        N_t = len(self._times_tensor)
        R_val = self._R_values[r_idx]
        rt = torch.stack([
            torch.full((N_t,), R_val, device=self.device),
            self._times_tensor,
        ], dim=1)

        orb_e = None
        if self._orb_energies is not None and r_idx in self._orb_energies:
            orb_e = self._orb_energies[r_idx].unsqueeze(0).expand(N_t, -1)
        omega_op = None
        if getattr(self, "_omega_op", None) is not None and r_idx in self._omega_op:
            omega_op = self._omega_op[r_idx].expand(N_t)

        pred = self.model(rt, orb_energies=orb_e, omega_op=omega_op).T  # (K, N_t)
        pred_w = pred * self._hann_window
        P_pred = torch.abs(torch.fft.rfft(pred_w, dim=1)) ** 2
        P_pred_norm = P_pred / (P_pred.sum(dim=1, keepdim=True) + 1e-10)

        return F.mse_loss(P_pred_norm, self._exact_P[r_idx])

    def _temporal_corr_loss(self, r_idx, eps=1e-8):
        """Per-observable temporal Pearson, averaged over observables, for one R.

        Mirrors the eval-time metric in regressor_eval.py: for each observable
        μ, correlate the model's full t-trajectory against the exact D[μ, :].
        Returns 1 - mean correlation (so minimizing pushes correlations → 1).
        """
        N_t = len(self._times_tensor)
        R_val = self._R_values[r_idx]
        rt = torch.stack([
            torch.full((N_t,), R_val, device=self.device),
            self._times_tensor,
        ], dim=1)

        orb_e = None
        if self._orb_energies is not None and r_idx in self._orb_energies:
            orb_e = self._orb_energies[r_idx].unsqueeze(0).expand(N_t, -1)
        omega_op = None
        if getattr(self, "_omega_op", None) is not None and r_idx in self._omega_op:
            omega_op = self._omega_op[r_idx].expand(N_t)

        pred = self.model(rt, orb_energies=orb_e, omega_op=omega_op).T   # (K, N_t)
        targ = self._exact_D[r_idx]                                       # (K, N_t)

        p = pred - pred.mean(dim=1, keepdim=True)
        t = targ - targ.mean(dim=1, keepdim=True)
        num = (p * t).sum(dim=1)
        denom = (p.pow(2).sum(dim=1).clamp_min(eps) *
                 t.pow(2).sum(dim=1).clamp_min(eps)).sqrt()
        corr = num / denom.clamp_min(eps)            # (K,)
        return 1.0 - corr.mean()

    def _eval(self, dataset):
        loader = self._loader(dataset, shuffle=False)
        self.model.eval()
        mse_m, corr_m = AverageMeter(), AverageMeter()
        with torch.no_grad():
            for batch in loader:
                _, mse, corr = self._compute_loss(batch)
                n = batch[0].shape[0]
                mse_m.update(mse.item(), n)
                corr_m.update(corr.item(), n)
        return mse_m.average(), corr_m.average()

    def train(self):
        if self.device.type.startswith("cuda"):
            cudnn.benchmark = True

        loader = self._loader(self.train_dataset, shuffle=True)
        loader_iter = iter(loader)

        pbar = tqdm(range(1, self.config.steps + 1), desc="Regressor training")
        for step in pbar:
            self.model.train()
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            loss, mse, corr = self._compute_loss(batch)

            # Spectral auxiliary loss
            if (self.config.alpha_spec > 0
                    and hasattr(self, '_exact_P')
                    and step % self.config.spec_loss_every == 0):
                n_geom = min(self.config.spec_n_geom, len(self.train_r_indices))
                sampled = self._rng.choice(
                    self.train_r_indices, size=n_geom, replace=False,
                )
                spec = torch.tensor(0.0, device=self.device)
                for r_idx in sampled:
                    spec = spec + self._spectral_loss(int(r_idx))
                spec = spec / n_geom
                loss = loss + self.config.alpha_spec * spec
                self.spec_meter.update(spec.item())

            # Temporal-correlation auxiliary loss (matches eval metric)
            if (self.config.alpha_temporal_corr > 0
                    and hasattr(self, '_exact_D')
                    and step % self.config.temporal_corr_every == 0):
                n_geom = min(self.config.temporal_corr_n_geom, len(self.train_r_indices))
                sampled = self._rng.choice(
                    self.train_r_indices, size=n_geom, replace=False,
                )
                tcorr = torch.tensor(0.0, device=self.device)
                for r_idx in sampled:
                    tcorr = tcorr + self._temporal_corr_loss(int(r_idx))
                tcorr = tcorr / n_geom
                loss = loss + self.config.alpha_temporal_corr * tcorr
                self.tcorr_meter.update(tcorr.item())

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.residual_grad_clip > 0 and self._residual_params:
                # Decoupled: clip the residual trunk on its own, leave the explicit
                # branch governed only by grad_clip (typically 0 = unthrottled).
                torch.nn.utils.clip_grad_norm_(
                    self._residual_params, max_norm=self.config.residual_grad_clip,
                )
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self._non_residual_params, max_norm=self.config.grad_clip,
                    )
            elif self.config.grad_clip > 0:
                # Joint clip over all params (old v16 behavior).
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.config.grad_clip,
                )
            self.optimizer.step()
            self.scheduler.step()

            n = batch[0].shape[0]
            self.mse_meter.update(mse.item(), n)
            self.corr_meter.update(corr.item(), n)

            if step % 200 == 0:
                spec_str = f" spec={self.spec_meter.average():.6f}" if self.spec_meter.count > 0 else ""
                tcorr_str = f" tcorr={self.tcorr_meter.average():.4f}" if self.tcorr_meter.count > 0 else ""
                pbar.set_postfix_str(
                    f"mse={self.mse_meter.average():.6f} corr={self.corr_meter.average():.4f}{spec_str}{tcorr_str}"
                )

            if step % self.config.eval_every == 0:
                val_mse, val_corr = self._eval(self.val_dataset)
                lr = self.scheduler.get_last_lr()[0]
                spec_avg = self.spec_meter.average() if self.spec_meter.count > 0 else None
                tcorr_avg = self.tcorr_meter.average() if self.tcorr_meter.count > 0 else None
                self.history["step"].append(step)
                self.history["train_mse"].append(self.mse_meter.average())
                self.history["val_mse"].append(val_mse)
                self.history["train_corr"].append(self.corr_meter.average())
                self.history["val_corr"].append(val_corr)
                self.history["lr"].append(lr)
                self.history["spec_loss"].append(spec_avg)
                self.history["temporal_corr_loss"].append(tcorr_avg)
                spec_str = f" spec={spec_avg:.6f}" if spec_avg is not None else ""
                tcorr_str = f" tcorr={tcorr_avg:.4f}" if tcorr_avg is not None else ""
                print(
                    f"[step {step:6d}/{self.config.steps}] "
                    f"train_mse={self.mse_meter.average():.6f} val_mse={val_mse:.6f} "
                    f"train_corr={self.corr_meter.average():.4f} val_corr={val_corr:.4f}"
                    f"{spec_str}{tcorr_str} lr={lr:.2e}",
                    flush=True,
                )
                self.spec_meter.reset()
                self.tcorr_meter.reset()

        return self.model


def load_checkpoint_model(path, device="cpu"):
    # Fail soft on flaky GPU nodes: the trig-node NVML breakage (crashed the
    # v16/s42 and v17/R_s1729 evals) leaves torch.cuda.is_available() == False,
    # which makes a map_location="cuda" deserialize raise. Fall back to CPU so
    # the eval still completes instead of losing the whole run.
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] requested device={device} but CUDA is unavailable on this node; "
              f"falling back to CPU for checkpoint load + eval")
        device = "cpu"
    payload = torch.load(path, map_location=device, weights_only=False)
    # Drop any model_config keys the current dataclass doesn't know about
    # (forward compat) and let absent new fields fall back to defaults
    # (backward compat with checkpoints from before flag X was added).
    valid_keys = {f.name for f in ObservableRegressorConfig.__dataclass_fields__.values()}
    cfg_dict = {k: v for k, v in payload["model_config"].items() if k in valid_keys}
    config = ObservableRegressorConfig(**cfg_dict)
    model = init_observable_regressor(**config.to_dict())
    # strict=False: tolerate missing buffers (orb_mean/orb_std on pre-flag ckpts)
    # and unexpected keys (modules removed in current code).
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        # Buffers default to 0/1, which is a no-op for standardization.
        # Anything else missing is a real load problem.
        non_buffer_missing = [k for k in missing if not (k.endswith("orb_mean") or k.endswith("orb_std"))]
        if non_buffer_missing or unexpected:
            print(f"[warn] state_dict mismatch: missing={non_buffer_missing}, unexpected={unexpected}")
    model = model.to(device)
    model.eval()
    return model, payload


def main():
    parser = argparse.ArgumentParser(description="Train the observable regressor")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--final_lr", type=float, default=1e-7)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_fourier", type=int, default=64)
    parser.add_argument("--fourier_scale", type=float, default=10.0)
    parser.add_argument("--conditioned_frequencies", action="store_true")
    parser.add_argument("--freq_net_hidden", type=int, default=64)
    parser.add_argument("--freq_net_layers", type=int, default=2)
    parser.add_argument("--use_orb_features", action="store_true",
                        help="Use HF orbital energies as freq_net input (requires dataset with hf_orbital_energies)")
    parser.add_argument("--adaptive_bandwidth", action="store_true",
                        help="ω_k(R) = ω_op(R) · sigmoid(freq_net(ε(R)))_k (requires dataset with omega_op)")
    parser.add_argument("--omega_op_floor", type=float, default=0.0,
                        help="v12: ω_max(R) = max(ω_op(R), floor); 0 = v11 behavior. "
                             "Floors the long-R band so slots aren't packed too tightly when ω_op(R) is small.")
    parser.add_argument("--alpha_corr", type=float, default=0.0,
                        help="Weight on cross-observable Pearson loss (per-(R,t) across K obs). "
                             "NOTE: this is NOT the same as the eval-time per-observable temporal "
                             "Pearson — use --alpha_temporal_corr for eval-aligned objective.")
    parser.add_argument("--alpha_spec", type=float, default=0.0)
    parser.add_argument("--spec_loss_every", type=int, default=10)
    parser.add_argument("--spec_n_geom", type=int, default=2)
    parser.add_argument("--alpha_temporal_corr", type=float, default=0.0,
                        help="Weight on per-observable temporal Pearson loss "
                             "(matches regressor_eval per-observable pearson axis).")
    parser.add_argument("--temporal_corr_every", type=int, default=10)
    parser.add_argument("--temporal_corr_n_geom", type=int, default=2)
    parser.add_argument("--soft_omega_floor", action="store_true",
                        help="Replace clamp(ω_op, min=floor) with smooth softplus form. "
                             "Removes the kink at the R-axis crossover, which lands inside [0.74, 1.0).")
    parser.add_argument("--soft_omega_beta", type=float, default=10.0)
    parser.add_argument("--standardize_orb_energies", action="store_true",
                        help="Standardize HF orbital energies per-feature (mean/std over training R-set) "
                             "before feeding to freq_net. Stats are stored as model buffers.")
    parser.add_argument("--explicit_amplitude", action="store_true",
                        help="Use explicit linear-in-amplitude composition: "
                             "y_μ(R,t) = Σ_k a_kμ(R) cos(ω_k t) + b_kμ(R) sin(ω_k t) + dc_μ(R). "
                             "Replaces the GELU trunk; targets the composition-failure mode.")
    parser.add_argument("--amp_rank", type=int, default=16,
                        help="Low-rank factorization rank for a_kμ = U_kr · V_rμ (0 = full rank).")
    parser.add_argument("--with_residual", action="store_true",
                        help="Add a v12f8-style GELU trunk residual on top of the explicit "
                             "amplitude branch (shared ω). Output layer is zero-initialized so "
                             "training starts at v15_explicit and additively learns long-R "
                             "non-Fourier structure. Requires --explicit_amplitude.")
    parser.add_argument("--grad_clip", type=float, default=0.0,
                        help="Max gradient norm (clip_grad_norm_); 0 disables. With "
                             "--residual_grad_clip set, this applies to the explicit branch only.")
    parser.add_argument("--residual_grad_clip", type=float, default=0.0,
                        help="If >0, clip the residual trunk's gradients separately at this norm "
                             "and apply --grad_clip only to the explicit branch (decoupled "
                             "clipping, v17). Fixes the v16 5/27 finding that joint clipping "
                             "throttled the explicit branch and broke borderline composition. "
                             "Recommended: --residual_grad_clip 1.0 --grad_clip 0.")
    args = parser.parse_args()

    if args.with_residual and not args.explicit_amplitude:
        raise ValueError("--with_residual requires --explicit_amplitude")

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    handle = RegressionDatasetHandle(args.data_path)
    train_r, test_r = split_r_indices(
        len(handle.R_values), test_fraction=args.test_fraction, seed=args.seed,
    )

    train_ds = RegressionTorchDataset(handle, r_indices=train_r)
    test_ds = RegressionTorchDataset(handle, r_indices=test_r)

    n_orb = 0
    if args.use_orb_features:
        if handle.hf_orbital_energies is None:
            raise ValueError("--use_orb_features requires dataset with hf_orbital_energies")
        n_orb = handle.hf_orbital_energies.shape[1]
        print(f"[info] using HF orbital energies as freq_net input ({n_orb} features)")

    if args.adaptive_bandwidth:
        if handle.omega_op is None:
            raise ValueError(
                "--adaptive_bandwidth requires dataset with omega_op field "
                "(run `python3 -m fermionic_pipeline.data.compute_omega_op` "
                "on the HDF5 to add it)"
            )
        if not args.conditioned_frequencies:
            raise ValueError("--adaptive_bandwidth requires --conditioned_frequencies")
        floor_str = f", floored at {args.omega_op_floor}" if args.omega_op_floor > 0 else ""
        print(f"[info] adaptive bandwidth: ω_k(R) = max(ω_op(R){floor_str}) · sigmoid(freq_net(ε(R)))_k")

    model = init_observable_regressor(
        n_observables=handle.n_observables,
        d_hidden=args.d_hidden,
        n_layers=args.n_layers,
        n_fourier=args.n_fourier,
        fourier_scale=args.fourier_scale,
        conditioned_frequencies=args.conditioned_frequencies,
        freq_net_hidden=args.freq_net_hidden,
        freq_net_layers=args.freq_net_layers,
        n_orb_features=n_orb,
        adaptive_bandwidth=args.adaptive_bandwidth,
        omega_op_floor=args.omega_op_floor,
        soft_omega_floor=args.soft_omega_floor,
        soft_omega_beta=args.soft_omega_beta,
        standardize_orb_energies=args.standardize_orb_energies,
        explicit_amplitude=args.explicit_amplitude,
        amp_rank=args.amp_rank,
        with_residual=args.with_residual,
    )

    if args.standardize_orb_energies:
        if n_orb == 0:
            raise ValueError(
                "--standardize_orb_energies requires --use_orb_features"
            )
        # Compute per-feature mean/std over the training R-set only.
        train_orb = handle.hf_orbital_energies[train_r].astype(np.float32)
        orb_mean = torch.from_numpy(train_orb.mean(axis=0))
        orb_std = torch.from_numpy(train_orb.std(axis=0))
        model.set_orb_normalization(orb_mean, orb_std)
        print(f"[info] orb-energy standardization: μ={orb_mean.numpy()}, σ={orb_std.numpy()}")

    if args.explicit_amplitude:
        rank_str = "full rank" if args.amp_rank <= 0 else f"rank {args.amp_rank}"
        print(f"[info] explicit-amplitude composition: y = Σ_k a_kμ(x) cos(ω_k t) + b_kμ(x) sin(ω_k t) + dc_μ(x), {rank_str}")
        if args.with_residual:
            print("[info] residual trunk enabled (v12f8-style MLP on shared ω, zero-init output)")

    if args.residual_grad_clip > 0:
        print(f"[info] decoupled gradient clipping: residual trunk max_norm={args.residual_grad_clip}, "
              f"explicit branch max_norm={args.grad_clip if args.grad_clip > 0 else 'unclipped'}")
    elif args.grad_clip > 0:
        print(f"[info] gradient clipping enabled (joint): max_norm={args.grad_clip}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] model: {n_params:,} params | K={handle.n_observables} observables")
    print(f"[info] train: {len(train_ds)} samples | test: {len(test_ds)} samples")

    train_cfg = RegressorTrainConfig(
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        final_lr=args.final_lr, warmup_frac=args.warmup_frac,
        weight_decay=args.weight_decay, eval_every=args.eval_every,
        test_fraction=args.test_fraction, seed=args.seed,
        alpha_corr=args.alpha_corr,
        alpha_spec=args.alpha_spec,
        spec_loss_every=args.spec_loss_every,
        spec_n_geom=args.spec_n_geom,
        alpha_temporal_corr=args.alpha_temporal_corr,
        temporal_corr_every=args.temporal_corr_every,
        temporal_corr_n_geom=args.temporal_corr_n_geom,
        grad_clip=args.grad_clip,
        residual_grad_clip=args.residual_grad_clip,
    )

    trainer = RegressorTrainer(
        model, train_ds, test_ds, train_cfg, device,
        handle=handle, train_r_indices=train_r.tolist(),
    )
    trained_model = trainer.train()

    payload = {
        "state_dict": trained_model.state_dict(),
        "model_config": model.config.to_dict(),
        "train_config": asdict(train_cfg),
        "train_r_indices": train_r.tolist(),
        "test_r_indices": test_r.tolist(),
        "R_values": handle.R_values.tolist(),
        "times": handle.times.tolist(),
        "observable_keys": [list(k) for k in handle.observable_keys],
    }
    ckpt_path = os.path.join(args.save_dir, "regressor.pt")
    torch.save(payload, ckpt_path)
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(trainer.history, f, indent=2)

    print(f"[done] checkpoint -> {ckpt_path}")
    print(f"[done] test geometries -> {[handle.R_values[i] for i in test_r]}")


if __name__ == "__main__":
    main()
