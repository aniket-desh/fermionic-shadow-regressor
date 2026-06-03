"""
Trainer for the full conditional classifier.
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

from fermionic_pipeline.data.exact_conditional_dataset import (
    ExactConditionalDatasetHandle,
    ExactConditionalTorchDataset,
    split_r_indices,
)
from fermionic_pipeline.models.conditional_classifier import (
    ConditionalClassifierConfig,
    init_conditional_classifier,
)


@dataclass
class TrainConfig:
    steps: int = 50000
    batch_size: int = 256
    lr: float = 1.0e-3
    final_lr: float = 1.0e-7
    warmup_frac: float = 0.05
    weight_decay: float = 1.0e-4
    eval_every: int = 1000
    test_fraction: float = 0.2
    seed: int = 42
    lambda_obs: float = 0.0
    alpha_corr: float = 1.0


def _pearson_corr_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    num = (pred_centered * target_centered).sum(dim=-1)
    denom = torch.sqrt(
        pred_centered.pow(2).sum(dim=-1).clamp_min(eps)
        * target_centered.pow(2).sum(dim=-1).clamp_min(eps)
    )
    return (num / denom.clamp_min(eps)).mean()


class ClassifierTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config: TrainConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.weight_decay,
        )
        warmup_steps = int(config.warmup_frac * config.steps)
        self.scheduler = warm_up_cosine_lr_scheduler(
            optimizer=self.optimizer,
            epochs=config.steps,
            warm_up_epochs=warmup_steps,
            eta_min=config.final_lr,
        )
        self.history = {
            "step": [],
            "train_kl": [],
            "val_kl": [],
            "train_obs": [],
            "val_obs": [],
            "lr": [],
        }
        self.loss_meter = AverageMeter()
        self.kl_meter = AverageMeter()
        self.obs_meter = AverageMeter()

    def _make_loader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=shuffle,
        )

    def _compute_losses(self, batch):
        q_feat, rt, target_probs, obs_target, _ids = batch
        q_feat = q_feat.to(self.device)
        rt = rt.to(self.device)
        target_probs = target_probs.to(self.device)
        obs_target = obs_target.to(self.device)

        logits, obs_pred = self.model(q_feat, rt)
        log_probs = F.log_softmax(logits, dim=-1)
        kl_loss = F.kl_div(log_probs, target_probs, reduction="batchmean")

        obs_loss = torch.tensor(0.0, device=self.device)
        if obs_pred is not None and self.config.lambda_obs > 0.0:
            mse = F.mse_loss(obs_pred, obs_target)
            corr = _pearson_corr_batch(obs_pred, obs_target)
            obs_loss = mse + self.config.alpha_corr * (1.0 - corr)

        total = kl_loss + self.config.lambda_obs * obs_loss
        return total, kl_loss, obs_loss

    def _eval(self, dataset):
        loader = self._make_loader(dataset, shuffle=False)
        self.model.eval()
        kl_meter = AverageMeter()
        obs_meter = AverageMeter()

        with torch.no_grad():
            for batch in loader:
                total, kl_loss, obs_loss = self._compute_losses(batch)
                batch_size = batch[0].shape[0]
                kl_meter.update(kl_loss.item(), batch_size)
                obs_meter.update(obs_loss.item(), batch_size)

        return kl_meter.average(), obs_meter.average()

    def train(self):
        if self.device.type.startswith("cuda"):
            cudnn.benchmark = True

        loader = self._make_loader(self.train_dataset, shuffle=True)
        loader_iter = iter(loader)

        pbar = tqdm(range(1, self.config.steps + 1), desc="Classifier training")
        for step in pbar:
            self.model.train()
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            total, kl_loss, obs_loss = self._compute_losses(batch)

            self.optimizer.zero_grad()
            total.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_size = batch[0].shape[0]
            self.loss_meter.update(total.item(), batch_size)
            self.kl_meter.update(kl_loss.item(), batch_size)
            self.obs_meter.update(obs_loss.item(), batch_size)

            if step % 200 == 0:
                pbar.set_postfix_str(
                    f"kl={self.kl_meter.average():.4f}, obs={self.obs_meter.average():.4f}, "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e}"
                )

            if step % self.config.eval_every == 0:
                val_kl, val_obs = self._eval(self.val_dataset)
                lr = self.scheduler.get_last_lr()[0]
                self.history["step"].append(step)
                self.history["train_kl"].append(self.kl_meter.average())
                self.history["val_kl"].append(val_kl)
                self.history["train_obs"].append(self.obs_meter.average())
                self.history["val_obs"].append(val_obs)
                self.history["lr"].append(lr)
                print(
                    f"[step {step:6d}/{self.config.steps}] "
                    f"train_kl={self.kl_meter.average():.5f} val_kl={val_kl:.5f} "
                    f"train_obs={self.obs_meter.average():.5f} val_obs={val_obs:.5f} "
                    f"lr={lr:.2e}",
                    flush=True,
                )

        return self.model


def load_checkpoint_model(path: str, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model_cfg = ConditionalClassifierConfig(**payload["model_config"])
    model = init_conditional_classifier(**model_cfg.to_dict())
    model.load_state_dict(payload["state_dict"])
    model = model.to(device)
    model.eval()
    return model, payload


def main():
    parser = argparse.ArgumentParser(description="Train the full conditional classifier")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--final_lr", type=float, default=1.0e-7)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--observable_head", action="store_true")
    parser.add_argument("--lambda_obs", type=float, default=0.0)
    parser.add_argument("--alpha_corr", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    handle = ExactConditionalDatasetHandle(args.data_path)
    train_r_idx, test_r_idx = split_r_indices(
        len(handle.R_values), test_fraction=args.test_fraction, seed=args.seed
    )

    train_ds = ExactConditionalTorchDataset(args.data_path, r_indices=train_r_idx)
    test_ds = ExactConditionalTorchDataset(args.data_path, r_indices=test_r_idx)

    model = init_conditional_classifier(
        n_qubits=handle.n_qubits,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        observable_head=args.observable_head,
        n_observables=handle.n_modes,
    )

    train_cfg = TrainConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        final_lr=args.final_lr,
        warmup_frac=args.warmup_frac,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        test_fraction=args.test_fraction,
        seed=args.seed,
        lambda_obs=args.lambda_obs,
        alpha_corr=args.alpha_corr,
    )
    trainer = ClassifierTrainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=test_ds,
        config=train_cfg,
        device=device,
    )
    trained_model = trainer.train()

    payload = {
        "state_dict": trained_model.state_dict(),
        "model_config": model.config.to_dict(),
        "train_config": asdict(train_cfg),
        "train_r_indices": train_r_idx.tolist(),
        "test_r_indices": test_r_idx.tolist(),
        "R_values": handle.R_values.tolist(),
        "times": handle.times.tolist(),
    }
    ckpt_path = os.path.join(args.save_dir, "classifier.pt")
    torch.save(payload, ckpt_path)
    with open(os.path.join(args.save_dir, "history.json"), "w") as f:
        json.dump(trainer.history, f, indent=2)
    with open(os.path.join(args.save_dir, "metadata.json"), "w") as f:
        json.dump(payload | {"state_dict": "<omitted>"}, f, indent=2)

    print(f"[done] checkpoint -> {ckpt_path}")
    print(f"[done] test geometries -> {[handle.R_values[i] for i in test_r_idx]}")


if __name__ == "__main__":
    main()
