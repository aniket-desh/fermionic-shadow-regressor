"""
Iteration-based trainer for the FiLM-conditioned fermionic shadow transformer.

Adapts the RydbergConditionalTransformerTrainer pattern but handles the
per-sample (Q, b, x, t) conditioning structure of fermionic shadows.
"""

import os
import sys
import types

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from scipy.special import comb as binom
from tqdm.auto import tqdm

# Bypass src.training.__init__ which imports BaseTrainer → torch_geometric
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _mod in ["src.training", "src.data.loading"]:
    if _mod not in sys.modules:
        _pkg = types.ModuleType(_mod)
        _pkg.__path__ = [os.path.join(_repo_root, *_mod.split("."))]
        _pkg.__package__ = _mod
        sys.modules[_mod] = _pkg

from src.models.transformer_core.loss import LabelSmoothing
from src.training.utils import AverageMeter, warm_up_cosine_lr_scheduler

from fermionic_pipeline.data.dataset import (
    fermionic_collate_fn,
    PAD_TOKEN,
    TOKEN_SHIFT,
    N_OUTCOMES,
)


class FermionicTrainer:
    """Iteration-based trainer for FiLMConditionalTransformer.

    Supports optional auxiliary losses controlled by config:
      - contrastive_q: pushes apart distributions for different Q's at same (x,t)
      - observable_head: predicts Majorana expectations from hidden states
      - observable_expectation: differentiable expected shadow estimator loss

    Unlike the Rydberg trainer which samples random keys from a dict of
    datasets, this trainer uses a standard DataLoader with random sampling
    from the flat (Q, b, x, t) dataset.
    """

    def __init__(
        self,
        model,
        train_dataset,
        iterations,
        lr=1e-3,
        final_lr=1e-7,
        warmup_frac=0.05,
        weight_decay=0.0,
        batch_size=512,
        smoothing=0.0,
        eval_every=1000,
        val_dataset=None,
        device=None,
        aux_loss=None,
        aux_weight=0.1,
        obs_loss_every=1,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.iterations = iterations
        self.batch_size = batch_size
        self.eval_every = eval_every

        # Loss
        vocab_size = N_OUTCOMES + TOKEN_SHIFT
        self.criterion = LabelSmoothing(
            size=vocab_size, padding_idx=PAD_TOKEN, smoothing=smoothing
        )

        # Optimizer + scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=weight_decay,
        )
        warmup_steps = int(warmup_frac * iterations)
        self.scheduler = warm_up_cosine_lr_scheduler(
            optimizer=self.optimizer,
            epochs=iterations,
            warm_up_epochs=warmup_steps,
            eta_min=final_lr,
        )

        # Auxiliary loss
        self.aux_loss_type = aux_loss  # None, "contrastive_q", or "observable_head"
        self.aux_weight = aux_weight

        if aux_loss == "observable_head":
            # Small MLP that predicts per-qubit diagonal Majorana expectations
            # from the model's output hidden states. Trained against ground-truth
            # values derived from bitstrings: <gamma_{2i} gamma_{2i+1}> = (-1)^{b_i}.
            d_model = model.generator.proj.in_features if hasattr(model.generator, 'proj') else 128
            self.obs_head = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model),
                torch.nn.ReLU(),
                torch.nn.Linear(d_model, 1),
                torch.nn.Tanh(),  # expectations are in [-1, 1]
            ).to(device)
            # Add obs_head params to optimizer
            self.optimizer.add_param_group({"params": self.obs_head.parameters()})

        # Observable expectation loss setup
        self.obs_loss_every = obs_loss_every
        if aux_loss == "observable_expectation":
            # n_qubits from model structure
            n_q = model.generator.proj.out_features - TOKEN_SHIFT  # vocab - shift
            # Actually get it from the dataset's n_modes
            self.n_qubits_obs = None  # set on first batch from Q_flat shape
            self.shadow_coeff = None

        self.loss_meter = AverageMeter()
        self.obs_loss_meter = AverageMeter()

        # Loss history for plotting
        self.history = {"step": [], "train_loss": [], "val_loss": [], "lr": [], "obs_loss": []}

    def _contrastive_q_loss(self, Q_flat, xt, tgt, tgt_mask):
        """Push apart distributions for different Q's at the same (x, t).

        Shuffles Q within the batch (so each sample gets a different Q
        but same xt and target prefix) and computes negative KL divergence
        between the original and shuffled first-token distributions.
        If the model ignores Q, KL ≈ 0 and the loss is maximal.
        """
        # Shuffle Q within batch
        perm = torch.randperm(Q_flat.shape[0], device=Q_flat.device)
        Q_shuffled = Q_flat[perm]

        # Get first-token logits for original and shuffled Q
        out_orig = self.model.forward(tgt, tgt_mask, Q_flat, xt)
        out_shuf = self.model.forward(tgt, tgt_mask, Q_shuffled, xt)

        # First-token distributions (position 0 after start token)
        logits_orig = self.model.generator(out_orig[:, 0])  # (batch, vocab)
        logits_shuf = self.model.generator(out_shuf[:, 0])

        p_orig = torch.softmax(logits_orig, dim=-1)
        log_p_shuf = torch.log_softmax(logits_shuf, dim=-1)

        # Negative KL: we WANT distributions to differ, so minimize -KL
        kl = torch.nn.functional.kl_div(
            log_p_shuf, p_orig, reduction="batchmean"
        )
        return -kl

    def _observable_head_loss(self, hidden_states, tgt_y):
        """Predict diagonal Majorana expectations from hidden states.

        For qubit i, the diagonal 2-body Majorana operator gamma_{2i} gamma_{2i+1}
        has expectation (-1)^{b_i} for a computational basis state |b>.
        This provides a per-position regression target that directly links
        the model's internal representation to physical observables.
        """
        # hidden_states: (batch, seq_len, d_model)
        # tgt_y: (batch, seq_len) token indices (shifted: 2 = bit 0, 3 = bit 1)
        batch_size, seq_len = tgt_y.shape

        # Ground truth: (-1)^{b_i} for each qubit position
        bits = (tgt_y - TOKEN_SHIFT).float()  # (batch, seq_len) in {0, 1}
        target = 1.0 - 2.0 * bits  # (batch, seq_len) in {-1, +1}

        # Predict from hidden states
        pred = self.obs_head(hidden_states).squeeze(-1)  # (batch, seq_len)

        # Mask out padding
        mask = (tgt_y != PAD_TOKEN).float()
        mse = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        return mse

    def _observable_expectation_loss(self, Q_flat, xt, tgt_y):
        """Differentiable loss on expected shadow estimators.

        1. Enumerate p_θ(b | Q, x, t) for all 2^n bitstrings
        2. Compute expected observable estimates via precomputed matrix elements
        3. Compare to single-shot targets from training bitstrings
        """
        from fermionic_pipeline.inference.spectral_analysis import batch_obs_matrix_elements

        B = Q_flat.shape[0]
        N = Q_flat.shape[1]
        n_modes = int(round(N ** 0.5))
        n_qubits = n_modes // 2
        dim = 1 << n_qubits

        # Lazy init of shadow coefficient
        if self.shadow_coeff is None:
            self.n_qubits_obs = n_qubits
            self.shadow_coeff = binom(2 * n_qubits, 2, exact=True) / binom(n_qubits, 1, exact=True)

        # Extract permutations from Q_flat
        Q_mat = Q_flat.view(B, n_modes, n_modes)
        perms = Q_mat.abs().argmax(dim=2).cpu().numpy()  # (B, 2n)
        # Signs: value at the argmax position
        signs_np = np.array([
            [Q_mat[s, perms[s, j], j].item() for j in range(n_modes)]
            for s in range(B)
        ])
        signs_np = np.sign(signs_np).astype(np.int8)

        # Precompute matrix elements (numpy, no grad needed)
        obs_keys, M_np = batch_obs_matrix_elements(perms, signs_np, n_qubits, k=1)
        M = torch.tensor(M_np, dtype=torch.float32, device=self.device)  # (B, 2^n, n_obs)

        # Enumerate model distribution (differentiable)
        p_theta = self.model.enumerate_distribution(Q_flat, xt, n_qubits)  # (B, 2^n)

        # Expected observable estimates: E_b[o_mu] = f_k * sum_b p(b) * M[b, mu]
        expected = self.shadow_coeff * torch.einsum("bd,bdo->bo", p_theta, M)  # (B, n_obs)

        # Single-shot targets from training bitstrings
        bits = (tgt_y - TOKEN_SHIFT).cpu().numpy()  # (B, n_qubits)
        b_indices = np.zeros(B, dtype=np.int64)
        for i in range(n_qubits):
            b_indices += bits[:, i].astype(np.int64) << (n_qubits - 1 - i)

        # target[s, mu] = f_k * M[s, b_true, mu]
        targets = self.shadow_coeff * M[torch.arange(B), torch.tensor(b_indices, device=self.device)]  # (B, n_obs)

        loss = ((expected - targets.detach()) ** 2).mean()
        return loss

    def _make_loader(self, dataset, shuffle=True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=fermionic_collate_fn,
            drop_last=True,
            num_workers=0,
        )

    def train(self):
        if self.device.type.startswith("cuda"):
            cudnn.benchmark = True

        loader = self._make_loader(self.train_dataset)
        loader_iter = iter(loader)

        pbar = tqdm(range(1, self.iterations + 1), desc="Training")

        for step in pbar:
            self.model.train()

            # Get batch (cycle through dataset)
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            Q_flat, tgt, tgt_y, tgt_mask, xt = [b.to(self.device) for b in batch]

            # Forward
            out = self.model.forward(tgt, tgt_mask, Q_flat, xt)
            log_probs = self.model.generator(out)

            ntokens = (tgt_y != PAD_TOKEN).sum().item()
            ce_loss = (
                self.criterion(
                    log_probs.contiguous().view(-1, log_probs.size(-1)),
                    tgt_y.contiguous().view(-1),
                )
                / ntokens
            )

            # Auxiliary losses
            aux = torch.tensor(0.0, device=self.device)
            obs_loss_val = None
            if self.aux_loss_type == "contrastive_q":
                aux = self._contrastive_q_loss(Q_flat, xt, tgt, tgt_mask)
            elif self.aux_loss_type == "observable_head":
                aux = self._observable_head_loss(out, tgt_y)
            elif self.aux_loss_type == "observable_expectation":
                if step % self.obs_loss_every == 0:
                    aux = self._observable_expectation_loss(Q_flat, xt, tgt_y)
                    obs_loss_val = aux.item()
                    self.obs_loss_meter.update(obs_loss_val, tgt.shape[0])

            loss = ce_loss + self.aux_weight * aux

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.loss_meter.update(loss.item(), tgt.shape[0])

            # Frequent lightweight log (every 500 steps)
            if step % 500 == 0:
                avg_loss = self.loss_meter.average()
                lr = self.scheduler.get_last_lr()[0]
                obs_str = f", obs={self.obs_loss_meter.average():.4f}" if self.obs_loss_meter.count > 0 else ""
                pbar.set_postfix_str(f"loss={avg_loss:.4f}, lr={lr:.2e}{obs_str}")

            if step % self.eval_every == 0:
                avg_loss = self.loss_meter.average()
                lr = self.scheduler.get_last_lr()[0]
                status = f"loss={avg_loss:.4f}, lr={lr:.2e}"

                val_loss = None
                if self.val_dataset is not None:
                    val_loss = self._eval(self.val_dataset)
                    status += f", val_loss={val_loss:.4f}"

                obs_avg = self.obs_loss_meter.average() if self.obs_loss_meter.count > 0 else None
                if obs_avg is not None:
                    status += f", obs_loss={obs_avg:.4f}"

                self.history["step"].append(step)
                self.history["train_loss"].append(avg_loss)
                self.history["val_loss"].append(val_loss)
                self.history["lr"].append(lr)
                self.history["obs_loss"].append(obs_avg)

                print(
                    f"[step {step:6d}/{self.iterations}] {status}",
                    flush=True,
                )
                pbar.set_postfix_str(status)
                self.loss_meter.reset()
                self.obs_loss_meter.reset()

        return self.model

    @torch.no_grad()
    def _eval(self, dataset, max_batches=10):
        self.model.eval()
        loader = self._make_loader(dataset, shuffle=False)
        total_loss = 0.0
        n = 0

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            Q_flat, tgt, tgt_y, tgt_mask, xt = [b.to(self.device) for b in batch]

            out = self.model.forward(tgt, tgt_mask, Q_flat, xt)
            log_probs = self.model.generator(out)

            ntokens = (tgt_y != PAD_TOKEN).sum().item()
            loss = (
                self.criterion(
                    log_probs.contiguous().view(-1, log_probs.size(-1)),
                    tgt_y.contiguous().view(-1),
                )
                / ntokens
            )
            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)
