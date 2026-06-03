"""
Direct observable regression: f_θ(R, t) → {⟨Γ_μ⟩(R, t)}.

Bypasses the shadow/Q-conditioning pipeline entirely. Predicts signal
matrix entries directly using learnable Fourier features for time encoding.

Two variants:
  - Shared frequencies (v2/v3): ω_k are global learnable parameters
  - Geometry-conditioned (v4+): ω_k(R) = ω_k^{(0)} + g_φ(R)_k where g_φ
    is a small MLP. Energy gaps depend on R, so the optimal Fourier basis
    should too.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ObservableRegressorConfig:
    n_observables: int = 120
    d_hidden: int = 256
    n_layers: int = 3
    n_fourier: int = 64
    fourier_scale: float = 10.0
    conditioned_frequencies: bool = False
    freq_net_hidden: int = 64
    freq_net_layers: int = 2
    n_orb_features: int = 0  # 0 = use scalar R; >0 = use HF orbital energies
    adaptive_bandwidth: bool = False  # ω_k(R) = ω_op(R) · sigmoid(freq_net(ε(R)))_k
    omega_op_floor: float = 0.0  # ω_max(R) = max(ω_op(R), floor); 0 disables (= v11)
    soft_omega_floor: bool = False  # smooth max via softplus; removes clamp kink in R-axis
    soft_omega_beta: float = 10.0
    standardize_orb_energies: bool = False  # apply (ε - μ)/σ in forward; stats live in buffers
    explicit_amplitude: bool = False  # y = Σ_k a_kμ(R) cos(ω_k t) + b_kμ(R) sin(ω_k t) + dc_μ(R)
    amp_rank: int = 16  # low-rank factorization of (K, n_obs) coefficient matrix; 0 = full rank
    with_residual: bool = False  # add v12f8-style MLP residual on top of explicit branch; shared ω

    def to_dict(self):
        return asdict(self)


class ObservableRegressor(nn.Module):
    """Direct regression: (R, t) → K observable expectations.

    Architecture:
      1. Fourier features for t: sin(ω_k(R) * t), cos(ω_k(R) * t)
         - Shared mode: ω_k are global learnable parameters
         - Conditioned mode: ω_k(R) = ω_base_k + freq_net(R)_k
      2. Input = [R, Fourier features] ∈ R^{1 + 2*n_fourier}
      3. MLP with GELU activations → K outputs
    """

    def __init__(self, config: ObservableRegressorConfig):
        super().__init__()
        self.config = config

        # Base frequencies — log-uniform initialization for broad coverage
        # Covers from ~0.05 to ~fourier_scale Eₕ
        log_omega = torch.linspace(
            np.log(0.05), np.log(config.fourier_scale), config.n_fourier
        )
        self.omega_base = nn.Parameter(log_omega.exp())

        # Geometry-conditioned frequency shift (also used as the σ_k head in
        # adaptive_bandwidth mode).
        if config.adaptive_bandwidth and not config.conditioned_frequencies:
            raise ValueError("adaptive_bandwidth requires conditioned_frequencies=True")
        if config.conditioned_frequencies:
            freq_in = config.n_orb_features if config.n_orb_features > 0 else 1
            fn_layers = [nn.Linear(freq_in, config.freq_net_hidden), nn.GELU()]
            for _ in range(config.freq_net_layers - 2):
                fn_layers.extend([
                    nn.Linear(config.freq_net_hidden, config.freq_net_hidden),
                    nn.GELU(),
                ])
            fn_layers.append(nn.Linear(config.freq_net_hidden, config.n_fourier))
            self.freq_net = nn.Sequential(*fn_layers)
        else:
            self.freq_net = None

        # Orbital-energy standardization buffers (no-op when stats are 0/1).
        # Always registered so state_dict shape is consistent regardless of flag,
        # but only meaningful when n_orb_features > 0 and standardize_orb_energies=True.
        n_orb = max(config.n_orb_features, 1)
        self.register_buffer("orb_mean", torch.zeros(n_orb))
        self.register_buffer("orb_std", torch.ones(n_orb))

        if config.explicit_amplitude:
            # Heads producing per-(K, n_obs) cosine and sine coefficients
            # plus a per-observable DC offset, all functions of x = R or ε(R).
            amp_in = config.n_orb_features if config.n_orb_features > 0 else 1
            r = config.amp_rank
            if r > 0 and r < min(config.n_fourier, config.n_observables):
                # Low-rank: a_kμ = Σ_r U_kr · V_rμ, both functions of x.
                # Output 2*(K*r + r*n_obs) for (a, b) cos/sin coefficients.
                amp_out = 2 * (config.n_fourier * r + r * config.n_observables)
            else:
                # Full rank: 2*K*n_obs.
                amp_out = 2 * config.n_fourier * config.n_observables
            amp_layers = [nn.Linear(amp_in, config.d_hidden), nn.GELU()]
            for _ in range(config.n_layers - 1):
                amp_layers.extend([nn.Linear(config.d_hidden, config.d_hidden), nn.GELU()])
            amp_layers.append(nn.Linear(config.d_hidden, amp_out))
            self.amp_net = nn.Sequential(*amp_layers)
            self.dc_net = nn.Sequential(
                nn.Linear(amp_in, config.freq_net_hidden), nn.GELU(),
                nn.Linear(config.freq_net_hidden, config.n_observables),
            )
            # NOTE: residual trunk is built AFTER the xavier loop below so its
            # nn.Linear-time RNG consumption doesn't shift the xavier draws for
            # amp_net/dc_net/freq_net. This keeps v16's explicit-branch params
            # bit-identical to v15_explicit's at init, which is required for
            # the "v16 == v15_explicit at step 0" invariant to hold.
            self.net = None
        else:
            input_dim = 1 + 2 * config.n_fourier
            layers = [nn.Linear(input_dim, config.d_hidden), nn.GELU()]
            for _ in range(config.n_layers - 1):
                layers.extend([nn.Linear(config.d_hidden, config.d_hidden), nn.GELU()])
            layers.append(nn.Linear(config.d_hidden, config.n_observables))
            self.net = nn.Sequential(*layers)
            self.amp_net = None
            self.dc_net = None

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        if config.explicit_amplitude and config.with_residual:
            # Build v12f8-style trunk AFTER the xavier loop above. nn.Linear
            # default kaiming init consumes RNG state at construction time;
            # building the trunk earlier would shift xavier draws for the
            # explicit-branch params, breaking the v16 == v15_explicit init
            # invariant. Constructed here, then zero-init its last layer so
            # residual ≡ 0 at step 0.
            input_dim = 1 + 2 * config.n_fourier
            layers = [nn.Linear(input_dim, config.d_hidden), nn.GELU()]
            for _ in range(config.n_layers - 1):
                layers.extend([nn.Linear(config.d_hidden, config.d_hidden), nn.GELU()])
            layers.append(nn.Linear(config.d_hidden, config.n_observables))
            self.net = nn.Sequential(*layers)
            for p in self.net.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        if self.freq_net is not None:
            nn.init.zeros_(self.freq_net[-1].weight)
            if config.adaptive_bandwidth:
                # Spread initial σ_k across (0, 1) so initial ω_k tile [0, ω_op]
                # rather than collapsing at ω_op/2.
                init_sigma = torch.linspace(0.05, 0.95, config.n_fourier)
                init_logits = torch.log(init_sigma / (1.0 - init_sigma))
                self.freq_net[-1].bias.data.copy_(init_logits)
            else:
                # Original behavior: initial ω ≈ ω_base.
                nn.init.zeros_(self.freq_net[-1].bias)

    def set_orb_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        """Assign standardization stats; called by trainer after computing
        per-feature mean/std over the training R-set."""
        self.orb_mean.copy_(mean.to(self.orb_mean.device, self.orb_mean.dtype))
        self.orb_std.copy_(std.clamp_min(1e-8).to(self.orb_std.device, self.orb_std.dtype))

    def forward(
        self,
        rt: torch.Tensor,
        orb_energies: torch.Tensor = None,
        omega_op: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            rt: (B, 2) tensor with [R, t] per sample
            orb_energies: (B, n_orb) tensor of HF orbital energies, or None
            omega_op: (B,) tensor of operational frequency ceilings ω_op(R),
                required when config.adaptive_bandwidth is True.
        Returns:
            (B, K) predicted observable expectations
        """
        R = rt[:, 0:1]  # (B, 1)
        t = rt[:, 1:2]  # (B, 1)

        if (
            self.config.standardize_orb_energies
            and self.config.n_orb_features > 0
            and orb_energies is not None
        ):
            orb_energies = (orb_energies - self.orb_mean) / self.orb_std

        if self.config.adaptive_bandwidth:
            if omega_op is None:
                raise ValueError("adaptive_bandwidth requires omega_op input")
            if self.config.omega_op_floor > 0.0:
                if self.config.soft_omega_floor:
                    # Smooth max(ω_op, floor): differentiable across the kink so
                    # freq_net can interpolate smoothly through R*.
                    floor = self.config.omega_op_floor
                    beta = self.config.soft_omega_beta
                    omega_op_eff = floor + F.softplus(omega_op - floor, beta=beta)
                else:
                    omega_op_eff = torch.clamp(omega_op, min=self.config.omega_op_floor)
            else:
                omega_op_eff = omega_op
            x = orb_energies if (self.config.n_orb_features > 0 and orb_energies is not None) else R
            sigma = torch.sigmoid(self.freq_net(x))           # (B, n_fourier) in (0, 1)
            omega = omega_op_eff[:, None] * sigma             # (B, n_fourier) in (0, max(ω_op(R), floor))
        elif self.freq_net is not None:
            if self.config.n_orb_features > 0 and orb_energies is not None:
                omega = self.omega_base[None, :] + self.freq_net(orb_energies)
            else:
                omega = self.omega_base[None, :] + self.freq_net(R)
        else:
            omega = self.omega_base[None, :]  # (1, n_fourier)

        if self.config.explicit_amplitude:
            # Linear-in-amplitude composition: y_μ = Σ_k a_kμ cos(ω_k t) + b_kμ sin(ω_k t) + dc_μ.
            # Removes the GELU-trunk burden of synthesizing per-observable, R-dependent
            # mixings of Fourier features — the failure mode flagged by composition_diagnostic.
            x_in = orb_energies if (self.config.n_orb_features > 0 and orb_energies is not None) else R
            wt = omega * t                                    # (B, K), broadcasts even if omega is (1, K)
            if wt.shape[0] == 1:
                wt = wt.expand(t.shape[0], -1)
            cos_t = torch.cos(wt)                             # (B, K)
            sin_t = torch.sin(wt)                             # (B, K)
            amp = self.amp_net(x_in)                          # (B, amp_out)
            B = R.shape[0]
            K = self.config.n_fourier
            n_obs = self.config.n_observables
            r = self.config.amp_rank
            if r > 0 and r < min(K, n_obs):
                # Split into (a_U, a_V, b_U, b_V) low-rank factors.
                splits = [K * r, r * n_obs, K * r, r * n_obs]
                a_U, a_V, b_U, b_V = torch.split(amp, splits, dim=-1)
                a_U = a_U.view(B, K, r); a_V = a_V.view(B, r, n_obs)
                b_U = b_U.view(B, K, r); b_V = b_V.view(B, r, n_obs)
                # y_μ = Σ_r V_rμ Σ_k U_kr cos(ω_k t) + analogous sin term
                tmp_a = torch.einsum("bk,bkr->br", cos_t, a_U)        # (B, r)
                tmp_b = torch.einsum("bk,bkr->br", sin_t, b_U)        # (B, r)
                y = torch.einsum("br,brn->bn", tmp_a, a_V) \
                  + torch.einsum("br,brn->bn", tmp_b, b_V)
            else:
                a, b = amp.view(B, 2, K, n_obs).unbind(dim=1)         # each (B, K, n_obs)
                y = torch.einsum("bk,bkn->bn", cos_t, a) \
                  + torch.einsum("bk,bkn->bn", sin_t, b)
            y = y + self.dc_net(x_in)
            if self.config.with_residual:
                # v12f8-style trunk on shared ω. Last linear layer zero-init →
                # residual ≡ 0 at step 0 → v16 starts as v15_explicit.
                fourier_res = torch.cat([sin_t, cos_t], dim=-1)        # (B, 2K)
                if R.shape[0] == 1 and fourier_res.shape[0] != 1:
                    R_in = R.expand(fourier_res.shape[0], -1)
                else:
                    R_in = R
                trunk_in = torch.cat([R_in, fourier_res], dim=-1)      # (B, 1+2K)
                y = y + self.net(trunk_in)
            return y

        fourier = torch.cat(
            [torch.sin(omega * t), torch.cos(omega * t)],
            dim=-1,
        )  # (B, 2*n_fourier)
        x = torch.cat([R, fourier], dim=-1)
        return self.net(x)


def init_observable_regressor(
    n_observables: int = 120,
    d_hidden: int = 256,
    n_layers: int = 3,
    n_fourier: int = 64,
    fourier_scale: float = 10.0,
    conditioned_frequencies: bool = False,
    freq_net_hidden: int = 64,
    freq_net_layers: int = 2,
    n_orb_features: int = 0,
    adaptive_bandwidth: bool = False,
    omega_op_floor: float = 0.0,
    soft_omega_floor: bool = False,
    soft_omega_beta: float = 10.0,
    standardize_orb_energies: bool = False,
    explicit_amplitude: bool = False,
    amp_rank: int = 16,
    with_residual: bool = False,
):
    return ObservableRegressor(
        ObservableRegressorConfig(
            n_observables=n_observables,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_fourier=n_fourier,
            fourier_scale=fourier_scale,
            conditioned_frequencies=conditioned_frequencies,
            freq_net_hidden=freq_net_hidden,
            freq_net_layers=freq_net_layers,
            n_orb_features=n_orb_features,
            adaptive_bandwidth=adaptive_bandwidth,
            omega_op_floor=omega_op_floor,
            soft_omega_floor=soft_omega_floor,
            soft_omega_beta=soft_omega_beta,
            standardize_orb_energies=standardize_orb_energies,
            explicit_amplitude=explicit_amplitude,
            amp_rank=amp_rank,
            with_residual=with_residual,
        )
    )
