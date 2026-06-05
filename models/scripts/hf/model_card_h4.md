---
license: mit
tags:
  - quantum-computing
  - molecular-simulation
  - shadow-spectroscopy
  - regression
  - pytorch
library_name: pytorch
---

# molecular-shadows-h4

**The H4 direct-observable regressor** for fermionic shadow spectroscopy on
linear $\mathrm{H_4}$ / STO-3G. Given an equal nearest-neighbor bond length $R$
and a propagation time $t$, it predicts the length-120 vector of time-evolved
Majorana expectation values

$$ D_\mu(R, t) = \langle \psi_0(R) \,|\, \Gamma_\mu(t) \,|\, \psi_0(R) \rangle,
\qquad \Gamma_\mu(t) = e^{iH(R)t}\,\Gamma_\mu\,e^{-iH(R)t}, $$

which feed the downstream matchgate-shadow spectroscopy post-processing
(FFT $\to$ peaks $\to$ energy gaps). This is the four-hydrogen model; a
single-reference $\mathrm{H_2}$ companion exists separately.

> **What's new vs the earlier H4 release.** This model uses an
> **explicit-amplitude composition** head and **adaptive Fourier bandwidth**,
> with **chemically-informed orbital-energy inputs**. Short-bond accuracy — the
> structural weak spot of the earlier release ($R < 1.0$ Å, $r \approx 0.40$) —
> is now strong ($r \approx 0.90$). Overall held-out Pearson $0.83 \to 0.93$.

## Architecture

A linear-in-time Fourier head with geometry-conditioned frequencies and an
explicit amplitude/phase factorization (no nonlinear trunk):

$$ \hat{y}_\mu(R, t) = \sum_{k=1}^{256} a_{k\mu}(R)\cos(\omega_k(R)\, t)
   + b_{k\mu}(R)\sin(\omega_k(R)\, t) + \mathrm{dc}_\mu(R), $$

with geometry-conditioned, bandwidth-limited frequencies

$$ \omega_k(R) = \omega_{\mathrm{op}}(R)\cdot
   \sigma\!\big(\mathrm{freq\_net}(\varepsilon(R))\big)_k, $$

where $\varepsilon(R)$ are the HF orbital energies and $\omega_{\mathrm{op}}(R)$
is the per-geometry operational frequency ceiling (soft-floored at 8.0). The
amplitude tensors $a_{k\mu}, b_{k\mu}$ are low-rank factorized (rank 16).

```
(R, t) + HF orbital energies ε(R)
      │
      ├── freq_net(ε(R)) ─ sigmoid ─ × ω_op(R)  → 256 frequencies ω_k(R)   (adaptive bandwidth)
      │
      ├── amp_net(ε(R)) → low-rank a_kμ(R), b_kμ(R), dc_μ(R)   (rank 16)
      │
      └── y_μ(R,t) = Σ_k a_kμ cos(ω_k t) + b_kμ sin(ω_k t) + dc_μ   (explicit-amplitude head)
```

| Hyperparameter | Value |
|---|---|
| n_observables | 120 (degree-1 Majorana monomials, 8 spin-orbitals) |
| n_fourier | 256 |
| explicit_amplitude / amp_rank | True / 16 |
| adaptive_bandwidth | True ($\omega_{\mathrm{op}}$ floor 8.0, soft) |
| conditioned_frequencies | True |
| trunk depth × width | 6 × 768 |
| freq_net depth × width | 3 × 128 |
| residual trunk | None |
| n_orb_features | 4 (HF spatial-orbital energies of H4/STO-3G) |
| Parameter count | ~12.3 M |
| Training | 150k steps, AdamW, cosine LR $10^{-3}\to10^{-7}$, grad-clip 1.0, seed 42 |

## Chemically-informed inputs

The model conditions on **HF orbital energies $\varepsilon(R)$** rather than the
bare scalar $R$. On a paired held-out comparison (identical geometries, same
architecture, same data), orbital-energy inputs beat geometry-only inputs
decisively where multi-reference character lives, and converge with it at
dissociation:

| $R$ bin (Å) | orbital-energy input | scalar-$R$ input |
|---|---|---|
| $<0.74$ | **0.89** | 0.57 |
| $[0.74, 1.0)$ | **0.98** | 0.71 |
| $[1.0, 1.5)$ | 0.96 | 0.88 |
| $[1.5, 2.0)$ | 0.94 | 0.89 |
| $\geq 2.0$ | 0.93 | 0.94 |

This repo ships the orbital-energy model.

## Held-out evaluation

50 held-out geometries on the dense $R \in [0.5, 3.0]$ Å grid (251 total),
per-observable temporal Pearson $r$:

| $R$ bin (Å) | pearson_mean |
|---|---|
| $<0.74$ | 0.90 |
| $[0.74, 1.0)$ | 0.97 |
| $[1.0, 1.5)$ | 0.96 |
| $[1.5, 2.0)$ | 0.93 |
| $\geq 2.0$ | 0.92 |

**Aggregate:** mean Pearson **0.928** across all 50 held-out geometries; top-1
spectral-peak match 44/50. The result is seed-robust — pooling a second seed
($n=100$) gives overall 0.94 with borderline-bin seed spread $|\Delta| < 0.02$.
See `eval_results.json` for per-$R$ numbers.

## Inputs / outputs

- **Input.** $(R, t)$ — equal nearest-neighbor bond length in Å (linear chain:
  H atoms at $0, R, 2R, 3R$) and propagation time in a.u.
- **Output.** Length-120 vector of expectation values $D_\mu(R,t)$ for degree-1
  Majorana observables on H4/STO-3G's 8 spin-orbital JW encoding.
- **Valid range.** $R \in [0.5, 3.0]$ Å, $t \in [0, 300]$ a.u. Accuracy is now
  reasonably uniform across the curve; the only mild residual deficit is in the
  long-$R$ time-domain fit, which the spectral post-processing absorbs.

## Quickstart

```python
from inference import MolecularShadowsRegressor
import numpy as np

m = MolecularShadowsRegressor.from_hub(
    "aniketdesh/molecular-shadows-h4",
    revision="v18-orb",    # immutable architecture pin
    token="hf_...",
)

t_grid = np.linspace(0, 300, 1500)
y = m.predict_trajectory(R=0.8, t_grid=t_grid)   # (1500, 120) — short bond now reliable
```

## Notes & limitations

Earlier H4 releases were structurally weak at short bond ($R < 1.0$ Å): linear
H4 has a multi-reference singlet manifold whose eigenvectors rotate
near-discontinuously through avoided crossings as the chain compresses, and the
old GELU trunk could not encode that rapid amplitude rotation. The
explicit-amplitude head (which factorizes the prediction into per-geometry
amplitudes and frequencies) plus orbital-energy conditioning resolve most of
this — short-bond Pearson is now $\approx 0.90$. The remaining limitation is a
small loss of time-domain fidelity at large $R$ (slowly-varying non-sinusoidal
structure the linear-in-time basis cannot represent); because the downstream
Chan-style spectral analysis reads off frequencies, not waveforms, this does
not degrade recovered energy gaps.

## Files in this repo

| File | Purpose |
|---|---|
| `regressor.pt` | torch payload (state_dict + config + R/t grids) |
| `observable_regressor.py` | architecture (single file) |
| `inference.py` | loader (`MolecularShadowsRegressor`) |
| `orbital_energies.npz` | R-grid + HF orbital-energy table (+ $\omega_{\mathrm{op}}$) |
| `eval_results.json` | per-R held-out metrics (50 geoms) |
| `eval_summary.json` | aggregate |
| `history.json` | training curves |
| `README.md` | this file |

## Versioning

- `v18-orb` (current): explicit-amplitude composition + adaptive bandwidth +
  orbital-energy conditioning, grad-clip 1.0. Mean Pearson 0.928, short-bond
  recovered. Pin via `revision="v18-orb"`.
- Future architectures push as new commits with new tags; existing pins keep
  loading the exact committed version.

## Citation

Method: matchgate-shadow spectroscopy following
[arXiv:2212.11036](https://arxiv.org/abs/2212.11036) and
[matchgate-shadow theory](https://link.springer.com/article/10.1007/s00220-023-04844-0).

## License

MIT.
