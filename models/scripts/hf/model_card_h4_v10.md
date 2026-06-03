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

# molecular-shadows-h4-v10

Direct observable regressor for fermionic shadow spectroscopy on linear-H4 /
STO-3G. Predicts time-evolved expectation values of 120 Majorana observables
\(\Gamma_\mu(t) = e^{iH(R)t}\Gamma_\mu e^{-iH(R)t}\) as a function of equal
nearest-neighbor bond length \(R\) and time \(t\), to feed downstream
shadow-spectroscopy post-processing.

> **Heads-up.** v10 H4 has uneven accuracy across the PES — strong at \(R \geq 1.5\) Å
> but degraded at \(R < 1.0\) Å where low-lying singlet avoided crossings drive
> non-analytic eigenvector rotation. Use this model with awareness of the
> short-R regime; see "Known limitations" below.

## Architecture (v10)

```
(R, t) + HF orbital energies ε(R)
      │
      ├── freq_net(ε(R))  → 256 learnable Fourier frequencies ω_k(R)
      │
      ├── Fourier features [sin(ω_k t), cos(ω_k t)]  (256 × 2 = 512 features)
      │
      └── Trunk MLP: input [R, fourier]  → 6 layers × 768 hidden → 120 outputs
```

| Hyperparameter           | Value                                        |
|--------------------------|----------------------------------------------|
| n_observables            | 120 (k=1 Majorana operators on 8 spin-orbitals) |
| n_fourier                | 256                                          |
| trunk depth × width      | 6 × 768                                      |
| freq_net depth × width   | 3 × 128                                      |
| n_orb_features           | 4 (HF spatial-orbital energies of H4/STO-3G) |
| conditioned_frequencies  | True                                         |
| adaptive_bandwidth       | False (v10)                                  |
| activation               | GELU                                         |
| Parameter count          | ~14 M                                        |

## Held-out evaluation

50 held-out geometries on the dense \(R \in [0.5, 3.0]\) Å grid (251 total).

| R bin (Å)   | pearson_mean (approx) |
|-------------|-----------------------|
| 0.5–1.0     | ~0.40 (short-R weak)  |
| 1.0–1.5     | ~0.60                 |
| 1.5–2.0     | ~0.90                 |
| 2.0–3.0     | >0.95                 |

Aggregate: mean Pearson 0.834 / median 0.978 across all 50 held-out geometries.
See `eval_results.json` for per-R numbers.

## Inputs / outputs

- **Input.** `(R, t)` — equal nearest-neighbor bond length in Å (linear chain
  geometry: H atoms at 0, R, 2R, 3R) and propagation time in a.u.
- **Output.** Length-120 vector of expectation values
  \(\langle\psi_0(R)|\Gamma_\mu(t)|\psi_0(R)\rangle\) for k=1 Majorana
  observables on H4/STO-3G's 8 spin-orbital JW encoding.
- **Valid range.** \(R \in [0.5, 3.0]\) Å, \(t \in [0, 300]\) a.u. Recommended
  high-confidence range: \(R \geq 1.5\) Å.

## Quickstart

```python
from inference import MolecularShadowsRegressor
import numpy as np

m = MolecularShadowsRegressor.from_hub(
    "aniketdesh/molecular-shadows-h4-v10",
    revision="v10",
    token="hf_...",
)

t_grid = np.linspace(0, 300, 1500)
y = m.predict_trajectory(R=1.8, t_grid=t_grid)   # (1500, 120)
```

## Known limitations

Short-R (R < 1.0 Å) accuracy is structurally weaker. Linear H4 has a
multi-reference singlet manifold whose eigenvectors rotate near-discontinuously
through avoided crossings as the chain compresses. The current freq_net
correctly tracks energy-gap motion (eigenvalues are smooth), but the trunk
struggles to encode the rapid amplitude rotation that lives in the eigenvector
sector. H2 v10 (single-reference, no avoided crossings) confirms the recipe is
sound on simpler chemistry — the bottleneck is H4-specific.

Resource-experiment guidance: trust v10 H4 most strongly for \(R \geq 1.5\) Å;
treat short-R predictions as exploratory. A v11 release with adaptive Fourier
bandwidth is in development to address part of this.

## Files in this repo

| File | Purpose |
|------|---------|
| `regressor.pt`            | torch payload (state_dict + config + R/t grids) |
| `observable_regressor.py` | architecture |
| `inference.py`            | loader |
| `orbital_energies.npz`    | R-grid + HF orbital-energy table |
| `eval_results.json`       | per-R held-out metrics (50 geoms) |
| `eval_summary.json`       | aggregate |
| `history.json`            | training curves |
| `README.md`               | this file |

## Versioning

- `v10` (current): HF orbital-energy freq_net + dense R-grid + 6×768 trunk.
  Mean Pearson 0.834 (R≥1.5 Å strong, R<1.0 Å weak).
- Future versions will be pushed as new commits with new tags. Pin via
  `revision="v10"` to preserve loading across architecture changes.

## Citation

Method: matchgate-shadow spectroscopy following
[arXiv:2212.11036](https://arxiv.org/abs/2212.11036) and
[matchgate-shadow theory](https://link.springer.com/article/10.1007/s00220-023-04844-0).

## License

MIT.
