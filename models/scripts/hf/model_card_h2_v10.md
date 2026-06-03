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

# molecular-shadows-h2-v10

Direct observable regressor for fermionic shadow spectroscopy on H2 / STO-3G.
Predicts time-evolved expectation values of 28 Majorana observables
\(\Gamma_\mu(t) = e^{iH(R)t}\Gamma_\mu e^{-iH(R)t}\) as a function of bond
length \(R\) and time \(t\), with the goal of feeding the resulting signal
matrix into the Chan et al. shadow-spectroscopy post-processing pipeline to
recover energy gaps without expensive quantum-circuit-side time evolution.

## Architecture (v10)

```
(R, t) + HF orbital energies ε(R)
      │
      ├── freq_net(ε(R))  → 256 learnable Fourier frequencies ω_k(R)
      │
      ├── Fourier features [sin(ω_k t), cos(ω_k t)]  (256 × 2 = 512 features)
      │
      └── Trunk MLP: input [R, fourier]  → 6 layers × 768 hidden → 28 outputs
```

| Hyperparameter           | Value                                        |
|--------------------------|----------------------------------------------|
| n_observables            | 28 (k=1 Majorana operators on 4 spin-orbitals) |
| n_fourier                | 256                                          |
| trunk depth × width      | 6 × 768                                      |
| freq_net depth × width   | 3 × 128                                      |
| n_orb_features           | 2 (HF spatial-orbital energies of H2/STO-3G) |
| conditioned_frequencies  | True                                         |
| adaptive_bandwidth       | False (v10)                                  |
| activation               | GELU                                         |
| Parameter count          | ~14 M                                        |

## Held-out evaluation

50 held-out geometries on the dense \(R \in [0.5, 3.0]\) Å grid
(\(\Delta R = 0.01\), 251 total). Trained on the remaining 201.

| R bin (Å)  | n  | pearson_mean | pearson_median | range_ratio | MSE       |
|------------|----|--------------|----------------|-------------|-----------|
| 0.65–1.11  | 11 | 0.9859       | 0.9984         | 0.9893      | 3.58e-6   |
| 1.11–1.56  | 8  | 0.9861       | 0.9980         | 0.9903      | 6.19e-6   |
| 1.56–2.02  | 7  | 0.9918       | 0.9982         | 0.9923      | 7.07e-6   |
| 2.02–2.47  | 12 | 0.9987       | 0.9994         | 0.9988      | 2.52e-6   |
| 2.47–2.93  | 12 | 0.9996       | 0.9997         | 0.9995      | 1.42e-6   |
| **all**    | 50 | **0.9931**   | **0.9967**     | 0.9946      | 3.71e-6   |

Pearson is per-observable, then averaged across the 28 observables and reported
as `mean` and `median` of those 28 values for each held-out R.

## Inputs / outputs

- **Input.** `(R, t)` where `R` is bond length in Å and `t` is propagation time
  in atomic units (\(\hbar/E_h\)).
- **Output.** Length-28 vector of expectation values
  \(\langle\psi_0(R)|\Gamma_\mu(t)|\psi_0(R)\rangle\) for the 28 k=1 Majorana
  observables on H2/STO-3G's 4 spin-orbital JW encoding. Initial state
  \(|\psi_0\rangle\) is Hartree–Fock with explicit symmetry-breaking
  excitations to populate non-trivial gap manifolds.
- **Valid range.** Trained on \(R \in [0.5, 3.0]\) Å, \(t \in [0, 300]\) a.u.
  Extrapolation outside is unsupported.

## Quickstart

```python
from huggingface_hub import snapshot_download
from inference import MolecularShadowsRegressor

# token only needed while the repo is private
m = MolecularShadowsRegressor.from_hub(
    "aniketdesh/molecular-shadows-h2-v10",
    revision="v10",                # pin the architecture version
    token="hf_...",
)

import numpy as np
t_grid = np.linspace(0, 300, 1500)
y = m.predict_trajectory(R=1.4, t_grid=t_grid)   # (1500, 28) trajectory at R=1.4 Å
```

## Training data

- Bond-length grid: \(R \in [0.5, 3.0]\) Å, \(\Delta R = 0.01\) Å (251 points).
- Time grid: \(t \in [0, 300]\) a.u., 1500 points (\(\Delta\omega \approx 0.021\,E_h\),
  \(\omega_{\max} \approx 15.7\,E_h\) Nyquist).
- Initial state: Hartree–Fock with symmetry-breaking excitations.
- Targets: exact ED of H2/STO-3G via PennyLane, observables k=1 Majorana operators.
- Train/test split: 201 / 50, random per-R holdout, seed 42.

## Files in this repo

| File | Purpose |
|------|---------|
| `regressor.pt`            | torch payload: state_dict + model_config + R/t grids + observable_keys |
| `observable_regressor.py` | model architecture (single file, no project deps) |
| `inference.py`            | `MolecularShadowsRegressor.from_local` / `from_hub` loader |
| `orbital_energies.npz`    | R-grid + HF orbital-energy table for inference-time interpolation |
| `eval_results.json`       | per-R held-out eval metrics (50 geometries) |
| `eval_summary.json`       | aggregate metrics |
| `history.json`            | training loss / val MSE curves |
| `README.md`               | this file |

## Versioning

- `v10` (current): HF orbital-energy freq_net + dense R-grid + 6×768 trunk.
  Mean Pearson 0.993 across the full PES.
- Future versions (`v11+`) will be pushed as new commits on `main` with new
  tags. Pin via `revision="v10"` to preserve loading across architecture
  changes; `main` always tracks the latest.

## Citation

Method: matchgate-shadow spectroscopy following
[arXiv:2212.11036](https://arxiv.org/abs/2212.11036) and
[matchgate-shadow theory](https://link.springer.com/article/10.1007/s00220-023-04844-0).

Code and training pipeline are research-internal; please contact for citation
text once the manuscript is on arXiv.

## License

MIT.
