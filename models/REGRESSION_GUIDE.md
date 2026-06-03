## Observable Regression Pipeline — Collaborator Guide

### What this does

The regression pipeline predicts signal matrix entries directly: $f_\theta(R, t) \to S \in \mathbb{R}^K$, where $K=120$ (degree-2 Majorana monomials for 8 qubits / H$_4$). Input is bond length $R$ and evolution time $t$. Output is the exact shadow estimator expectation values — no $Q$ conditioning, no bitstring sampling, no autoregressive generation. From the predicted signal matrix you get the spectral function $I(E)$ via FFT, which reveals energy gaps.

The key physics question for your work: **how does learning quality change across the dissociation curve?** The model recovers spectral peaks well at large $R$ (dissociation limit, simple electronic structure) but fails at short $R$ (near equilibrium, multi-reference character, dense level crossings). The transition region around $R \approx 0.7$–$1.5$ Å for H$_4$ is where things break down.

### Key files

| File | What it does |
|------|-------------|
| `fermionic_pipeline/models/observable_regressor.py` | Model definition. MLP with learnable Fourier features for time. v4+ uses geometry-conditioned frequencies: $\omega_k(R) = \omega_k^{(0)} + g_\phi(R)_k$. ~540k params. |
| `fermionic_pipeline/data/regression_dataset.py` | Dataset generation. Computes exact signal matrix entries analytically from statevectors + matchgate shadow formalism. Stores $(n_R, n_T, K)$ targets in HDF5. |
| `fermionic_pipeline/training/regressor_trainer.py` | Training loop. MSE + Pearson correlation loss ($\alpha_\text{corr}$) + spectral auxiliary loss ($\alpha_\text{spec}$, new in v5). AdamW + cosine LR. Geometry-based train/test split. |
| `fermionic_pipeline/eval/regressor_eval.py` | Spectral evaluation. Builds $D$ matrix from model predictions, runs FFT-based spectral analysis, compares peaks to exact energy gaps. |
| `fermionic_pipeline/eval/plot_regression.py` | Plotting. Produces spectral comparison grids, per-geometry time series, $D$ matrix / covariance / spectrum pipeline plots (Chan et al. Fig 2 style). |
| `fermionic_pipeline/inference/spectral_analysis.py` | Shared spectral analysis utilities. FFT, Ljung-Box pre-screening, peak extraction. Used by both regression and generative pipelines. |
| `log.md` | Full experiment log with results tables and diagnoses for v1–v4. Start here for context. Search for "4/2 — direct observable regression". |

### How to run

**Environment**: Python 3.11, PyTorch, h5py, scipy, pyscf (for Hamiltonian construction). On the cluster, `module load StdEnv/2023 python/3.11 cuda/12.2 scipy-stack/2024a` and activate the `gqs` virtualenv.

**1. Generate dataset** (or reuse existing v2 dataset):
```bash
python3 -m fermionic_pipeline.data.regression_dataset \
  --output results/fermionic_pipeline/regression/my_dataset/regression_targets.h5 \
  --n_atoms 4 --r_start 0.5 --r_end 3.0 --r_step 0.05 \
  --t_max 100.0 --n_times 500 --n_q 1000 --n_workers 8
```

For H$_2$, use `--n_atoms 2`. This changes $K$ from 120 to 6 (degree-2 Majorana monomials for 4 qubits). H$_2$ is much faster to generate and train.

**2. Train**:
```bash
python3 -m fermionic_pipeline.training.regressor_trainer \
  --data_path <dataset.h5> \
  --save_dir <output_dir> \
  --device cuda \
  --steps 200000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 \
  --d_hidden 512 --n_layers 4 --n_fourier 128 --fourier_scale 15.0 \
  --conditioned_frequencies --freq_net_hidden 64 \
  --alpha_corr 1.0
```

**3. Evaluate**:
```bash
python3 -m fermionic_pipeline.eval.regressor_eval \
  --data_path <dataset.h5> \
  --checkpoint <output_dir>/regressor.pt \
  --save_dir <output_dir>/eval \
  --device cuda --ljung_box_p 0.06
```

**4. Plot**:
```bash
python3 -m fermionic_pipeline.eval.plot_regression \
  --data_path <dataset.h5> \
  --checkpoint <output_dir>/regressor.pt \
  --save_dir <output_dir>/plots \
  --device cuda --ljung_box_p 0.06
```

Or use the slurm scripts for the full pipeline: `bash slurm/regression_v5.sh <tag>` submits train $\to$ eval as a 2-job chain on the cluster.

### Experiment history (v2–v4 results)

| Version | Change | Mean Pearson (10 test geom) | Where peaks match |
|---------|--------|-----------------------------|-------------------|
| v2 | 51 geometries, baseline | 0.51 | $R \geq 2.5$ |
| v3 | Bigger model (530k), correlation loss | 0.53 | $R \geq 2.5$, some at $R \approx 1.0$ |
| v4 | Geometry-conditioned frequencies | 0.59 | $R \geq 1.9$ |
| v5 | Spectral auxiliary loss (submitted, pending) | TBD | Targeting $R < 1.5$ |

### Notes for exploring phase transitions

- The dataset HDF5 stores exact eigenvalues at each geometry (`eigvals` dataset, shape $(n_R, 2^{n_\text{qubits}})$). You can extract the full spectrum vs $R$ directly without running the model.
- The `R_values` dataset gives the geometry grid. For H$_4$ with $\Delta R = 0.05$, you get 51 points from 0.5–3.0 Å.
- The signal matrix `expectations` dataset (shape $(n_R, n_T, K)$) contains exact targets — useful for analyzing how the observable structure itself changes across the PES, independent of model quality.
- The eval JSON (`regressor_eval.json`) has per-geometry Pearson correlations, peak locations, and exact gaps — directly tells you where the model succeeds/fails as a function of $R$.
- For H$_2$ specifically: set `--n_atoms 2` in dataset generation. H$_2$ has 4 qubits, $K=6$ observables, and only 1 relevant energy gap along most of the PES. The model should work well everywhere except possibly near the Coulson–Fischer point.
