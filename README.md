# Fermionic Shadow Regressor

A pipeline for amortized learning of time-dependent fermionic observables from
matchgate (fermionic) classical-shadow data. A quantum simulation supplies shadow
records of evolved molecular states on a grid of geometries and times; the fermionic
shadow regressor (FSR) — an explicit amplitude–phase model with geometry-conditioned
frequencies — learns to predict the full vector of degree-two Majorana expectation
values at unseen geometries, feeding downstream spectral analysis and observable
reconstruction.

A trained H4 model is available on Hugging Face:
[aniketdesh/molecular-shadows-h4](https://huggingface.co/aniketdesh/molecular-shadows-h4).

## Layout

- `models/fermionic_pipeline/` — the pipeline: data generation (`data/`), the FSR
  model (`models/observable_regressor.py`), training (`training/`), and evaluation
  (`eval/`), plus a vendored Gaussian-process active-learning module (`bo/`).
- `models/scripts/` — analysis and deployment scripts (dipole decomposition,
  GP active-learning experiments, Hugging Face upload).
- `src/` — supporting code and vendored third-party shadow-tomography references.

Run artifacts (datasets, checkpoints, logs, plots) are untracked; the commands below
regenerate them. Molecular Hamiltonians are built on the fly with PennyLane's
`qchem` module, so no external data is required.

## Reproducing the H4 model

Dependencies: `torch`, `numpy`, `scipy`, `h5py`, `pennylane`, `matplotlib`, `tqdm`,
`pyyaml` (Python ≥ 3.11). All commands run from `models/`.

**1. Generate the dataset** (linear H4, STO-3G; R ∈ [0.5, 3.0] Å, t ∈ [0, 300] a.u.;
exact-marginal matchgate-shadow targets over a fixed 500-element library), then add
the per-geometry operational frequency ceiling used by the adaptive bandwidth:

```bash
python -m fermionic_pipeline.data.regression_dataset \
  --output results/regression_targets.h5 \
  --n_atoms 4 --r_start 0.5 --r_end 3.0 --r_step 0.01 \
  --t_max 300.0 --n_times 6001 --n_q 500 --n_workers 16

python -m fermionic_pipeline.data.compute_omega_op \
  --data_path results/regression_targets.h5
```

**2. Train the FSR** (the released configuration: explicit amplitude–phase head,
rank-16 amplitudes, orbital-energy-conditioned frequencies with a soft-floored
adaptive bandwidth):

```bash
python -m fermionic_pipeline.training.regressor_trainer \
  --data_path results/regression_targets.h5 \
  --save_dir results/fsr_h4 \
  --device cuda --seed 42 \
  --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 --warmup_frac 0.05 \
  --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 --fourier_scale 20.0 \
  --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \
  --adaptive_bandwidth --omega_op_floor 8.0 --soft_omega_floor \
  --explicit_amplitude --amp_rank 16 --use_orb_features --standardize_orb_energies \
  --grad_clip 1.0 --alpha_corr 1.0
```

**3. Evaluate** held-out accuracy, render the summary plots, and run the
amplitude-versus-phase composition diagnostic:

```bash
python -m fermionic_pipeline.eval.regressor_eval \
  --data_path results/regression_targets.h5 \
  --checkpoint results/fsr_h4/regressor.pt --save_dir results/fsr_h4/eval

python -m fermionic_pipeline.eval.plot_regression \
  --data_path results/regression_targets.h5 \
  --checkpoint results/fsr_h4/regressor.pt --save_dir results/fsr_h4/plots

python -m fermionic_pipeline.eval.composition_diagnostic \
  --data_path results/regression_targets.h5 \
  --checkpoint results/fsr_h4/regressor.pt --save_dir results/fsr_h4/eval
```

To probe extrapolation, generate a dataset on a wider grid (e.g. R ∈ [0.3, 3.5] Å,
t ∈ [0, 600] a.u.) with the same datagen command and evaluate the trained checkpoint
on it via `fermionic_pipeline.eval.extrapolation_heatmap`.

Each entrypoint documents its full flag set under `--help`.
