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

- `models/fermionic_pipeline/` — the pipeline: data generation (`data/`), model
  (`models/observable_regressor.py`), training (`training/`), evaluation (`eval/`),
  and a vendored Gaussian-process active-learning module (`bo/`).
- `models/scripts/` — analysis and deployment scripts (dipole decomposition,
  GP/active-learning experiments, Hugging Face upload).
- `models/slurm/` — SLURM drivers for data generation, training, and evaluation on
  a cluster, plus `fetch_results.sh` / `deploy.sh` for syncing with it.
- `src/` — supporting code and vendored third-party shadow-tomography references.
- `notebooks/` — exploratory notebooks.

Run artifacts (datasets, checkpoints, logs, plots) are intentionally untracked; the
SLURM drivers regenerate them.
