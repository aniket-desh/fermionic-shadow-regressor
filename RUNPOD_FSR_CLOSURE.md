# RunPod briefing — FSR scientific-closure experiment fleet

## Mission

Run the experiments that distinguish genuine geometry amortization of coherent dynamics from
dense interpolation, oracle-screen dependence, fixed-library residual fitting, orbital-gauge
artifacts, and favorable time-domain metrics. Do not redesign the paper or add molecules. The
deliverable is an evidence package and a short decision-oriented report.

This fleet implements the five closure tests requested for the FSR:

1. sparse/coarse/blocked geometry interpolation and acquisition learning curves;
2. train-only, global, and misspecified line-screen experiments;
3. same-library, independent-library, and exact-Majorana evaluation;
4. maximum-overlap/sign/Procrustes orbital-gauge alignment on N2 and H2O;
5. suite-wide end-to-end spectral metrics and a structured line-extraction baseline.

It also runs the requested architecture audits: exact signal-matrix singular values, effective
frequency usage, one-particle physicality, K/rank sweeps, known-t=0 anchoring, and an ordered
frequency bank. The last group is improvement work; the first five groups determine closure.

## Non-negotiable Git boundary

The branch to run is `dev`. The code and this briefing are intentionally committed there.

Do **not** commit or push any of the following:

- HDF5 datasets, checkpoints, raw logs, Hugging Face snapshots, caches, or tokens;
- anything under `models/results/`, `models/logs/`, or `models/runpod_logs/`;
- generated `_*.sh` files, environment files, or changes to paper/manuscript directories;
- regenerated QA/QB/exact/aligned datasets or trained `.pt` files.

At the end, create a separate branch named `runpod-results-fsr-closure`. Commit only:

- `FSR_CLOSURE_RESULTS.md` (the required concise report);
- a new `fsr_closure_results/` directory containing compact aggregate JSONs and the two geometry
  learning-curve PDFs; exclude per-frequency arrays if the directory would exceed roughly 20 MB.

Never force-add ignored datasets/results. Never modify or merge `dev`. Push the results branch and
report its commit hash. If you need to change code to fix a real bug, make a separate small commit
on the results branch and describe it explicitly; do not silently change the experiment.

## Setup

Use a large CPU pod with one modern CUDA GPU. Data generation is the long pole; training is light
for a 12M-parameter model. A persistent volume is strongly recommended. The full fleet is large;
budget several pod-days if every independent H4 library is regenerated at the full grid.

```bash
git clone <repo-url> fermionic-shadow-regressor
cd fermionic-shadow-regressor
git checkout dev
git pull --ff-only origin dev

python -m pip install -r requirements.txt
python -m pip install huggingface_hub
cd models
python -m scripts.fetch_plot_data --with-datasets
```

The fetch command prints the Hugging Face snapshot path. Export it exactly:

```bash
export RAW_ROOT=/root/.cache/huggingface/hub/datasets--aniketdesh--molecular-shadows-datasets/snapshots/<HASH>
export OUT=results/fsr_closure
export N_WORKERS=$(nproc)
export DEVICE=cuda
export STEPS=150000
```

Verify that `$RAW_ROOT/{h4,lih,beh2,n2,h2o}/` each contains
`regression_targets.h5` and `regressor.pt`. Do not proceed with missing files.

Run the local gate before spending compute:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig python -m pytest tests/ -q
bash -n scripts/run_fsr_closure_runpod.sh
```

## Runner and restart behavior

All stages use one entrypoint and write only beneath `$OUT`:

```bash
bash scripts/run_fsr_closure_runpod.sh <stage>
```

Stages are resumable: an existing checkpoint or terminal JSON is skipped. Re-running a stage after
an interruption is the normal recovery procedure. Keep the pod volume; do not delete partial HDF5s
unless the log proves they are corrupt.

Recommended order:

```bash
# Cheap gates and metrics first
bash scripts/run_fsr_closure_runpod.sh splits
bash scripts/run_fsr_closure_runpod.sh spectra

# Primary claim: sparse geometry amortization
bash scripts/run_fsr_closure_runpod.sh geometry

# Oracle sensitivity
bash scripts/run_fsr_closure_runpod.sh screen

# CPU-heavy measurement-ensemble test
bash scripts/run_fsr_closure_runpod.sh libraries

# PySCF overlap audit + two retrains
bash scripts/run_fsr_closure_runpod.sh gauge

# Diagnostics and optional architecture improvements
bash scripts/run_fsr_closure_runpod.sh architecture
```

`all` runs the same sequence but is less convenient for monitoring:

```bash
bash scripts/run_fsr_closure_runpod.sh all
```

## Stage 1 — geometry amortization

Molecules: H4 (primary) and N2 (hard/congested control).

The runner pre-registers explicit index manifests and feeds them to the trainer. It runs:

- learning curves at 5, 10, 20, 40, 80, and (H4) 160 training geometries;
- coarse-to-fine grids at 0.05, 0.10, and 0.20 Angstrom, evaluated on the original 0.01 grid;
- contiguous holdouts of width 0.10, 0.20, and 0.30 Angstrom;
- H4 blocks centered at 0.85 and 1.25 Angstrom;
- N2 blocks centered at 2.20 Angstrom, around the avoided-crossing failure region.

Every split receives both the FSR and the structured comparator:

> extract a shared line bank at each training geometry, fit all channel amplitudes by linear least
> squares, track line identities, interpolate frequencies/amplitudes in geometry, reconstruct.

Primary outputs:

- `$OUT/geometry/<mol>/<split>/spectral_metrics.json`;
- `$OUT/geometry/<mol>/<split>/structured_baseline.json`;
- `$OUT/geometry/<mol>/geometry_summary.json`;
- `$OUT/geometry/<mol>/geometry_learning_curve.pdf`.

Decision readout:

- report the smallest `N_train` reaching temporal Pearson 0.90 and 0.95;
- report the smallest `N_train` reaching dominant-line recall 0.90;
- state `N_train / N_all` and the explicit acquisition count `N_R N_T N_Q S`;
- compare FSR with line extraction at every threshold;
- do not average blocked failures into random/coarse results.

The exact-marginal data use `S=1` only as an accounting placeholder, not as one hardware shot. For
finite-shot translation, multiply by the desired Born samples per Q. State explicitly that
`S=1000, N_Q=500` means 500,000 outcomes per geometry-time point.

## Stage 2 — screen robustness

The released H4 checkpoint is evaluated twice: legacy per-test-geometry dataset ceiling (oracle
upper bound) and interpolation from checkpoint training geometries only. Retrained variants use:

- a fully train-only grid variant: global ceiling, time stride, and horizon derived only from the
  released checkpoint's training geometries (with a 10% bandwidth safety factor);
- ceiling multipliers 0.75, 1.25, and 1.50;
- one conservative global ceiling equal to the maximum over training geometries only;
- shortened training horizons 150 and 225 a.u. versus the 300-a.u. reference;
- bandwidth floors 6, 10, and 12 Eh versus the 8-Eh reference.

The effective training ceiling is stored in each checkpoint, and train-interp evaluation reads
that stored value. It must not reopen held-out screen values.

Decision readout: identify the largest misspecification retaining at least 95% of the reference
Pearson and dominant-line recall. If a 20-50% overestimate is safe, say that the screen need only
provide a conservative bound. If a modest change collapses performance, state that the screen is
part of the effective model, not merely grid design.

## Stage 3 — matchgate-library generalization

The runner uses the released seed-42 H4 dataset as QA, generates three independent NQ=500
libraries (seeds 101, 202, 303), and constructs exact degree-two Majorana targets on the same R/t
grid. It evaluates:

- released QA model -> QA, independent QB, exact;
- one model trained on each independent library -> its own Q, a different Q, exact.

Outputs are `$OUT/libraries/h4/*_cross_library.json`. These separate regression error,
finite-library residual, and Born-shot error. Report cross-library mean and standard deviation,
not only one favorable QA->QB pair.

Gate: QA->QB and QA->exact should preserve coherent performance. A large QA-only advantage means
the network learns deterministic finite-library residual and must be reported as such.

## Stage 4 — orbital gauge

For N2 and H2O, PySCF canonical active orbitals are tracked across neighboring geometries using:

1. maximum absolute overlap assignment;
2. consistent sign fixing;
3. Procrustes rotation within subspaces whose neighboring orbital-energy gaps are below 0.05 Eh.

The 120-channel Majorana covariance is rotated into that tracked basis. The runner writes an
aligned HDF5, retrains the unchanged FSR recipe, and compares raw versus aligned spectral metrics.

Inspect `$OUT/gauge/<mol>/gauge_report.json` for permutations, sign flips, minimum matched overlap,
off-diagonal overlap norm, Procrustes blocks, and raw/aligned smoothness. Do not interpret a low
overlap step as physical nonsmoothness without mentioning the gauge event.

Decision readout:

- if alignment materially improves N2/H2O, revise the failure interpretation to include orbital
  gauge discontinuity;
- if raw-channel smoothness and model accuracy remain essentially unchanged, the evidence supports
  genuinely difficult geometry-to-dynamics behavior.

## Stage 5 — spectral endpoint and negative control

The suite stage evaluates H4, LiH, BeH2, N2, and H2O. Metrics per held-out geometry are:

- temporal Pearson and MSE;
- dominant peak energy error;
- matched-line precision/recall within 0.05 Eh;
- dominant versus weak transition recall;
- matched intensity error;
- normalized spectral Wasserstein distance.

H2O is an explicit negative control. Report its endpoint metrics in the same table, not only in
prose. Plot geometry-resolved failures near the N2 crossing in the final report if space permits.

## Architecture audits and variants

The audit records exact signal-matrix singular-value decay, 99%-energy rank, effective learned
frequency count, frequencies above 1% amplitude weight, and one-particle physicality violations
(Hermiticity, trace/particle number, and eigenvalues outside [0,1]).

H4 variants are:

- known initial condition using `D(R,0) + a[cos(wt)-1] + b sin(wt)`; `D(R,0)` is
  computed analytically from the repository-standard broadened-HF state, not read from a held-out
  finite-library target;
- cumulative-softplus ordered frequency bank;
- both changes together;
- K in {32, 64, 128} against released K=256;
- amplitude rank in {4, 8, 32} against released rank=16.

Use the standalone spectral metrics, never trainer `val_corr`, for model selection. Report every
run, including collapse. For the three architecture variants, use training history plus endpoint
metrics to decide whether ordering/anchoring reduces seed instability; do not normalize restarts.

## Required final report

`FSR_CLOSURE_RESULTS.md` should be short enough to review in one sitting and contain:

1. hardware, wall time, code commit, package versions, and any interrupted/restarted stages;
2. geometry learning-curve threshold table and acquisition accounting;
3. blocked/coarse interpolation table, separate from random/even learning curves;
4. screen robustness table and the largest safe perturbation;
5. QA->QA, QA->QB, QA->exact mean +/- std and target-to-target residuals;
6. N2/H2O raw versus aligned smoothness, Pearson, and spectral metrics;
7. five-molecule endpoint table including H2O;
8. structured-line baseline comparison;
9. physicality/SVD/frequency-usage summary and K/rank/architecture variant table;
10. one verdict per closure question: PASS, FAIL, or INCONCLUSIVE, with the numeric reason.

Do not make a quantum-advantage claim. Use “recipe transfer,” not cross-molecule generalization.
Describe the finite-shot count literally. Distinguish line-screen resolution from geometry
smoothness. If a stage fails technically, preserve its log, mark it incomplete, and continue with
independent stages.

## Returning results

```bash
cd /workspace/fermionic-shadow-regressor   # adjust to your clone
git switch -c runpod-results-fsr-closure

# Write FSR_CLOSURE_RESULTS.md.
# Copy only aggregate JSON/PDF artifacts into fsr_closure_results/.
git add FSR_CLOSURE_RESULTS.md fsr_closure_results/
git status --short
git commit -m "results: FSR scientific-closure experiment fleet"
git push -u origin runpod-results-fsr-closure
```

Before committing, confirm `git status` contains no `.h5`, `.pt`, logs, caches, tokens, paper
files, or generated scripts. Report the branch name, commit hash, unfinished stages, and the path
to the full ignored `$OUT` tree on the persistent volume.
