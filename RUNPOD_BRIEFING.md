# RunPod Briefing — molecule-generalization batch (for "RunPod Claude")

You are a Claude Code instance running on a RunPod GPU pod with this repo cloned.
Your job is to run the **molecule-generalization experiments** that are queued (and
stuck behind low-priority 24h jobs) on the Trillium cluster, here on the personal pod
instead. A second Claude instance ("Local Claude") wrote this briefing and is handling
the manuscript; you produce the experimental results.

> **Note:** this public repo intentionally omits the internal research logs
> (`research-logs/`) and the manuscript. Don't go looking for them — this
> briefing is self-contained and has everything you need (grids, recipe, flags,
> scoring, pitfalls). If something seems missing, ask Aniket rather than assume.

## What this project is

The fermionic shadow regressor (FSR) amortizes quantum-generated matchgate-shadow data
into predictions of time-dependent degree-two Majorana observables across a molecular
bond-length scan. One recipe ("v18-orb": explicit amplitude–phase head, rank-16
amplitudes, HF-orbital-energy-conditioned adaptive-bandwidth frequencies) already
transfers across H4 and LiH. We are extending the transfer claim to **N2, BeH2, H2O**,
all at the same 8-qubit / 120-channel CAS(.,4) interface. The architecture is **identical
across molecules** — the *only* per-molecule settings are the sampling grid and the
bandwidth floor, both frozen by a line-spectrum pre-flight (see the table below).

## Your task

Run, for each of **N2, BeH2, H2O**, the full chain:
1. **datagen** (training grid) + `compute_omega_op` — CPU, multiprocessing.
2. **datagen** (extended/coarse grid for the in/out-of-box heatmap) + `compute_omega_op`.
3. **train** the v18-orb recipe (floor per molecule) — GPU, ~12M-param model.
4. **eval**: `regressor_eval` + `plot_regression` + `composition_diagnostic`,
   all with `--omega_op_source train-interp` (non-oracle).
5. **heatmap**: `extrapolation_heatmap` on the extended grid (the in/out-of-box figure).

This is exactly what `slurm/regression_molecule.sh <mol>` does on Trillium; it has been
ported to a no-scheduler runner: **`models/scripts/run_molecules_runpod.sh`**. (LiH and
H4 are already done elsewhere — skip them unless asked. The runner knows the `lih` grid
too if needed.)

## Hardware (why this pod, and how to size it)

The bottleneck is **datagen, which is CPU-bound** (multiprocessing across geometries).
Training is a ~12M-param MLP — it barely uses a GPU and needs <4 GB VRAM. So:

- **Use A40, not H100/A100.** The model is far too small to benefit from an H100; you
  would pay ~6× ($2.89/hr vs $0.44/hr) for no speedup because the limiter is CPU datagen.
- **On RunPod, CPU/RAM is bundled per GPU** (an A40 = **9 vCPU + 50 GB RAM** at $0.44/hr,
  Community). You cannot dial up CPU independently on a GPU pod. The way to get more
  vCPUs (which is what datagen wants) is **more GPUs**: **2× A40 ≈ 18 vCPU + 100 GB RAM +
  2 GPUs for ~$0.88/hr** — the sweet spot. 1× A40 also works (datagen ~2× slower).
- **Disk:** datasets are ~1–2 GB each (≈10 GB total incl. extended grids); checkpoints
  ~50 MB. 50–100 GB volume is plenty; 200 GB is safe.
- **Expected cost:** whole batch ≈ 3–5 h wall-clock on 2× A40 ⇒ **~$3–5 total**.

If you have **2 GPUs**, parallelize by running two subsets at once (see the runner header):
`CUDA_VISIBLE_DEVICES=0 MOLS="n2 beh2" bash ...` and `CUDA_VISIBLE_DEVICES=1 MOLS="h2o" bash ...`.

## Setup

```bash
# 1. Clone (the dev branch has all molecule code + standardized plotting):
git clone -b dev https://github.com/aniket-desh/fermionic-shadow-regressor.git
cd fermionic-shadow-regressor

# 2. Deps. Base RunPod PyTorch image already has torch+CUDA; add the rest.
#    pyscf is MANDATORY (HF orbital-energy features); it is in requirements.txt.
pip install -r requirements.txt

# 3. Sanity-check the env (the runner also does this and fails fast):
python -c "import torch,pyscf,pennylane,h5py,scipy; print('cuda',torch.cuda.is_available())"
```

## Run

Everything runs from `models/`. Use tmux/nohup so it survives disconnects.

```bash
cd models
tmux new -s runpod        # or: nohup ... &
bash scripts/run_molecules_runpod.sh    # default MOLS="n2 beh2 h2o", SEED=42
```

The runner **overlaps each molecule's GPU stage with the next molecule's CPU datagen**, so
the GPU isn't idle during the long datagens. `N_WORKERS` defaults to `nproc`. Per-stage
logs land in `models/runpod_logs/`. To run one molecule or change workers:
`MOLS="n2" N_WORKERS=16 bash scripts/run_molecules_runpod.sh`.

If you prefer to drive stages by hand, read `slurm/regression_molecule.sh` (it is NOT in
this clone — it's Trillium-only) or just read the `datagen()`/`gpu()` functions in the
runner; they hold the exact commands and flags.

## Frozen per-molecule grids (pre-flight 6/15 — do NOT change)

| molecule | active space | train box R (Å) | t_max | dt   | n_times | floor ω_f | note |
|----------|--------------|-----------------|-------|------|---------|-----------|------|
| n2       | CAS(4,4)     | [0.9, 2.3]      | 500   | 0.5  | 1001    | 1.5       | box stops at 2.3 to avoid the R≥2.35 avoided crossing |
| beh2     | CAS(4,4)     | [1.0, 3.0]      | 500   | 0.5  | 1001    | 0.6       | clean throughout (LiH-like) |
| h2o      | CAS(4,4)     | [0.7, 1.5]      | 800   | 0.4  | 2001    | 2.2       | restricted to the clean compression-to-near-eq box (2 interior crossings beyond) |
| lih      | CAS(2,4)     | [1.0, 3.2]      | 1200  | 0.25 | 4801    | 0.5       | already done — green (r̄ 0.97–0.98) |
| h4       | CAS(4,4)     | [0.5, 3.0]      | 300   | 0.05 | 6001    | 8.0       | already done — appendix |

The dense training grid uses `r_step 0.01`; the extended heatmap grid uses `r_step 0.05`
with a wider R range and ~2× t (baked into the runner).

## Success criteria / how to score

**Rank ONLY on held-out temporal Pearson + dominant-peak recovery — never on val loss
or train MSE.** (This is a hard-won lesson; the saturated proxies mislead.)

- After each molecule, read `results/.../<mol>_regress_v1_orb_s42_model/eval/regressor_eval.json`
  → `pearson_mean`. **Green ≈ comparable to LiH/H4's in-box bar (≳0.89).**
- Look at `.../plots_extrap/coherence_heatmap.pdf`: deep green inside the training box;
  the theory *predicts where it should fail* out-of-box — **N2 should degrade for R≳2.35
  (avoided crossing); H2O should degrade where its interior avoided crossings sit
  (R≈1.55–1.65 and 2.10–2.25); BeH2 should be clean.** Mixed results that match these
  predictions are a STRENGTH, not a failure ("the pre-flight predicts which scans are
  learnable and where the surrogate fails").
- `train_mse` will look huge (~1e9). **This is expected** v18b gradient-clipping behavior
  and is decoupled from quality — judge by `pearson_mean` / composition, not MSE.

## Pitfalls (read before running)

- **pyscf must be installed** or the orb recipe aborts (`validate_orb_features()` errors on
  all-zero HF energies). The runner's preflight check catches this.
- **Datagen sets `OMP_NUM_THREADS=1`** etc. so the worker processes don't oversubscribe
  BLAS — the runner does this; don't remove it.
- **Eval must use `--omega_op_source train-interp`** (non-oracle): the bandwidth ceiling at
  held-out geometries is interpolated from training geometries, never read from the
  held-out signal. Fresh checkpoints store `train_omega_op`, so this is non-oracle from the
  start (no separate audit needed).
- Sanity-check that datagen's printed `omega_op` range roughly matches the pre-flight
  bandwidth for each molecule (n2 ~0.7–1.3, beh2 ~0.33–0.43, h2o ~0.48–2.03 Eₕ). A big
  mismatch means the wrong molecule/grid.

## Reporting results back

`results/` is gitignored, so don't expect a normal `git add`. When done:

1. **Report the numbers in your final message**: per molecule, `pearson_mean`, whether it's
   green, and whether the extended heatmap matches the predicted failure zones.
2. **Push the lightweight artifacts** so Local Claude / Aniket can fetch them — eval JSONs
   and the plot PDFs only (NOT the multi-GB `.h5` datasets):
   ```bash
   git checkout -b runpod-results
   git add -f models/results/fermionic_pipeline/regression/*_regress_v1_orb_s42_model/eval/*.json \
               models/results/fermionic_pipeline/regression/*_regress_v1_orb_s42_model/plots*/*.pdf
   git commit -m "RunPod results: N2/BeH2/H2O eval JSONs + coherence heatmaps"
   git push -u origin runpod-results
   ```
3. If you hit a red molecule, don't hill-climb the architecture — diagnose with the
   composition diagnostic + the line-spectrum pre-flight and report; the recipe is meant to
   be frozen.

## Claim framing (keep honest)

The result is "the **same** FSR recipe transfers across fixed-active-space, 1-D molecular
scans under a theory-guided pre-flight grid" — **not** universal/arbitrary-geometry
generalization (still per-molecule retrain, 1 geometric dof each). CAS(4,4) for N2/H2O is a
shared 8-qubit benchmark interface, not converged chemistry.
