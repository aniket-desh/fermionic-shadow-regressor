# RunPod Briefing #6 — H4 coherence grid (for the 4-molecule averaged heatmap)

## Goal
Produce **H4's** extended-grid `coherence_grid.npz` (windowed Pearson `r(R,t)`), the last
gap for the cross-molecule averaged coherence heatmap. N2/BeH2/LiH grids are already on the
`runpod-results` branch under `…/<mol>_regress_v1_orb_s42_model/plots_extrap/coherence_grid.npz`.
H4's npz is missing only because the `np.savez` dump postdates H4's original Trillium plots —
the model itself is final (v18-orb, seed 42), so **do NOT retrain H4**.

## Why this is cheap (read before sizing anything)
The only thing `train-interp` (non-oracle ω_op) needs from the H4 *training* set is its per-geometry
ω_op ceiling table — so **pull the trained H4 bundle from HF instead of regenerating the 6.8h fine
grid.** The HF uploader bundles `regressor.pt` **and** the training `regression_targets.h5` (with the
`omega_op` field) in the same repo. The ONLY fresh datagen is the coarse *extended* eval set (~tens
of minutes). No retrain, no fine regen. A modest GPU + normal CPU pod is plenty.

## H4 grid (from the checkpoint payload — verified)
- **train:** R ∈ [0.50, 3.00], step 0.01 (n=251), t_max=300, n_times=6001 (dt=0.05), seed 42
- **extended (this run):** R ∈ [0.40, 3.30], step 0.05, t_max=600, n_times=1201 (dt=0.5), N_Q=500
  - mirrors the other molecules: R a bit past train both ends, t_max doubled, dt=0.5

---

## Step 0 — get the trained H4 bundle from HF
`snapshot_download` the H4 v18-orb repo (likely `aniketdesh/molecular-shadows-h4`; `list_models`
on your `aniketdesh` org if the exact id/revision is unclear). You need two files from it:
- `regressor.pt`            → place at `results/fermionic_pipeline/regression/h4_regress_v1_orb_s42_model/regressor.pt`
- `regression_targets.h5`   → place at `results/fermionic_pipeline/regression/h4_regress_v1/regression_targets.h5`

**Sanity-check the training .h5 matches the checkpoint** (else ω_op won't line up):
```python
import h5py, numpy as np
f = h5py.File("results/fermionic_pipeline/regression/h4_regress_v1/regression_targets.h5")
R = f["R_values"][:]; t = f["times"][:]
print("R", len(R), R.min(), R.max(), "| t", len(t), t.min(), t.max(), "| omega_op?", "omega_op" in f)
# expect: R 251 0.50 3.00 | t 6001 0.0 300.0 | omega_op? True
```
If `omega_op? False`, run `python -m fermionic_pipeline.data.compute_omega_op --data_path <that .h5>`.
If the repo doesn't carry the .h5, regenerate the training set with the **train grid above** (this is
the only case where you pay the fine-grid datagen — avoid if HF has it).

## Step 1 — extended eval datagen (the only datagen; coarse + fast)
```bash
cd models
EX=results/fermionic_pipeline/regression/h4_regress_v1_extrap/regression_targets.h5
mkdir -p results/fermionic_pipeline/regression/h4_regress_v1_extrap
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m fermionic_pipeline.data.regression_dataset --output "$EX" --molecule h4 \
  --r_start 0.40 --r_end 3.30 --r_step 0.05 --t_max 600 --n_times 1201 \
  --n_q 500 --n_workers $(nproc)
python -m fermionic_pipeline.data.compute_omega_op --data_path "$EX"
```

## Step 2 — coherence heatmap + grid dump (non-oracle, matches the other 3)
```bash
MDIR=results/fermionic_pipeline/regression/h4_regress_v1_orb_s42_model
python -m fermionic_pipeline.eval.extrapolation_heatmap \
  --data_path results/fermionic_pipeline/regression/h4_regress_v1_extrap/regression_targets.h5 \
  --checkpoint "$MDIR/regressor.pt" \
  --save_dir   "$MDIR/plots_extrap" \
  --train_r_range 0.50 3.00 --train_t_range 0 300 \
  --omega_op_source train-interp \
  --train_data_path results/fermionic_pipeline/regression/h4_regress_v1/regression_targets.h5 \
  --device cuda
```
This writes `$MDIR/plots_extrap/coherence_grid.npz` (keys: `corr, R, t, train_R_range, train_t_range`)
and `coherence_heatmap.pdf`.

## Step 3 — commit + report
- Commit to the **results branch**: `…/h4_regress_v1_orb_s42_model/plots_extrap/coherence_grid.npz`
  and `coherence_heatmap.pdf` (the npz is the one Local Claude actually needs).
- Paste back: the printed **in-box / out-of-box r̄**, and the npz axes sanity
  (`R≈[0.40,3.30]`, `t` up to ~590, `corr` shape ≈ (59, ~119)).
- Flag anything odd (e.g. ω_op interp range warnings, or in-box r̄ far from H4's known ~0.92).

That's all — no retrain, no GPU-heavy work. Local Claude drops the npz into
`plot_coherence_average.py` and builds the 4-molecule averaged heatmap on a common
(R/R_eq, t/t_train) axis.
