# RunPod Briefing #3 — LiH extended coherence heatmap + r(R,t) grid dumps

We're assembling the preprint figures. LiH is the headline. Two gaps only RunPod can fill:
1. **LiH's extended (in/out-of-box) coherence heatmap** — LiH only ever got the in-box
   heatmap; the `lih --extrap` stage was never run. We need the bracketing version
   (like H4's) for the appendix per-molecule panel.
2. **The raw r(R,t) grid arrays** for LiH, N2, BeH2 — for a later cross-molecule
   *averaged* coherence heatmap (normalize R by R_eq, t by horizon, then average).
   None of the datasets/checkpoints exist off-pod, so this has to happen where they live.

## Step 0 — pull the grid-save update (REQUIRED)
```bash
cd <repo> && git pull origin dev    # must include commit 4996287
```
That commit makes `plot_coherence_heatmap` also write **`coherence_grid.npz`** (the
`corr` grid + `R`/`t` axes + train ranges) next to every `coherence_heatmap.pdf`.
Without it you'll get PDFs but no grids.

## Step 1 (primary) — regenerate LiH end-to-end (gives both heatmaps + grids)
```bash
cd models
MOLS="lih" bash scripts/run_molecules_runpod.sh
```
The `lih` grid is already baked into the runner (train R[1.0,3.2] t_max1200; extended
R[0.8,3.6] t_max2400). This reproduces LiH from seed 42 and produces, under
`results/fermionic_pipeline/regression/lih_regress_v1_orb_s42_model/`:
- `plots/coherence_heatmap.pdf` + `plots/coherence_grid.npz`  — **in-box** (headline figure)
- `plots_extrap/coherence_heatmap.pdf` + `plots_extrap/coherence_grid.npz` — **in/out-of-box** (appendix)
- `eval/regressor_eval.json`

**Report the in-box mean `pearson_mean` (expect ≈ 0.979).** This is our headline number,
so flag it if it drifts by more than ~0.01 from 0.979 (different GPU/numerics can nudge
it slightly; a big drift means something's off). We'll use *this* re-run's heatmaps for
BOTH the main (in-box) and appendix (extended) LiH figures so they share one checkpoint.

## Step 2 (secondary, cheap) — dump N2 + BeH2 extended grids
You still have their checkpoints + extended datasets from the earlier batch, and the
grid-save is now automatic, so just re-run the extended-heatmap step for each so a
`coherence_grid.npz` lands next to their existing `plots_extrap/coherence_heatmap.pdf`:
```bash
for m in n2 beh2; do
  RES=results/fermionic_pipeline/regression
  python -m fermionic_pipeline.eval.extrapolation_heatmap \
    --data_path $RES/${m}_regress_v1_extrap/regression_targets.h5 \
    --checkpoint $RES/${m}_regress_v1_orb_s42_model/regressor.pt \
    --save_dir $RES/${m}_regress_v1_orb_s42_model/plots_extrap \
    --train_r_range <m's train R range> --train_t_range 0 <m's train t_max> \
    --omega_op_source train-interp \
    --train_data_path $RES/${m}_regress_v1/regression_targets.h5 --device cuda
done
```
(Train ranges from the runner's `grid()`: n2 = R[0.9,2.3] t_max 500; beh2 = R[1.0,3.0] t_max 500. If the extended datasets were cleaned, regenerate just them with the runner's `datagen` block — but no need to retrain.)

## Step 3 — push results back
- **`runpod-results` branch** (`git add -f`): all the new `coherence_heatmap.pdf` **and
  `coherence_grid.npz`** for lih (plots/ + plots_extrap/), n2 (plots_extrap/), beh2
  (plots_extrap/); plus lih's `regression_summary.pdf` and `eval/regressor_eval.json`.
  Do NOT push the multi-GB `.h5`.
- **HuggingFace**: push the LiH **training + extended `.h5` datasets** and the LiH **s42
  checkpoint** (`regressor.pt`) so we can reuse them without a re-run. Report the HF paths.
- In your final message: the LiH in-box r̄ (vs 0.979), confirmation the extended heatmap
  looks right (deep green across [1.0,3.2]×[0,1200], behaviour degrading out beyond the
  box in R∈[0.8,3.6]), and the pushed paths.

## Not needed here
H4's grid is the only remaining piece for the averaged heatmap, and H4 isn't on this pod
(its checkpoint is on HuggingFace at `aniketdesh/molecular-shadows-h4`). We'll handle H4
separately — **don't re-run H4.** H2O is dropped from the paper; ignore it.
