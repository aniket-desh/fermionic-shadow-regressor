# RunPod Briefing #4 — fix (or rule out) the LiH dissociation-edge spike

LiH's only sub-0.9 held-out geometry is **R = 3.20 Å (the box edge)**: pearson 0.45,
envelope 0.39, phase 0.44 rad. The interior (R ~ 1.13-3.16) is uniformly 0.92-0.9996.
In the cross-molecule overlay this lone edge point shows as a spike.

Hypothesis: it's a **boundary / interpolation artifact** — the non-oracle ω_op ceiling
is interpolated in R and has the thinnest (one-sided) support at the box edge, and the
amplitude trunk generalizes worst at the boundary. If so, the fix is to train on a
**buffer beyond 3.2** so 3.2 becomes interior, and report held-out only on [1.0, 3.2].

BUT first rule out that 3.2-3.4 is a real near-crossing (LiH ionic->covalent character
change toward dissociation). If it is, extending won't help and we trim instead.

## Step 0
`git pull origin dev`  (you already have the grid-save + compute_req from before)

## Step 1 (decider, ~minutes) — pre-flight the extended LiH range
```bash
cd models
python -m fermionic_pipeline.data.line_spectrum_preflight --molecule lih --r 0.8 3.4 --t 1200 --dt 0.25
```
Report the per-R table, focusing on **[3.2, 3.4]** (and [0.8, 1.0]):
- **All PASS** (co-dominant spacing stays resolvable, t_need <= ~1200, n_strong stable)
  -> boundary artifact -> **go to Step 2**.
- **Congestion near 3.2-3.4** (dw_strong -> small, t_need >> 1200, n_strong jumps)
  -> physical near-crossing like N2's -> **STOP and report**; we'll trim to [1.0, 3.1]
  instead (no retrain).

## Step 2 (only if Step 1 is clean; high compute ~5 h) — retrain on a buffered box
New tag `lih_regress_v2`, train box **[0.8, 3.4]** (buffers both the 1.0 and 3.2 edges),
recipe verbatim (floor 0.5). Keep the original `lih_regress_v1` intact.
```bash
RES=results/fermionic_pipeline/regression; TAG=lih_regress_v2
DATA=$RES/$TAG/regression_targets.h5; MDIR=$RES/${TAG}_orb_s42_model; mkdir -p $RES/$TAG
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m fermionic_pipeline.data.regression_dataset --output $DATA --molecule lih \
  --r_start 0.8 --r_end 3.4 --r_step 0.01 --t_max 1200 --n_times 4801 --n_q 500 --n_workers $(nproc)
python -m fermionic_pipeline.data.compute_omega_op --data_path $DATA
python -m fermionic_pipeline.training.regressor_trainer --data_path $DATA --save_dir $MDIR \
  --device cuda --seed 42 --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 \
  --warmup_frac 0.05 --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 \
  --fourier_scale 20.0 --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \
  --adaptive_bandwidth --omega_op_floor 0.5 --soft_omega_floor --explicit_amplitude \
  --amp_rank 16 --grad_clip 1.0 --use_orb_features --standardize_orb_energies \
  --alpha_corr 1.0 --eval_every 2000
python -m fermionic_pipeline.eval.regressor_eval --data_path $DATA --checkpoint $MDIR/regressor.pt \
  --save_dir $MDIR/eval --omega_op_source train-interp --device cuda --ljung_box_p 0.06
python -m fermionic_pipeline.eval.plot_regression --data_path $DATA --checkpoint $MDIR/regressor.pt \
  --save_dir $MDIR/plots --omega_op_source train-interp --device cuda --ljung_box_p 0.06
```
Report:
- held-out mean r̄ **overall** and **restricted to R in [1.0, 3.2]** (the reported range);
- the held-out geometries near R = 3.2 specifically — is the spike gone (r >~ 0.9)?

Push to **runpod-results**: the eval JSON, `plots/coherence_heatmap.pdf` (+ its
`coherence_grid.npz`), `plots/regression_summary.pdf`. Push the v2 dataset + checkpoint to HF.

## What we do with it
- Step 1 clean + Step 2 fixes 3.2 -> `lih_regress_v2` becomes the LiH model for the paper;
  we report held-out on [1.0, 3.2] (now interior), the overlay spike is gone, the LiH
  headline number updates slightly. (We use this checkpoint's heatmaps too, for consistency.)
- Step 1 flags a crossing -> no retrain; trim the reported box to [1.0, 3.1] and note the
  dissociation near-crossing. H2O stays dropped; don't touch it.
