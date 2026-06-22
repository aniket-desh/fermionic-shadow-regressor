# RunPod Briefing #2 — H2O is a negative result; diagnose + re-run at longer horizon

This follows up `RUNPOD_BRIEFING.md`. N2 and BeH2 came back as expected; **H2O came
back RED**, and Local Claude diagnosed why. Your job: confirm the diagnosis with a
cheap pre-flight, then **re-run H2O with a longer training horizon (t_max=2500)**. If
that fixes it, great; if not, stop — we'll decide by hand whether to present it as a
failed case or drop it.

## The diagnosis (what went wrong with the v1 H2O run)

The v1 H2O run (`h2o_regress_v1`, t_max=800, dt=0.4, box [0.7,1.5], floor 2.2) failed
**uniformly across the whole box**, which is *not* the predicted localized
avoided-crossing failure:

- Held-out **mean r̄ = 0.385, median 0.41, all 16 geometries < 0.9** (15 < 0.5),
  range_ratio 1.22 (amplitude overshoot). Failures are scattered across all R in
  [0.75, 1.49], including R≈1.0–1.2 the pre-flight called "clean."
- Training **completed all 150k steps** but val Pearson was stuck ~0.35–0.47 with phase
  errors 0.3–1.96 rad. So the model genuinely could not fit H2O — not a crash.
- **Datagen is healthy**: ω_op = 0.93–2.03 E_h (mean 1.39), HF energies fine, no
  errors/NaNs/zeros. The dataset is good; the model couldn't learn it at this grid.
- **Smoking gun — the in-box coherence heatmap** (`.../h2o_regress_v1_orb_s42_model/plots/coherence_heatmap.pdf`):
  GREEN at early times (t ≲ 120 a.u.) across most R, then **decohering to RED by
  t ≈ 150–250 everywhere**. That time-decay means the model learned *roughly-right but
  imprecise* frequencies — the predicted sinusoids drift out of phase as t grows.
  (Contrast LiH, which stayed green all the way to t=1200: frequencies nailed.)

**Conclusion:** imprecise frequencies ⇒ the dense/congested-spectrum failure mode (same
as the old H2 case), spread across the whole box. The most likely cause is that H2O's
in-box co-dominant Bohr lines are **sub-resolution at t_max=800** (resolution
2π/800 ≈ 0.0079 E_h), and the **coarse 6/15 pre-flight mis-called H2O "clean in
[0.7,1.5]."** Longer horizon → finer frequency resolution → should fix it (this is
exactly how LiH was fixed: its grid went 300→1200).

## Step 1 — confirm the diagnosis (cheap, ~minutes)

Run the EXACT line-spectrum pre-flight on the real H2O in-box geometries and compare the
co-dominant line spacing to the resolution at each horizon. Check `--help` for exact arg
forms; intended usage:

```bash
cd models
python -m fermionic_pipeline.data.line_spectrum_preflight --help
# in-box scan at the FAILED horizon and the PROPOSED horizon:
python -m fermionic_pipeline.data.line_spectrum_preflight --molecule h2o --r 0.7 1.5 --t 800  --dt 0.4
python -m fermionic_pipeline.data.line_spectrum_preflight --molecule h2o --r 0.7 1.5 --t 2500 --dt 0.4
```

**What to look for:** the minimum spacing among co-dominant (≥5% power) lines, `dw_strong`,
and the required horizon `t_need = 2π/dw_strong`. Expectation: at many in-box geometries
`t_need > 800` (so t_max=800 was sub-resolution — confirms the diagnosis) and ideally
`t_need ≤ 2500` (so the longer horizon resolves them). Report the `dw_strong` / `t_need`
range you see. If `t_need` greatly exceeds 2500 even in-box, say so — that means no
feasible horizon fixes it and we go straight to "failed case / drop."

## Step 2 — re-run H2O at t_max=2500 (new tag `h2o_regress_v2`, don't clobber v1)

Recipe is the **v18-orb recipe VERBATIM**; the ONLY change from v1 is the horizon
(t_max 800 → 2500, n_times → 6251 at the same dt=0.4). Keep dt=0.4 (Nyquist ceiling
π/0.4 ≈ 7.85 ≫ content ~2 E_h — resolution, not aliasing, is the problem), box
[0.7,1.5], and **floor 2.2 unchanged** (single-variable test — do NOT also tweak the
floor; if 2500 fails we decide by hand, we don't keep knob-twisting).

```bash
cd models
RESULTS=results/fermionic_pipeline/regression
TAG=h2o_regress_v2
DATA=$RESULTS/$TAG/regression_targets.h5
MDIR=$RESULTS/${TAG}_orb_s42_model
CKPT=$MDIR/regressor.pt
mkdir -p $RESULTS/$TAG

# --- datagen (CPU) : t_max=2500, n_times=6251 ---
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m fermionic_pipeline.data.regression_dataset --output $DATA --molecule h2o \
  --r_start 0.7 --r_end 1.5 --r_step 0.01 --t_max 2500 --n_times 6251 --n_q 500 --n_workers $(nproc)
python -m fermionic_pipeline.data.compute_omega_op --data_path $DATA

# --- train (GPU) : v18-orb verbatim, floor 2.2 ---
python -m fermionic_pipeline.training.regressor_trainer --data_path $DATA --save_dir $MDIR \
  --device cuda --seed 42 --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 \
  --warmup_frac 0.05 --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 \
  --fourier_scale 20.0 --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \
  --adaptive_bandwidth --omega_op_floor 2.2 --soft_omega_floor --explicit_amplitude \
  --amp_rank 16 --grad_clip 1.0 --use_orb_features --standardize_orb_energies \
  --alpha_corr 1.0 --eval_every 2000

# --- eval (GPU) : non-oracle, + in-box coherence heatmap ---
python -m fermionic_pipeline.eval.regressor_eval --data_path $DATA --checkpoint $CKPT \
  --save_dir $MDIR/eval --omega_op_source train-interp --device cuda --ljung_box_p 0.06
python -m fermionic_pipeline.eval.plot_regression --data_path $DATA --checkpoint $CKPT \
  --save_dir $MDIR/plots --omega_op_source train-interp --device cuda --ljung_box_p 0.06
python -m fermionic_pipeline.eval.composition_diagnostic --data_path $DATA --checkpoint $CKPT \
  --save_dir $MDIR/eval --omega_op_source train-interp --device cuda
```

Note: datagen is bigger now (81 geoms × 6251 times) — roughly LiH-scale, expect a couple
hours on the A40 vCPUs. Training is unchanged (~12M params).

## Step 3 — GO / NO-GO

Look at `h2o_regress_v2_orb_s42_model/eval/regressor_eval.json` (aggregate `pearson_mean`
over `results[]`) and the new `plots/coherence_heatmap.pdf`:

- **GREEN (fixed)** — in-box mean r̄ ≳ 0.9 **and** the heatmap stays green across the full
  t∈[0,2500] horizon (no early-green/late-red decoherence). If so: also generate the
  extended in/out-of-box heatmap so it can join the figure set:
  ```bash
  EX=$RESULTS/${TAG}_extrap/regression_targets.h5; mkdir -p $RESULTS/${TAG}_extrap
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  python -m fermionic_pipeline.data.regression_dataset --output $EX --molecule h2o \
    --r_start 0.6 --r_end 2.0 --r_step 0.05 --t_max 5000 --n_times 12501 --n_q 500 --n_workers $(nproc)
  python -m fermionic_pipeline.data.compute_omega_op --data_path $EX
  python -m fermionic_pipeline.eval.extrapolation_heatmap --data_path $EX --checkpoint $CKPT \
    --save_dir $MDIR/plots_extrap --train_r_range 0.7 1.5 --train_t_range 0 2500 \
    --omega_op_source train-interp --train_data_path $DATA --device cuda
  ```
- **STILL RED at t_max=2500** — **STOP. Do not keep increasing t_max or touch the floor.**
  Report the numbers + heatmap + the Step-1 pre-flight output. Local Claude and Aniket
  will decide by hand whether to present H2O as a documented failed case (pre-flight
  under-predicts bent-polyatomic difficulty) or drop it from the study.

## Report back

Push to the existing `runpod-results` branch (eval JSONs + plots + the pre-flight stdout;
NOT the multi-GB `.h5`). In your final message, give: the Step-1 `dw_strong`/`t_need`
range, the v2 `pearson_mean`, whether the heatmap decoherence is gone, and your GO/NO-GO
read. Keep the `h2o_regress_v1` artifacts intact for comparison.
