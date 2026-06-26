# RunPod Briefing #8 — dipole-BO resource experiment on BeH2

## Why BeH2 (and why LiH was the wrong pick)
The LiH dipole-BO came back **0/26 within 5% under the fair (dynamic) metric** — a genuine
negative. LiH is heteronuclear: its large permanent dipole confounds the relative-RMS metric, and
its dynamic dipole is a small high-frequency signal, so the zero-sample prior never clears DFT
tolerance even though the observables transfer at r̄ $=0.992$. So LiH is out of the resource
experiment.

**BeH2 is the right non-H4 candidate.** It is **centrosymmetric (linear symmetric, zero permanent
dipole)**, exactly like H4, so the dipole is purely dynamic and the metric is honest with no
mean-subtraction trick. Its observable transfer is clean (r̄ $=0.997$). If a non-H4 molecule shows
H4-like zero-sample gains, this is it. Luis wants a second molecule for the resource figure (all
four are 8 qubits, so there is no "larger" example to show); BeH2 is the honest choice.

**Be honest with the numbers.** If BeH2 also gives few or zero zero-sample successes, report that
plainly — we will keep H4 as the demonstration and note the molecule-dependence. Do not tune to a
positive.

## Reuse from #7 (no new code expected)
`compute_dipole_coeffs.py` is already registry-generalized (Step 1 of #7) and `dipole_bo.py` is
molecule-agnostic. BeH2 uses the same active-space path as LiH. Because BeH2's permanent dipole is
~0, the standard and `--dynamic` metrics should be nearly identical; run the standard metric and
note the c0 (permanent-dipole) magnitude as a check.

## Step 1 — self-test (gate; cheap)
```bash
cd models
python3 -m scripts.bo.compute_dipole_coeffs --self_test --molecule beh2 \
  --data_h5 results/fermionic_pipeline/regression/beh2_regress_v1/regression_targets.h5
```
Must print SELF-TEST PASSED (all xyz |Δ|<1e-7 reconstructing ⟨μ(0)⟩). For centrosymmetric BeH2 the
permanent-dipole constant c0 should be ≈0; flag it if it isn't. Do not proceed unless it passes.

## Step 2 — dipole coeffs
```bash
python3 -m scripts.bo.compute_dipole_coeffs --molecule beh2 \
  --data_h5 results/fermionic_pipeline/regression/beh2_regress_v1/regression_targets.h5 \
  --out    results/fermionic_pipeline/regression/beh2_regress_v1/dipole_coeffs.npz
```

## Step 3 — dipole-BO sweep (non-oracle prior)
```bash
B=results/fermionic_pipeline/regression
python3 -m scripts.bo.dipole_bo --molecule BeH2 \
  --npz        $B/beh2_regress_v1/dipole_coeffs.npz \
  --data_h5    $B/beh2_regress_v1/regression_targets.h5 \
  --checkpoint $B/beh2_regress_v1_orb_s42_model/regressor.pt \
  --save_dir   $B/beh2_regress_v1/bo \
  --omega_op_source train-interp \
  --sweep_lo 1.0 --sweep_hi 3.0 --n_sweep 26 \
  --rel_error 0.05 --max_samples 40 --device cuda
```
(If you want a side check, add `--dynamic`; for BeH2 it should barely change anything.)

## Step 4 — report + commit (results branch)
Commit to `runpod-results`: `beh2_regress_v1/dipole_coeffs.npz`, `beh2_regress_v1/bo/`
(`dipole_bo_sweep_beh2.pdf` + plotdata pkl). Paste back:
- Self-test result (PASSED) and the c0 magnitude (expected ≈0).
- **Zero-sample relative error vs R**: how many of the 26 geometries clear the 5% DFT-level
  tolerance with ZERO samples, the min error and its R, and where the 40-sample budget saturates.
- A one-line verdict vs H4 (which is 7/26 zero-sample): does BeH2 give a comparable mid-bond
  zero-sample region, or not?

## Code-to-dev (Aniket's call)
The #7 code changes (registry-generalized `compute_dipole_coeffs.py` + `dipole_bo.py --dynamic`)
are reusable infra. You flagged wanting Aniket's OK before pushing to `dev` — that still stands;
keep results on `runpod-results`, hold the code push until Aniket confirms.
