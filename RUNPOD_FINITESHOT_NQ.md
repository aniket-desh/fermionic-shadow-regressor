# RunPod Briefing #10 — last experiments: finite-shot (BLOCKING E1) + N_Q sweep (E4)

The pod is idle and the approaches are ready (your handoff). These two are the last evidence
gaps. **Autonomous: run both, self-verify, push per-block to `runpod-results`, don't wait.**
Order: **3A first** (it closes the one remaining BLOCKING review item), then 2D.

Standing conventions (same as B9): score held-out `pearson_mean` non-oracle
(`--omega_op_source train-interp`) + dominant-peak recovery; new code stays on `runpod-results`
pending Aniket review; token hygiene. Use the COARSE grids below (NOT the fine training grids —
fine-grid datagen is the ~14h blocker we're avoiding).

Coarse grid per molecule (reuse `scripts/revision/train_eval.sh` + `fleet_runner.sh`):
- **h4:** R[0.5,3.0] step 0.05, t_max 300, n_times 1001, floor 8.0
- **lih:** R[1.0,3.2] step 0.05, t_max 1200, n_times 1001, floor 0.5
(20% held-out split, seed 42; verbatim v18-orb recipe otherwise.)

---

## 3A — finite-shot / shot-noise full study (BLOCKING E1)  [do FIRST]
The sampler is already validated (`finite_shot_convergence.py`, 1/√S confirmed). Wrap that
empirical-marginal sampler into the coarse-grid datagen and run the training study.
- For h4 + lih, generate targets at **S ∈ {10, 100, 1000}** shots (the empirical-marginal path: draw
  S Born samples per (Q, R, t) from the rotated-state probabilities, accumulate the matchgate
  estimator, average over the N_Q=500 library — NO conditional storage). Also keep the **S=∞**
  (exact-marginal) coarse-grid target as the noiseless reference.
- Train the verbatim v18-orb FSR on each S, eval held-out `pearson_mean` + dominant-peak recovery.
- **Output:** `finite_shot_<mol>.json = {S: {pearson_mean, peak_recovery}}` for h4, lih, and a
  `pearson-vs-S` degradation figure. Commit to `runpod-results`.
- **Self-checks:** (i) re-confirm the S=∞ coarse-grid result is in the ballpark of the fine-grid
  number (h4 ≈0.93-ish on the coarser grid, lih ≈0.98); (ii) expect MONOTONE degradation as S
  drops (S=10 worst). If S=∞ coarse ≉ fine, note the grid effect separately from the shot effect.
- **Honest reporting:** this is the evidence that the pipeline survives (or doesn't) realistic
  shot noise. Report the real degradation, whatever it is — if S=10 tanks it, that's a finding
  (and informs how many shots the protocol actually needs).

## 2D — N_Q sensitivity sweep (E4)  [do SECOND]
For h4 + lih on the coarse grid: regenerate targets at **N_Q ∈ {50, 100, 250, 500, 1000}**
(the finite-library size; the target depends on it), retrain (seed 42), eval.
- **Output:** `nq_sweep_<mol>.json = {n_q: {pearson_mean, peak_recovery}}` + a `pearson-vs-N_Q`
  figure. Commit to `runpod-results`.
- **Self-check:** roughly monotone improvement with N_Q; the N_Q=500 point should match the
  existing coarse-grid result.

## 3G — TDHF/CIS dipole baseline (E2.5, optional)
Skip unless the above finish fast and it's quick in PySCF (CIS dipole dynamics for h4+beh2 vs the
FSR prior). Not required.

## Report back
Append a `FINITE_SHOT_NQ_RESULTS.md` on `runpod-results`: the pearson-vs-S curve (the headline
E1 evidence) + the pearson-vs-N_Q curve, with the self-check notes. Flag honestly if shot noise or
small N_Q breaks the pipeline. New code (the `--shots` datagen path) stays on `runpod-results`,
tagged pending Aniket review.
