# RunPod Briefing #9 — peer-review revision experiment fleet (AUTONOMOUS)

## Context
The FSR preprint got a major-revision peer review. Decision = **HYBRID**: reframe the claims
honestly AND add the evidence that is cheap on 8 qubits. This briefing is that evidence fleet.
Full review register: `research-logs/peer-review-2026-06-26.md` (issue IDs like E1/E2/E4/T4 are
referenced below so your commits map to it).

## Autonomy mandate (important)
**Work through the whole list without waiting for Aniket between blocks.** As long as results are
correct (use the per-block self-checks), just run each block and **push to the `runpod-results`
branch as you finish it**, with a commit message tagging the issue ID. Only stop for a genuine
blocker: broken env, a spec you truly can't resolve, or results wildly off the known baselines
(then push what you have + a note and move on to the next independent block). **Be honest** — if a
baseline beats FSR or an ablation shows a component doesn't help, report it plainly; that is
exactly the evidence the review wants. Do not tune toward a positive.

## Standing conventions (from prior briefings)
- **Score only** held-out `pearson_mean` (held-out temporal Pearson) + dominant-peak recovery.
  NEVER rank by val loss / val corr (saturated proxy — see architecture-log). Save ckpt + history
  + eval JSON per run.
- **Eval is non-oracle:** `--omega_op_source train-interp` everywhere.
- **Results → `runpod-results`.** NEW CODE (finite-shot mode, baseline scripts, diagnostics) you
  can commit to `runpod-results` so it's not lost, but **do NOT push code to `dev`** — flag it
  "code pending Aniket review" and he'll port the good parts. (Briefing #7/#8 dipole code is in
  the same holding pattern.)
- **Token hygiene:** `git status` for stray HF-token files before any `git add`; never commit
  `hf.txt`/`*tok*`.
- Funding/acknowledgement placeholder is **Aniket's job** — ignore it.

## Frozen per-molecule grids + floors (from `scripts/run_molecules_runpod.sh` `grid()`)
| mol | R range | step | t_max | n_times | floor ω_f |
|---|---|---|---|---|---|
| h4   | [0.5, 3.0] | 0.01 | 300  | 6001 | 8.0 |
| lih  | [1.0, 3.2] | 0.01 | 1200 | 4801 | 0.5 |
| beh2 | [1.0, 3.0] | 0.01 | 500  | 1001 | 0.6 |
| n2   | [0.9, 2.3] | 0.01 | 500  | 1001 | 1.5 |

The v18-orb training command (verbatim recipe) is the `gpu()` block of `run_molecules_runpod.sh`:
`--steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 --warmup_frac 0.05 --weight_decay 5e-4
--d_hidden 768 --n_layers 6 --n_fourier 256 --fourier_scale 20.0 --conditioned_frequencies
--freq_net_hidden 128 --freq_net_layers 3 --adaptive_bandwidth --omega_op_floor <FL>
--soft_omega_floor --explicit_amplitude --amp_rank 16 --grad_clip 1.0 --use_orb_features
--standardize_orb_energies --alpha_corr 1.0`. Reuse the existing per-molecule `.h5` datasets
(pull from HF or regenerate from the grid above if not on the pod).

---

# TIER 1 — foundational stats + ablations (highest value; reuses existing infra)

## 1A — Multi-seed statistics (E4)
For **each** molecule {h4, lih, beh2, n2}, retrain the verbatim v18-orb recipe at **5 seeds**
`{42, 1729, 7, 13, 101}` (reuse the same `.h5`; only `--seed` changes), eval each (non-oracle).
- **NOTE on splits:** `--seed` seeds BOTH weight init AND the random 20% geometry split (there is
  no separate split-seed), so 5 seeds already gives 5 distinct geometry splits — this covers both
  the seed-variance AND geometry-split-variance the reviewer asked for, in one set. State that
  explicitly in your report. (Optional, if quick: add a `--split_seed` flag to decouple them and
  run 3 split-seeds at fixed init-seed 42; only if it's a clean small change.)
- **Output:** one `stats_<mol>.json` = `{seed: pearson_mean, median, peak_recovery}` + computed
  `mean ± std`. Commit all 20 eval JSONs + the 4 stats JSONs.
- **Self-check:** each run should land near the known single-seed value (h4 ≈0.93, lih ≈0.99,
  beh2 ≈0.997, n2 mean ≈0.91 / median ≈0.99). Flag any seed that deviates by >0.05.

## 1B — Component ablations (E2.6 / T4)
For each molecule (seed 42), retrain with ONE component removed at a time, eval held-out:
- **no-orb:** drop `--use_orb_features --standardize_orb_energies`.
- **no-adaptive-bw:** drop `--adaptive_bandwidth --soft_omega_floor` (fixed ceiling).
- **no-lowrank:** set `--amp_rank 120` (full rank = N_o; removes the low-rank channel sharing).
- **Output:** `ablation_<mol>.json` = `{full, no_orb, no_adaptive_bw, no_lowrank: pearson_mean}`.
- **Self-check:** no-orb should HURT most at short bond (the known orb-improvement region); if it
  doesn't change anything, the flag isn't taking effect — investigate before reporting.
- If the `validate_orb_features()` guard errors when orb features are off, that's expected for the
  no-orb run — pass whatever fallback feature the model uses without orb energies (bare R), or add
  a `--no_orb_baseline` path; note what you did.

## 1C — Linear-harmonic baseline (E2.3 — the cheapest, most pointed baseline)
Tests whether the LEARNED frequencies beat a FIXED pre-flight bank. New script
`scripts/baselines/linear_harmonic.py`:
- For each training geometry, least-squares fit per-channel `{a_μk, b_μk, c_μ}` to a FIXED
  frequency bank = the molecule's exact dominant Bohr lines (from
  `line_spectrum_preflight` / diagonalization) or a uniform bank up to ω_op.
- Interpolate the fitted amplitudes in R (1-D) to held-out geometries; reconstruct
  `D̂(R*,t) = Σ a cos + b sin + c`; eval held-out `pearson_mean` with the SAME metric as the FSR.
- **Output:** `baseline_linharm_<mol>.json` (held-out pearson_mean) for all 4 molecules.
- **Self-check:** in-sample (training-geometry) fit should be near-perfect if the bank covers the
  lines; if not, widen the bank. Report held-out vs FSR side by side.

---

# TIER 2 — neural / classical surrogate baselines + N_Q sweep

## 2A — Fourier-feature MLP baseline (E2.2)
New `scripts/baselines/fourier_mlp.py`: a generic coordinate network, inputs = (HF orbital-energy
features ε(R), t) → random Fourier features → MLP → 120 channels, trained on the SAME data/split
as FSR at a **matched parameter budget** (~12M). Same held-out metric. This isolates whether the
FSR's explicit harmonic structure beats a generic neural surrogate with identical inputs.
- Output `baseline_fourmlp_<mol>.json`. Self-check: it should fit training geometries; report
  held-out vs FSR.

## 2B — GP / KRR over (R,t) (E2.1)
New `scripts/baselines/gp_krr.py`: per-channel (or PCA-reduced over the 120 channels) KRR/GP
regression over (R,t) → observable (sklearn). Subsample (R,t) if memory-bound. Held-out
pearson_mean. Output `baseline_gpkrr_<mol>.json`.

## 2C — Per-geometry Prony/ESPRIT/FFT (E2.4)
New `scripts/baselines/spectral_classical.py`: at each held-out geometry, given the exact time
series, run Prony/ESPRIT (or FFT peak-pick) and compare recovered dominant frequencies/peaks to
exact. This is the "what can classical spectral estimation do with the same time data" baseline.
Report **dominant-peak recovery rate** per molecule. Output `baseline_spectral_<mol>.json`.

## 2D — N_Q sensitivity (E4) — H4 + LiH only (representative fast + slow)
For h4 and lih: regenerate the dataset at `--n_q ∈ {50,100,250,500,1000}` (the target depends on
N_Q), retrain (seed 42), eval. Report held-out pearson_mean + peak recovery vs N_Q.
- Output `nq_sweep_<mol>.json` = `{n_q: pearson_mean, peak_recovery}`. Self-check: monotone-ish
  improvement with N_Q expected; the N_Q=500 point should match the existing result.

---

# TIER 3 — finite-shot study (BLOCKING E1) + diagnostics

## 3A — Finite-shot / shot-noise study (E1) — H4 + LiH
The datagen stores the **exact conditional probabilities `p(b|Q,R,t)`**. So finite-shot needs NO
re-diagonalization: **resample** from those stored conditionals.
- Add a finite-shot target mode (a `--shots S` path, or a standalone
  `scripts/data/finite_shot_targets.py`): for each (Q-label, R, t), draw `S` Born samples from the
  stored `p(b|Q,R,t)`, accumulate the matchgate shadow estimator empirically, average over the
  N_Q labels → a NOISY target `D̂_S(R,t)`.
- Generate finite-shot datasets at `S ∈ {10, 100, 1000}` (+ reuse the existing S=∞ infinite-shot).
- Train FSR on each, eval held-out pearson_mean + dominant-peak recovery (+ dipole zero-sample
  count for h4 if cheap). Show the degradation curve vs S.
- **Self-check (critical):** as `S → ∞` the resampled targets must converge to the stored
  infinite-shot targets — verify at a handful of (Q,R,t) points (mean abs diff → 0 like 1/√S)
  BEFORE training. This validates the sampler.
- Output `finite_shot_<mol>.json` = `{S: pearson_mean, peak_recovery}` + the convergence check.

## 3B — Frequency-recovery diagnostics (T4) — all molecules
New `scripts/diagnostics/freq_recovery.py`: from each molecule's trained FSR, extract the learned
`ω_k(R)`; compare to the exact dominant Bohr lines (diagonalization / `line_spectrum_preflight`)
across the scan; quantify match (e.g. nearest-line distance) and plot **phase error vs line
spacing δω** (show error rising where lines crowd). Dump `freq_recovery_<mol>.npz` + a figure.

## 3C — Smoothness diagnostic (T2) — all molecules (+ H2O)
New `scripts/diagnostics/smoothness.py`: finite-difference norms `‖∂_R ⟨Γ_μ⟩‖`, `‖∂_t ⟨Γ_μ⟩‖`
across each molecule's scan from the exact targets; highlight the **N2 avoided crossing** and the
**failed H2O** stretch (regenerate a small H2O grid if needed). Dump arrays + figure. This is the
empirical version of Proposition 1's smoothness constant the reviewer wants.

## 3D — Pre-flight diagnostic arrays (W3) — all molecules
Run `line_spectrum_preflight` per molecule across its R scan; dump `ω_op(R)`, the co-dominant line
spacing `δω(R)`, and the congestion threshold as `preflight_<mol>.npz`/JSON. (Local Claude is
building the main-text pre-flight figure; your per-molecule arrays are the source / cross-check —
please push them early.)

## 3E — Scalable-pre-flight proxy ablation (T3) — H4 + LiH
Recompute `ω_op` from a CHEAP proxy (e.g. HF orbital-gap-based bandwidth, no full active-space
diagonalization) and compare to the oracle `ω_op(R)`. If the proxy is close, optionally retrain
one molecule with the proxy ceiling to show the performance delta. Output `proxy_preflight.json`
+ a short note on whether a cheap proxy could replace the oracle pre-flight.

## 3F — Scaling-table data (E3)
Record per molecule: qubits (8), active-space (Hilbert) dim, N_T, N_R, N_Q, datagen wall, train
wall, peak GPU memory. Most wall-times are in the runtime log; fill the gaps + measure peak mem.
Dump `scaling_table.json`.

## 3G — Chemistry dipole baseline (E2.5) — OPTIONAL / stretch, H4 + BeH2
If time permits: a TDHF/CIS (PySCF) dipole-dynamics baseline in the same active space to compare
against the FSR dipole prior (exact diag = ground truth; CIS = the actual cheap-method baseline).
Only if the others are done; skip if it balloons.

---

## Report back (one summary at the end, plus per-block commits as you go)
A `REVISION_RESULTS.md` on `runpod-results` collecting: the multi-seed mean±std table (1A), the
ablation deltas (1B), every baseline's held-out number vs FSR (1C/2A/2B/2C), the N_Q curve (2D),
the finite-shot degradation curve + convergence check (3A), and pointers to the diagnostic
npz/figures (3B–3F). Flag honestly anything that undercuts the FSR (a baseline winning, a useless
component, bad finite-shot robustness). Keep new code on `runpod-results` only, tagged "pending
Aniket review for dev."

Rough scale: Tier 1 ≈ 20 retrains + 12 ablation runs + 1 baseline script; Tier 2 ≈ 3 baseline
scripts + 10 N_Q runs; Tier 3 ≈ finite-shot path + 6 finite-shot runs + 4 diagnostics. Prioritize
top-down; partial completion still delivers the most important evidence (1A/1B/1C and 3A).
