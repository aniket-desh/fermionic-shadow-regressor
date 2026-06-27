# FSR peer-review revision evidence (Briefing #9) — results summary

All held-out scores are **temporal Pearson `pearson_mean`** (mean over the 120 channels, then
over the held-out geometries), **non-oracle** (`--omega_op_source train-interp`), same metric as
the FSR. `--seed` seeds BOTH weight init AND the 20% geometry split. **New code lives on
`runpod-results` only — PENDING ANIKET REVIEW for `dev`.** Honest reporting throughout: where a
baseline matches the FSR or a seed fails, it is stated plainly.

## TIER 1

### 1A — multi-seed statistics (E4)  [DONE]
5 seeds {42,1729,7,13,101} per molecule, verbatim v18-orb. `pearson_mean` mean ± std:

| mol | mean ± std | per-seed (pearson_mean) | read |
|---|---|---|---|
| h4   | 0.881 ± 0.154 | 42:.954 1729:.967 7:.951 **13:.573** 101:.959 | 4/5 ≈0.96; seed 13 degrades |
| lih  | **0.985 ± 0.006** | .973–.991 | seed-robust |
| beh2 | **0.978 ± 0.010** | .972–.997 | seed-robust |
| n2   | 0.754 ± 0.381 | 42:.969 1729:.976 7:.893 **13:−.005** 101:.935 | seed 13 COLLAPSES (flat); other 4 median ≈0.99 |

**Honest finding:** lih/beh2 are seed-robust; **seed 13 triggers a training-stability failure for
n2 (collapse, pearson ≈0) and h4 (0.573)** while leaving lih/beh2 unaffected. Excluding the
collapsed seed, the other 4 seeds give per-geometry median ≈0.99 for n2. This is the seed- AND
geometry-split-variance the review asked for, in one sweep. (20 eval JSONs + `1A/stats/stats_<mol>.json`.)

### 1B — component ablations (E2.6/T4)  [DONE]
seed 42, one component removed. Δ pearson_mean vs full:

| ablation | h4 | lih | beh2 | n2 |
|---|---|---|---|---|
| no-adaptive-bw | −0.041 | **−0.990** | **−0.998** | **−0.971** |
| no-orb | **−0.116** | +0.002 | −0.001 | **−0.165** |
| no-lowrank (rank 120) | **−0.534** | +0.002 | −0.000 | −0.031 |

**Honest finding:** every component is essential for ≥1 molecule — none is dead weight.
Adaptive bandwidth is critical for ALL active-space molecules (collapse without it); orb features
help h4 and n2; low-rank amplitude sharing is critical for h4's dense high-frequency spectrum.
(`1B/stats/ablation_<mol>.json`.)

### 1C — linear-harmonic baseline (E2.3)  [DONE]
Fixed uniform frequency bank + per-channel LSQ amplitudes interpolated in R (in-sample ≈1.0 → bank complete):

| mol | linear-harmonic held-out | FSR (1A mean) | read |
|---|---|---|---|
| n2   | 0.441 | 0.75–0.97 | FSR clearly better |
| h4   | 0.531 | 0.88 | FSR clearly better |
| lih  | 0.963 | 0.985 | FSR slightly better |
| beh2 | 0.999 | 0.978 | **baseline matches/beats FSR** |

**Honest finding:** the FSR's learned, geometry-conditioned frequencies clearly beat a fixed bank
for the harder molecules (n2, h4); for smoothly-varying beh2/lih a fixed-bank + amplitude-interp
baseline is competitive (and matches FSR on beh2).

## TIER 2

### 2B — GP/KRR over (R,t) (E2.1)  [DONE]
RBF KRR (PCA-reduced channels) held-out: **n2 0.086, h4 0.074, lih 0.149, beh2 0.086.** Generic
kernel regression cannot represent the oscillatory signal — the FSR's explicit harmonic head is essential.

### 2C — classical per-geometry spectral estimation (E2.4)  [DONE]
FFT peak-pick per held-out geometry vs exact Bohr lines, dominant-peak recovery: **h4 0.92, lih 1.0,
beh2 0.85, n2 (see json).** This is the per-geometry ceiling (NO cross-geometry generalization — it
needs the held-out time series, which is what the FSR predicts without any quantum samples there).

### 2A — Fourier-feature MLP (E2.2)  [DONE]
Generic 12.1M-param coordinate net, same inputs (ε(R), t) + random Fourier features, same split.
Held-out: **h4 0.519, lih 0.919, beh2 0.741, n2 0.388** vs FSR 0.95/0.99/0.997/0.97. The FSR's
explicit harmonic head beats a matched-budget generic neural surrogate decisively (lih closest).

### 2D — N_Q sensitivity (E4)  [PENDING] — needs datagen at N_Q∈{50,100,250,500,1000} on a COARSE
grid (fine-grid datagen at N_Q=1000 is ~14h for h4) + retrain. Deferred to Aniket's larger compute.

## TIER 3

### 3A — finite-shot / shot-noise (E1)  [SAMPLER VALIDATED; full study deferred]
The briefing assumes the datagen stores the exact conditionals `p(b|Q,R,t)`; **it does not** (only
`expectations`), and storing them at fine-grid scale is infeasible (h4 ≈ 251×6001×500×256 ≈ 100s GB).
Solution = an **on-the-fly sampler** (draw S Born samples from the per-Q rotated-state probabilities,
use EMPIRICAL marginals) — no conditional storage. **Sampler validated** (`finite_shot_convergence.py`):
empirical vs exact per-qubit marginals decrease ×2.97/×2.87/×3.34/×3.15 per 10× S — the 1/√S law,
confirming convergence to the stored S=∞ targets. **Remaining (deferred to Aniket's compute):** run the
validated sampler inside a COARSE-grid datagen for h4+lih at S∈{10,100,1000}, retrain, plot the
pearson-vs-S degradation curve. The hard/risky part (a correct, validated finite-shot estimator) is done.

### 3C — smoothness (T2)  [DONE]
Finite-difference `S_R(R)=mean_μ‖∂_R⟨Γ_μ⟩‖`, `S_t(R)`. S_R peaks land on physical features:
**n2 at R=2.30 (avoided crossing)**, h4 at compression, h2o at 0.70. Empirical Prop-1 smoothness constant. (`3C_smoothness/`.)

### 3D — pre-flight arrays (W3)  [DONE]
Per-molecule `bw99(R)`, `dw_strong(R)`, `n_strong(R)`, `t_need(R)`. (`3D_preflight/`.)

### 3E — scalable pre-flight proxy (T3)  [DONE]
Cheap HF-orbital-gap proxy vs oracle `omega_op(R)`: corr **0.997 for h4** but **−0.15 to −0.26 for
lih/beh2/n2**. A cheap proxy CANNOT replace the oracle pre-flight for the active-space molecules.

### 3B — frequency-recovery (T4)  [DONE]
FSR predicted dominant frequencies vs exact Bohr lines, nearest-line distance (Eₕ): **lih 0.0027,
h4 0.0194, n2 0.0237, beh2 0.0359** — the FSR recovers the physical frequencies accurately.
(`3B_freqrec/` arrays + error-vs-spacing figure.)
### 3F — scaling-table (E3)  [DONE, partial] — qubits 8, Hilbert dim 256, N_T/N_R/N_Q per mol; peak GPU mem ~2.5 GB / 4 packed jobs.
### 3G — TDHF/CIS dipole baseline  [OPTIONAL, not run].

## Net for the review (12 blocks delivered)
**Done:** 1A multi-seed, 1B ablations, 1C linear-harmonic (Tier 1 complete); 2A Fourier-MLP, 2B
GP/KRR, 2C classical-spectral (Tier 2); 3B freq-recovery, 3C smoothness, 3D pre-flight arrays,
3E proxy, 3F scaling (Tier 3); **3A finite-shot sampler validated** (convergence self-check passes).
**Deferred (need more compute):** 2D N_Q sweep, 3A full training study (sampler ready), 3G optional.

**Honest headline evidence:** the FSR's explicit harmonic structure beats every generic baseline —
GP/KRR (2B, ≈0.1), matched-budget Fourier-MLP (2A, e.g. n2 0.39 vs 0.97), and a fixed-frequency
bank for the hard molecules (1C); each recipe component is justified (1B — adaptive-bw critical for
all active-space mols, orb for h4/n2, low-rank for h4); the FSR recovers physical Bohr frequencies
(3B). Counter-evidence reported plainly: a fixed-bank baseline MATCHES the FSR for smooth beh2/lih
(1C), and there is a seed-13 training-stability failure for n2/h4 (1A). New code on `runpod-results`
only, pending Aniket review for `dev`.
