# Agent Briefing — Observable Regression (Fermionic-Shadow Regressor, FSR)

**Audience:** an AI coding agent (Claude Code) working on the *direct observable
regression* method. Read this top-to-bottom before touching code. It covers
(1) theory & physics, (2) the code map, (3) the Slurm / Trillium workflow in
detail, (4) hard-won lessons (this method's own — mostly first-hand), and (5) how
we keep the research log. It is the mirror of the briefing in the sister repo
(`fermionic-shadow-transformer`, the generative method).

> **One-sentence framing.** Instead of *generating* shadow bitstrings, we
> regress the shadow observables directly: a network `f_θ(R, t) → D ∈ ℝ^K`
> predicts the time-series of Majorana expectation values, and the same
> spectral pipeline (FFT → peaks) reads off molecular energy gaps. This method
> **works** where the generative one doesn't — it already recovers H₂ peaks to
> <0.005 Eₕ — and is the project's lead candidate for the paper.

---

## 1. Theory & physics

### 1.1 The scientific goal (shared with the generative method)
Recover the energy-gap spectrum `E_j − E_0` of small molecules (H₂, H₄) along the
dissociation curve `R ∈ [0.5, 3.0] Å`. The pipeline: prepare an initial state
(Hartree–Fock + symmetry-breaking), time-evolve `|ψ(t)⟩ = e^{−iH(R)t}|ψ₀⟩`,
measure Majorana expectation values via matchgate classical shadows into a signal
matrix `D(R,t) ∈ ℝ^{K×n_T}` (H₂: `K=6`; H₄: `K=120`, degree-2 Majorana
monomials), then run **Chan et al. (arXiv:2312.06992)** spectral analysis:
standardize rows → Ljung–Box pre-screen → covariance `C = DᵀD` → top eigenvectors
→ FFT → peaks of `I(E)` sit at the energy gaps.

### 1.2 What this method does differently
The generative method learns `p(b | Q, R, t)` and samples shadows. **This method
skips sampling entirely** and regresses the shadow estimator's *expectation*
directly: `f_θ(R, t) → D`. No `Q`-conditioning, no bitstrings, no autoregression
— just a deterministic map from geometry+time to the observable time-series. That
sidesteps the generative method's fatal flaw (cross-entropy is blind to temporal
correlations; see that repo's briefing §1.4 / §4.1).

### 1.3 Architecture lineage (v10 → v18) — know this
The model is an MLP/transformer head over **learnable Fourier time features**,
with geometry-conditioned frequencies. The pieces that matter, by version:

- **Fourier time features + geometry-conditioned frequencies** (v4+):
  `ω_k(R) = ω_k^{(0)} + g_φ(ε(R))_k`. The frequency net `g_φ` takes either scalar
  `R` or HF orbital energies `ε(R)`.
- **Adaptive bandwidth** (v11+): `ω_k(R) = ω_op(R)·sigmoid(freq_net(ε(R)))_k`,
  with a (soft) `omega_op_floor` so long-R bands aren't packed too tightly. `ω_op`
  is the per-geometry operational frequency ceiling (`data/compute_omega_op.py`).
- **Explicit-amplitude composition** (v15): replaces the GELU trunk with a
  linear-in-amplitude head
  `y_μ(R,t) = Σ_k a_kμ(R) cos(ω_k t) + b_kμ(R) sin(ω_k t) + dc_μ(R)`,
  low-rank factorized (`amp_rank 16`). This fixed the short-R "composition" failure
  mode and lifted `<0.74`-bin Pearson from 0.66 → 0.935.
- **Chemically-informed inputs**: `--use_orb_features` feeds HF orbital energies
  (standardized, stats stored as model buffers and applied in `forward`) instead
  of scalar `R`. The orb-vs-R comparison is **manuscript pillar 4**.
- **Optional residual trunk** (`--with_residual`) and gradient clipping
  (`--grad_clip`, `--residual_grad_clip`) — the v16/v17 stability machinery.

**Current best:** `v15_explicit` (explicit amplitude, `amp_rank 16`, orb features,
NO residual trunk, unclipped). Seed-pooled n=100: overall Pearson **0.915**,
borderline 0.953, short-R 0.935. **This is the defensible architecture to ship.**

### 1.4 The physics question the paper is about
*How does learning quality change across the dissociation curve?* The model
recovers spectra well at large `R` (simple, single-reference electronic structure)
and struggles at short `R` (near equilibrium, multi-reference character, dense
level crossings, avoided crossings). The transition region `R ≈ 0.7–1.5 Å` (H₄)
is where it breaks down. Two distinct problems live there: a **sampling-rate
obstruction** (the dt=0.05 dataset addresses it) and a **representation issue**
(the explicit-amplitude composition addresses it).

### 1.5 Manuscript framing (the north star)
Title: *"Amortized learning of dynamical fermionic-shadow observables."*
Spectroscopy is the **tested application**, not the headline. Target venue PRX /
PRX Quantum (else QMI; can't submit both). **Five pillars:** (1) short-bond
failures (sampling-rate + representation); (2) reframe as amortized FSR-observable
learning; (3) resource study (learned surrogate reduces quantum measurements;
break-even geometry count); (4) **chemically-informed features (orb-vs-R) — a
load-bearing ablation, not a side note**; (5) generalization beyond H-chains (LiH
next). The paper does NOT need one model best in every R-bin — it needs a stable
accurate surrogate (v15_explicit) + a clean orb-vs-R ablation.

---

## 2. Code map

Package root is `models/` (run Python with `models/` as cwd; imports are
`from fermionic_pipeline...` and `from src...`).

| Path | Role |
|---|---|
| `fermionic_pipeline/models/observable_regressor.py` | **The model.** Fourier-time MLP; adaptive bandwidth, explicit-amplitude head, orb-energy standardization buffers, optional residual trunk. `ObservableRegressorConfig` is saved in the checkpoint. |
| `fermionic_pipeline/models/film_transformer.py` | FiLM-conditioned transformer variant. **Imports the shared transformer blocks from `src.models.transformer_core`** (the only old-tree dependency — see below). |
| `fermionic_pipeline/models/conditioning.py` | conditioning modules (FiLM / cross-attention experiments). |
| `fermionic_pipeline/data/regression_dataset.py` | Dataset gen. Computes exact signal-matrix targets analytically from statevectors + matchgate shadow formalism (time-batched estimator, ~5–50× faster than per-t loop). Stores `(n_R, n_T, K)` + `eigvals`, `R_values`, `hf_orbital_energies`, `omega_op` in HDF5. |
| `fermionic_pipeline/data/generate_shadows.py` | Matchgate shadow / Majorana machinery + **pyscf** Hamiltonian construction (quantum chemistry). |
| `fermionic_pipeline/data/compute_omega_op.py` | per-geometry operational frequency ceiling `ω_op(R)`. |
| `fermionic_pipeline/data/dataset.py` | torch dataset wrappers (uses `make_std_mask` from `transformer_core`). |
| `fermionic_pipeline/training/regressor_trainer.py` | **Training loop.** MSE + cross-observable corr loss (`--alpha_corr`) + optional temporal-corr (`--alpha_temporal_corr`) + spectral aux (`--alpha_spec`); AdamW + cosine LR; joint/decoupled grad clipping; saves config + buffers + R/t grid into the checkpoint. **`load_checkpoint_model` lives here.** |
| `fermionic_pipeline/eval/regressor_eval.py` | **Spectral eval.** Builds D from predictions, runs FFT pipeline, reports per-geometry per-observable temporal Pearson + peak match vs exact gaps → `regressor_eval.json`. |
| `fermionic_pipeline/eval/plot_regression.py` | spectral-comparison grids, per-geometry time series, Chan-style D/covariance/spectrum plots. |
| `fermionic_pipeline/eval/composition_diagnostic.py` | **decomposes** predictions into amplitude-envelope correlation vs phase/frequency error, stratified by R-bin. The tool that localizes failures. |
| `fermionic_pipeline/eval/{fft_amplitude,freq_*}_diagnostic.py` | FFT-amplitude and freq-net alignment/derivative probes. |
| `fermionic_pipeline/inference/spectral_analysis.py` | shared FFT / Ljung-Box / peak-extraction utils. |
| `fermionic_pipeline/vendor/optimal_matchgate_circuit.py` | vendored matchgate-circuit code. |
| `scripts/hf/` | `upload_to_hf.py` (+ `inference.py`, model cards) — bundle a checkpoint to Hugging Face. |
| `scripts/diagnostics/` | standalone diagnostic drivers. |
| `slurm/regression_v*.sh` | the experiment drivers (see §3). |
| `src/models/transformer_core/` | **shared transformer building blocks** (attention/decoder/etc.) that `film_transformer.py` and `dataset.py` import. This is the ONLY code from the old generative tree present in this repo — it is a self-contained kernel, no generative model here. |
| `src/training/utils.py` | `AverageMeter`, LR schedulers (also imported by the trainer). |
| `src/` (repo root) | collaborators' standalone quantum-chemistry library (pyscf/pennylane: `symmetry_adjusted_classical_shadows_main`, `quantum_data`). Reference/utility; not imported by `fermionic_pipeline`. |

### 2.1 Quick usage (current architecture, not the stale v5 guide)
```bash
# generate dataset (H4, dt=0.05)
python3 -m fermionic_pipeline.data.regression_dataset \
  --output results/fermionic_pipeline/regression/<tag>/regression_targets.h5 \
  --n_atoms 4 --r_start 0.5 --r_end 3.0 --r_step 0.05 \
  --t_max 100.0 --n_times 6001 --n_q 1000 --n_workers 8

# train v15_explicit (the current best architecture)
python3 -m fermionic_pipeline.training.regressor_trainer \
  --data_path <dataset.h5> --save_dir <dir> --device cuda --seed 42 \
  --steps 150000 --batch_size 256 --lr 1e-3 --final_lr 1e-7 --warmup_frac 0.05 \
  --weight_decay 5e-4 --d_hidden 768 --n_layers 6 --n_fourier 256 --fourier_scale 20.0 \
  --conditioned_frequencies --freq_net_hidden 128 --freq_net_layers 3 \
  --adaptive_bandwidth --omega_op_floor 8.0 --soft_omega_floor \
  --explicit_amplitude --amp_rank 16 --use_orb_features --standardize_orb_energies \
  --alpha_corr 1.0 --eval_every 2000

# eval + plots + composition diagnostic
python3 -m fermionic_pipeline.eval.regressor_eval   --data_path <h5> --checkpoint <dir>/regressor.pt --save_dir <dir>/eval  --device cuda --ljung_box_p 0.06
python3 -m fermionic_pipeline.eval.plot_regression  --data_path <h5> --checkpoint <dir>/regressor.pt --save_dir <dir>/plots --device cuda --ljung_box_p 0.06
python3 -m fermionic_pipeline.eval.composition_diagnostic --data_path <h5> --checkpoint <dir>/regressor.pt --save_dir <dir>/eval --device cuda
```
The HDF5 stores exact `eigvals`, `R_values`, `expectations`, `hf_orbital_energies`,
`omega_op` — you can analyze the spectrum vs R without running the model.

---

## 3. Slurm / Trillium workflow — in detail

All heavy compute runs on **Trillium** (SciNet). Laptop = orchestration +
analysis; cluster = datagen / train / eval. This project is the *mature* reference
for the team's slurm conventions.

### 3.1 Cluster facts (memorize)
- **Host:** `aniketrd@trillium-gpu.scinet.utoronto.ca`.
- **Workdir on cluster:** `$SCRATCH/generative-quantum-states` (compute nodes only
  write to `$SCRATCH`). Every job `cd`s here.
- **Account:** `--account=rrg-aspuru`. **Partition:** `--partition=compute_full_node`.
  **GPUs:** `--gpus-per-node=4`.
- **Env:** `module load StdEnv/2023 python/3.11 cuda/12.2`; `source "$HOME/envs/gqs/bin/activate"`;
  then `export PYTHONNOUSERSITE=1`, `unset PYTHONPATH`, `export PYTHONUNBUFFERED=1`
  (prevents wrong-site-packages bugs; streams logs live). Datagen/eval that don't
  need CUDA can drop `cuda/12.2`.

### 3.2 The driver pattern (`slurm/regression_vNN.sh`)
Each experiment is a self-documenting driver script. Conventions to follow when
you add a new one:
- **Header block** = *why this run exists, the exact hypothesis, and the numeric
  decision rule* ("if metric X recovers above Y → ship; if it flatlines → broken,
  debug Z"). Read `regression_v18.sh` / `regression_v18b_orb_clip.sh` as templates.
- **Generated per-run scripts.** The driver writes `slurm/_train_<tag>.sh` /
  `slurm/_eval_<tag>.sh` per variant and `sbatch`es them. The `_`-prefixed
  generated scripts are **gitignored** (`slurm/_*.sh`); the driver is tracked.
  Don't commit generated scripts.
- **Composable stages:** drivers take `--train` / `--eval` / `--all` so you can
  re-run just eval on existing checkpoints.
- **Job chaining:** submit with `--parsable` to capture the train job id, then make
  eval depend on it: `--dependency=afterok:$TRAIN_JOB`. One command submits the
  whole train→eval chain.
- **Matrix runs:** e.g. `{orb, R} × {seed 42, 1729}` = 4 train + 4 eval. Always run
  **≥2 seeds** so you can seed-pool to n=100 and report seed spread.
- **Steering off bad nodes:** GPU nodes **`trig0019`, `trig0034`** have crashed
  evals with "No CUDA GPUs are available." Add `--exclude trig0019,trig0034`. The
  checkpoint loader also falls back to CPU if CUDA vanishes mid-job.

### 3.3 Getting results back + monitoring
- **Fetch to laptop:** `slurm/fetch_results.sh` rsyncs results+logs from Trillium,
  **excluding `*.h5`** (datasets are big and regen-able; `--include-h5` to override).
  Restrict with `--which <token>` (literal substring, e.g. `--which v18`), add
  `--logs` for logs. Checkpoints (`*.pt`) and datasets stay on the cluster; pull
  the small artifacts (eval JSON, plots, `history.json`) for analysis.
- **Watch jobs:** `slurm/tail_jobs.sh`; `squeue -u $USER`; read `logs/<job>.out`.
- **Datasets are not version-controlled** (gitignored `*.h5`). If a result looks
  off, confirm which dataset tag it used.

### 3.4 Hugging Face
`scripts/hf/upload_to_hf.py` bundles a checkpoint + the single-file model arch +
inference loader + R/orbital-energy grid + eval/history into an HF repo
(re-runnable; `--version_tag` pins an immutable revision). Reads `HF_TOKEN` from
`.env` (gitignored — never commit it).

---

## 4. Lessons (this method's own — apply them)

These are first-hand, paid for in compute. They shaped the current architecture
and are live constraints.

### 4.1 Monitor the metric that is the deliverable, not a proxy ⚠️ load-bearing
The trainer logs `val_corr`, which is the **cross-observable** Pearson (K obs at
fixed (R,t)) — explicitly **NOT** the per-observable **temporal** Pearson that
`regressor_eval.py` and spectroscopy actually need (documented at
`regressor_trainer.py` ~line 67–74). In v18 the orb arm showed a healthy
`val_corr ≈ 0.95` while the standalone eval's temporal Pearson had **collapsed to
0.32** — the proxy hid the failure for a full run. **Always run the standalone
eval (the real metric) and trust it over the training curve.** If you want the
training signal to track the deliverable, use `--alpha_temporal_corr`.

### 4.2 A train-vs-eval gap is a red flag
When `val_corr` is great but the standalone eval is bad, suspect (a) a metric
mismatch (different axes) or (b) an eval-path bug (e.g. standardization applied
differently). Reconcile before believing either number. Standardization stats for
orb features live in **model buffers saved in the checkpoint** and are applied in
`forward` — they must travel with the checkpoint and apply identically train/eval.

### 4.3 Stratify per-R-bin; pool seeds
Never report a single aggregate. The v18 aggregate (0.32) hid that short-R was
0.02 and long-R was 0.58. Report per-bin `(<0.74, [0.74,1.0), [1.0,1.5),
[1.5,2.0), ≥2.0)`, seed-pooled over ≥2 seeds, with the seed spread. A seed-fragile
result is not a result (the gate we use: borderline seed |Δ| ≤ 0.15).

### 4.4 Diagnostics that decompose the failure
`composition_diagnostic.py` splits each prediction into amplitude-envelope
correlation vs phase/frequency error. In v18 it instantly localized the orb
collapse: envelope ≈ 0 and phase ≈ π/2 (random) — the amp_net/freq_net stopped
tracking geometry. Build/extend decomposing diagnostics rather than staring at one
scalar.

### 4.5 Loss explosions & gradient clipping
v18 ran unclipped on the premise "no residual trunk ⇒ stable." It backfired:
`train_mse` exploded to ~1e5 (worst on the 4-dim orb input), corrupting the
delicate amp/freq composition. If training shows blow-ups, add `--grad_clip 1.0`.
(`regression_v18b_orb_clip.sh` is exactly this test.)

### 4.6 Honest baselines
The R-input baseline must genuinely train (not flatline to ~0) to be credible —
"works but worse than orb" is a result; "zero correlation" reads as a broken
pipeline and invites reviewer suspicion. v17's R arm flatlined (broken); v18's R
arm trained cleanly (0.737) and is the honest baseline.

### 4.7 Commit load-bearing source as you go
The v11–v18 architecture sat uncommitted for weeks; when an old run's exact source
was needed it was unrecoverable. Commit the model/trainer/data code (scaffolding
and generated scripts can stay out). This repo's history starts clean — keep it
that way.

### 4.8 Current open threads (pick up here)
- **v18 orb regression (OPEN).** `v18_orb` is config-identical to `v15_explicit`
  (which scored 0.915) yet collapsed to 0.32; the R arm (0.737) now *beats* orb,
  inverting pillar 4. Leading hypothesis: the unclipped loss explosion.
  **In flight:** `regression_v18b_orb_clip.sh` (re-run orb seed 42 with
  `--grad_clip 1.0`). Decision: if envelope/phase recover and train_mse stops
  exploding → restore clipping in v18 proper and re-run the full `{orb,R}×{42,1729}`
  matrix; if still collapsed → the standardization-into-forward path is the
  suspect.
- **HF upload.** The `v15_explicit` checkpoint (the 0.915 model) is the artifact to
  publish; fetch it from the cluster, re-eval with current code to confirm it still
  scores 0.915 (also confirms `forward` didn't drift), then `upload_to_hf.py`.
- **orb-vs-R (pillar 4).** Once orb is fixed, the clean comparison is
  `v15_explicit` (orb) vs the stable R arm, same arch/dataset/steps.

---

## 5. How to keep the research log

Keep a single chronological journal (`research-logs/architecture-log.md` — seeded
in this repo from the shared history). It is the project's memory; write it for an
agent (or human) who picks up cold. Match this format exactly.

### 5.1 Format & conventions
- **Header:** `M/DD — terse title` (month/day, no year, em-dash, 4–8 words; prose
  lowercase, terse). e.g. `5/31 — v18 orb collapse + repo split`.
- **Open** with a one-sentence executive summary of the finding / session purpose.
- **Body** = 3–7 labelled subsections: `**fix N: title**` /
  `**hypothesis:** / **diagnosis:** / **decision:**`, each with the
  physics/architecture rationale, what changed, and results.
- **End** with concrete next steps: the exact `sbatch`/`bash` command or a numeric
  decision rule.
- **Bold** for critical findings / decision triggers; `backticks` for paths /
  functions / flags; `$...$` LaTeX for quantities (`$\omega_k(R)$`); metrics to 2–3
  sig figs; per-geometry results as markdown tables (`R | exact gap | model peak | δ`).
- **Versions** as `vN`; run tags as `molecule_regress_vNN_<variant>_s<seed>`.
- End substantive code entries with a `files:` line; cite `sbatch` commands in
  place with `--time`/flags and a note on the compute budget.

### 5.2 When / why / what
- **When:** every meaningful checkpoint — a bug fixed, a hypothesis tested, an arch
  changed, a sweep launched, a result read. Cadence ≈ per experiment-cycle
  (datagen→train→eval is many hours), so daily-to-every-few-days when active.
- **Why:** the log is *why we did things*. Record the hypothesis and decision rule
  **before** seeing results, then the outcome and the call. This is what stops a
  fresh agent re-running dead ends.
- **What:** rationale > narration. Record negative results explicitly (e.g. "v17 R
  arm flatlined — broken baseline, not an honest result"). Always note the next
  concrete action.

### 5.3 Template
```markdown
M/DD — <terse title>

<one-sentence executive summary>

**hypothesis:** <what you expected and why>

**fix 1: <title>**
<rationale; what changed; result with numbers>
files: `fermionic_pipeline/...`, `slurm/...`

**diagnosis / result:**
<what the numbers showed; per-R-bin table if relevant>

**decision / next:**
<the call + exact command>
`bash slurm/regression_v18b_orb_clip.sh`
```

---

### TL;DR for the incoming agent
1. We regress shadow observables `f_θ(R,t) → D` directly (no sampling); FFT → peaks
   → energy gaps. It works where the generative method fails.
2. Best architecture = **`v15_explicit`** (explicit amplitude, orb features, no
   trunk, unclipped), overall Pearson 0.915. Ship it; don't chase long-R perfection.
3. **Trust the standalone eval's per-observable temporal Pearson, not the training
   `val_corr`** (different axis — it hid the v18 collapse). Stratify per-R-bin,
   seed-pool n=100.
4. Compute on Trillium (`$SCRATCH/generative-quantum-states`, `gqs` venv,
   `rrg-aspuru`, `compute_full_node`, 4 GPUs); drivers generate `_train/_eval`
   scripts and chain with `--dependency=afterok`; avoid nodes `trig0019/trig0034`;
   fetch small artifacts with `fetch_results.sh`.
5. Open thread: the **v18 orb collapse** (`regression_v18b_orb_clip.sh` probe) and
   publishing the `v15_explicit` checkpoint to HF. Manuscript north star = the 5
   pillars (§1.5); orb-vs-R is load-bearing.
6. Keep the research log in `M/DD — title` format — hypotheses and decision rules,
   not just narration.

---

## Appendix A — Persistent project memory (dumped)

> This is the assistant's cross-session memory from the original `molecular-shadows`
> workspace, dumped here so an agent working *in this repo* (which has its own,
> separate memory store) inherits the full project context. Dates are absolute.
> Treat file/line citations as point-in-time — verify against current code.

### A.0 User & working style
- **User:** Aniket (aniketdeshh@gmail.com). Git author `aniket-desh`, works on
  branch `aniket` (default branch `main`). Collaborator **Luis** (quantum-chemistry
  / resource-study side).
- **Preferences observed:** moves fast, values momentum; wants load-bearing
  **source committed** so progress isn't lost (the v11–v18 architecture going
  uncommitted for weeks caused real pain); terse lowercase research-log style;
  decisive answers with the numbers shown, not hedging.
- **Repo split (2026-06-02):** the single `molecular-shadows` tree was split into
  two repos so agents don't overlap — **FST** (`fermionic-shadow-transformer`, the
  generative method) and **FSR** (`fermionic-shadow-regressor`, this repo, the
  regression method). Both carry the collaborators' quantum-chemistry code under
  `src/`. The assistant works in **FSR** going forward. Neither new repo is
  committed yet (per instruction at split time).

### A.1 Commit discipline
In this research project, commits are made **sparingly** — only when a solid
result/model lands, NOT for tooling/script/infra changes (slurm helpers, fetch
scripts, diagnostics). *Why:* exploratory research with many in-flight scripts;
the git log is reserved for meaningful scientific progress. *How to apply:* don't
offer to commit after infra edits; commit when the user asks or a validated result
lands. **Exception (2026-05-31):** when load-bearing **source** is uncommitted and
at risk, flag it and commit it — the v11–v18 architecture sat uncommitted until
v15's source became unrecoverable. The repo has a **ruff pre-commit hook**
(`ruff` + `ruff format`); keep commits lint-clean.

### A.2 Manuscript plan (north star)
- **Venue:** PRX / PRX Quantum first (higher impact); if rejected → QMI. Cannot
  submit to both simultaneously (QMI xor PRX). Decided 2026-05-04 (Luis + Aniket).
- **Reframed title/framing:** *"Amortized learning of dynamical fermionic-shadow
  observables"* (was "learning absorption spectra from accurate data generated by
  quantum algorithms"). New framing highlights the novelties — time series,
  Majorana observables, resource surrogate — and casts **spectroscopy as the
  tested application**, not the headline.
- **Five pillars (Aniket, 2026-05-04):**
  1. **Short-bond failures** (immediate): two problems — sampling-rate obstruction
     (the dt=0.05 dataset addresses it) + a representation issue. v15_explicit's
     explicit composition lifted `<0.74` Pearson 0.66 → 0.935.
  2. **Reframe** as amortized learning of dynamical FSR observables; spectroscopy
     = tested application. Mostly rewriting.
  3. **Resource study:** show the learned FSR surrogate reduces extra quantum
     measurements; formalize cost for new geometries/time-points/shots; report a
     "break-even" geometry count. Mostly Luis-driven.
  4. **Chemically-informed features:** R-input (pure geometry) is the BASELINE
     CONTROL; orbital-energy inputs should improve data efficiency/robustness.
     This is the orb-vs-R ablation — a **load-bearing** experiment, not a side
     ablation.
  5. **Generalization beyond H-chains:** LiH easiest next; general arch =
     molecule/Hamiltonian encoder + observable-query encoder + time/Fourier
     dynamics head.
- **Implication:** the paper does NOT need one model best in every R-bin. It needs
  a stable accurate surrogate for spectroscopy (v15_explicit suffices; long-R
  Pearson 0.87 is absorbed by Chan peak post-processing) + a CLEAN orb-vs-R
  ablation.

### A.3 v15_explicit is the best architecture
- **Best & defensible to ship** (2026-05-20 readout). Seed-pooled n=100: overall
  Pearson **0.915**, borderline [0.74,1.0) **0.953**, `<0.74` **0.935**; stable, no
  loss explosion, no gradient clipping needed.
- **Only deficit:** long-R time-domain Pearson (`≥2.0`: 0.994→0.873; `[1.5,2.0)`:
  0.971→0.930), because the pure linear-in-time Fourier basis is too restrictive
  where trajectories carry small non-sinusoidal/DC structure. This is absorbed by
  the Chan spectral post-processing, so it does NOT hurt the spectroscopy claim
  (v13_v12f8 already hits 100% top-1 peak).
- **Why nothing beats it:** the long-R recovery needs a nonlinear GELU residual
  trunk, and that trunk reintroduces the training loss explosion + seed fragility.
  v16 (joint clip) bounded the explosion but throttled the borderline win. v17
  (decoupled clip) made it WORSE (train_mse→1e6, borderline seed |Δ|=0.78, 5× over
  the 0.15 gate). **Strategic point:** stop optimizing long-R; run the orb-vs-R
  ablation on the STABLE v15_explicit arch.
- **⚠️ CAVEAT (2026-05-31):** the working tree at the time of the split no longer
  reproduced v15_explicit — `v18_orb` is a config-identical re-run yet collapsed
  (see A.4). Treat 0.915 as the historical v15 *checkpoint's* number until the v18
  regression is explained.

### A.4 v18 orb-vs-R regression (OPEN)
- v18 ran orb-vs-R on the v15_explicit arch. **The orb arm collapsed and the
  comparison inverted** (2026-05-31, seed-pooled n=100, per-observable temporal
  Pearson): `v18_orb` overall **0.323** (`<0.74` 0.022, `≥2.0` 0.577; top-1 peak
  6/50); `v18_R` overall **0.737** (honest, monotonic-in-R; top-1 peak 25/50). So
  **R now beats orb**, inverting pillar 4; orb is worst at short R, exactly where
  v15's orb win was the headline.
- **It's a code/data regression, not a confound.** `v18_orb` is byte-identical in
  flags to `v15_explicit` (same dataset `h4_regress_v13` dt=0.05, 150k steps,
  `--alpha_corr 1.0`, `--standardize_orb_energies --explicit_amplitude --amp_rank
  16`; v15 hardcoded `--use_orb_features`). Same config + same data: 0.915 → 0.323.
  Something in the working-tree code/regenerated h5 drifted between the v15 run
  (~5/10) and v18 (~5/30). (The split + commits now make this recoverable.)
- **Failure localized** by `composition_diagnostic`: orb amp_net + freq_net stopped
  tracking geometry — envelope_pearson ≈ 0 (0.007/0.003/0.049/0.167/0.631
  short→long), phase_err ≈ 1.0–1.6 rad (≈ random). R arm fine (envelope 0.69–0.99).
- **Why it hid:** `val_corr` (~0.954) is cross-observable Pearson, NOT the
  per-observable temporal Pearson eval uses (`regressor_trainer.py` ~67–74). v18
  trained on `--alpha_corr 1.0` so the training signal was blind to the collapse.
- **Leading hypothesis:** `train_mse` exploded to ~1.3e5 (orb) vs ~480–6k (R) under
  v18's "no trunk ⇒ stable unclipped" decision (`--grad_clip 0`); unclipped spikes
  corrupt the delicate amp/freq composition.
- **In flight:** `slurm/regression_v18b_orb_clip.sh` re-runs `v18_orb` seed 42 with
  only `--grad_clip 1.0` added. If envelope/phase recover & train_mse stops
  exploding → restore clipping in v18, re-run `{orb,R}×{42,1729}`; else → bisect
  the standardization-into-forward refactor.
- **Shortcut for orb-vs-R + HF:** `v18_R` did NOT collapse (valid v15-arch R
  baseline). The clean pair is `{v15_explicit checkpoint (orb, 0.915, fetch from
  cluster) vs v18_R (0.737)}` — same arch/dataset/steps — IF the v15 checkpoint
  re-evals to 0.915 with current code (also confirms `forward` didn't drift). That
  checkpoint is the artifact to publish to HF.
