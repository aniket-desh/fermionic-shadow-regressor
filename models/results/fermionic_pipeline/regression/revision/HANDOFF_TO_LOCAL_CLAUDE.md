# Handoff from RunPod Claude → Local Claude / Aniket

Everything produced on the pod is on the **`runpod-results`** branch (results + ALL code) and on
HF (`aniketdesh/molecular-shadows-datasets`). Nothing was pushed to `dev`. Pod is idle. This file
lists what's deferred and the decisions that need a human/Local-Claude call.

## Where everything is
- **`runpod-results` branch:** all eval JSONs, plots, npz arrays, `REVISION_RESULTS.md` (the B9
  synthesis), and all new code (baselines/diagnostics/orchestration + the modified
  `nature_style.py`/`compute_dipole_coeffs.py`/`dipole_bo.py`).
- **HF (private) `aniketdesh/molecular-shadows-datasets`:** per-molecule train+extended `.h5` +
  checkpoints for n2, beh2, h2o(+v2), lih(+v2), h4 — so nothing needs regenerating.

## Deferred work (not done — needs more compute; approaches are ready)
1. **2D — N_Q sensitivity sweep (E4).** Fine-grid datagen at N_Q=1000 is ~14 h for h4, so it needs a
   COARSE grid (e.g. step 0.05, n_times ~1001) at N_Q∈{50,100,250,500,1000} for h4+lih, then retrain.
   Reuse `scripts/revision/train_eval.sh` + `fleet_runner.sh`. ~2 h datagen + 10 retrains.
2. **3A — finite-shot full study (E1, the BLOCKING review item).** The sampler is **validated**
   (`scripts/data/finite_shot_convergence.py`, 1/√S confirmed). What remains: wrap that empirical-
   marginal sampler into a coarse-grid datagen (`--shots S`, NO conditional storage), generate h4+lih
   targets at S∈{10,100,1000}, retrain, plot pearson-vs-S. The hard/risky part (a correct estimator)
   is done; this is mechanical compute.
3. **3G — TDHF/CIS dipole baseline (E2.5, optional).** Not run.
4. **H2O** stays dropped from the paper (briefing #2 decision) — no further work.

## DECISIONS NEEDED (your call)
1. **Code → `dev`?** All new code is on `runpod-results` tagged "pending Aniket review." Highest-value
   to port: **`nature_style.py` LaTeX→mathtext fallback** (a genuine fix — any LaTeX-less env needs it),
   and the **dipole infra** (`compute_dipole_coeffs.py --molecule`, `dipole_bo.py --dynamic`). The
   baseline/diagnostic scripts are review-quality but ad-hoc. Decide which to merge.
2. **`compute_req.py` bug (briefing #5).** Its `SCANS` windows are too narrow → argmin hits the
   boundary for all 4 molecules. Use the corrected CCSD(T)/cc-pVTZ values
   **R_eq = {n2:1.114, lih:1.609, beh2:1.332, h4:0.895}** (in `R_eq_ccsdt_results.txt`) and widen the
   committed script's windows.
3. **LiH v2 (briefing #4).** My recommendation: **keep v1.** v2's buffered box [0.8,3.4] REGRESSED the
   reported [1.0,3.2] (r̄ 0.978 vs v1 0.992); and my v1 re-run has NO R=3.2 spike (R3.20=0.992; the
   0.45 in the briefing was the original off-pod checkpoint). Confirm v1 is the paper model.
4. **Dataset D convention (dipole #7).** The current datagen stores `D` as a shadow estimate
   **≈0.1–0.2× the exact ⟨Γ⟩** (shadow_coeff/N_Q normalization; true of current-code H4 too). The
   dipole-BO relative/samples metric is unaffected (cancels), but the **absolute dipole y-axis (a.u.)
   in the fit panels is scaled** — unlike the original Trillium H4 figure. Decide if the paper needs
   physically-scaled absolute dipoles (then the datagen normalization must be reconciled).
5. **1A seed-13 collapse.** Seed 13 caused a training-stability failure (n2 pearson ≈0, h4 0.573)
   while the other 4 seeds are excellent. Report the honest mean±std as-is, or add a stability note /
   mitigation? (The reviewer asked for robustness — this IS the answer, but it's a wart.)
6. **n2 multi-seed reporting.** n2 `pearson_mean` is split-sensitive (the avoided-crossing edge
   geometries tank the MEAN when held out) — per-seed MEDIAN is robustly ~0.99. Report which?

## Quick pointers to the evidence (all on `runpod-results`)
- B9 synthesis: `…/revision/REVISION_RESULTS.md`
- Molecule batch (n2/beh2/lih/h4 GREEN, h2o RED): per-mol `…/<mol>_regress_v1_orb_s42_model/`
- Dipole-BO: LiH negative (`lih_regress_v1/bo/`), BeH2 positive 13/26 (`beh2_regress_v1/bo/`)
- Cross-molecule coherence grids (all 4): `…/<mol>…/plots_extrap/coherence_grid.npz`
- R_eq: `…/revision/3EF/` (STO-3G, flagged) + `R_eq_ccsdt_results.txt` (CCSD(T), use these)
