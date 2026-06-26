# RunPod Briefing #7 — dipole reconstruction on LiH (headline downstream experiment)

## Goal
Redo the dipole-reconstruction / Bayesian-optimization experiment (current paper §IV.C, H4
only) on **LiH**, so it becomes the headline molecule's downstream-utility result. Deliver a
`dipole_bo_sweep_lih.pdf` + per-R numbers analogous to H4's. If LiH transfers cleanly this
goes in the MAIN text; if messy, we keep H4's in main and note LiH in passing — so **report
the numbers honestly either way**.

## Why this is NOT a trivial flag (two real wrinkles — read before running)
1. **Frozen core / active space.** H4 in STO-3G is the *full* space (8 spin-orbitals), so its
   dipole decomposes exactly onto the 120 Majorana channels. **LiH CAS(2,4) freezes the Li 1s
   core** (6 spatial orbitals, 4 e⁻ → 1 core orbital + 4 active + 1 frozen virtual). The dipole
   operator must be built in the **same active space** as the Hamiltonian, i.e. matching
   `qml.qchem.molecular_hamiltonian(..., active_electrons=2, active_orbitals=4)`, so it acts on
   the same 8 qubits as the Γ_μ. The **self-test (reconstruct ⟨μ(0)⟩) is the gate** that this
   is consistent — do not trust the coeffs unless it passes.
2. **Permanent dipole breaks the metric.** H4 is centrosymmetric → zero permanent dipole, so
   ⟨μ⟩(t) oscillates around 0 and relative-RMS is honest. **LiH is heteronuclear with a large
   permanent dipole**, so `dipole_bo.py`'s tolerance denominator `rms = sqrt(mean(mu_exact²))`
   (line ~108, DC included) is dominated by the static offset → the 5% relative tolerance
   becomes trivially easy. Score the **mean-subtracted (dynamic) dipole** for a fair
   samples-to-tolerance comparison; report both so we can see the gap.

---

## Step 1 — generalize `compute_dipole_coeffs.py` to the molecule registry
Currently `_dipole_matrices()` is hard-coded to a hydrogen chain (`symbols=["H"]*n_atoms`,
full space) and `_self_test()` calls `build_hydrogen_chain_hamiltonian`. Generalize, keeping the
H-chain path byte-identical (mirror what was done for the datagen registry in
`generate_shadows.py`):
- `_dipole_matrices(molecule, R, n_qubits)`: for a named molecule, build the **active-space**
  dipole. Use `_MOLECULE_SPECS` + `_molecule_coordinates` (from `generate_shadows.py`) for the
  geometry, and call `qml.qchem.dipole_moment` with the **same core/active partition** the
  Hamiltonian uses. Get the partition from `qml.qchem.active_space(electrons, orbitals,
  active_electrons=spec["ae"], active_orbitals=spec["ao"])` (LiH: total electrons 4, total STO-3G
  orbitals 6, ae=2, ao=4 → 1 core + 4 active). The returned [μ_x,μ_y,μ_z] must be 2⁸×2⁸.
- `_self_test`: use `build_molecule_hamiltonian(molecule, R)` and `prepare_initial_state(...,
  n_electrons=molecule_n_electrons(molecule))`. **Gate: every ⟨μ_a(0)⟩ reconstructs to |Δ|<1e-7.**
- `main`: add `--molecule` (default keeps the `--n_atoms` H-chain path). n_qubits=8, keys=
  `majorana_2pt_keys(16)` (120) for LiH.

**Self-test first (cheap):**
```bash
cd models
python3 -m scripts.bo.compute_dipole_coeffs --self_test --molecule lih \
  --data_h5 results/fermionic_pipeline/regression/lih_regress_v1/regression_targets.h5
```
Do not proceed unless it prints SELF-TEST PASSED (all xyz |Δ|<1e-7). If it fails, the JW /
active-space / unit convention is off — debug that, don't paper over it.

## Step 2 — compute the LiH dipole coeffs
```bash
python3 -m scripts.bo.compute_dipole_coeffs --molecule lih \
  --data_h5 results/fermionic_pipeline/regression/lih_regress_v1/regression_targets.h5 \
  --out    results/fermionic_pipeline/regression/lih_regress_v1/dipole_coeffs.npz
```

## Step 3 — add a dynamic (mean-subtracted) scoring option to `dipole_bo.py`
In `_bo_one_geometry` (around lines 93–113), add a `--dynamic` flag (or `--subtract_mean`) that
subtracts the time-mean from `mu_exact` before forming `y_true`, the `rms` denominator, and the
residual. The c0 constant drops out, leaving the spectroscopically informative oscillation. Keep
the default (DC-included) path so H4 is unchanged; **report LiH under BOTH**.

## Step 4 — run the LiH dipole-BO sweep (non-oracle prior)
```bash
LIH=results/fermionic_pipeline/regression
python3 -m scripts.bo.dipole_bo --molecule LiH --dynamic \
  --npz        $LIH/lih_regress_v1/dipole_coeffs.npz \
  --data_h5    $LIH/lih_regress_v1/regression_targets.h5 \
  --checkpoint $LIH/lih_regress_v1_orb_s42_model/regressor.pt \
  --save_dir   $LIH/lih_regress_v1/bo \
  --omega_op_source train-interp \
  --sweep_lo 1.0 --sweep_hi 3.2 --n_sweep 26 \
  --rel_error 0.05 --max_samples 40 --device cuda
```
(`--omega_op_source train-interp` keeps it non-oracle, matching the rest of the LiH results.
The `--R` example geometries default to [1.40, 2.50] — pick two representative LiH geometries,
e.g. one mid-bond and one stretched, if you want cleaner example panels.)

## Step 5 — report + commit (results branch)
Commit to `runpod-results`: `lih_regress_v1/dipole_coeffs.npz`, `lih_regress_v1/bo/` (the
`dipole_bo_sweep_lih.pdf` + plotdata pkl), and the generalized `compute_dipole_coeffs.py` /
`dipole_bo.py` (push the code to **dev**). Paste back:
- **Self-test result** (must be PASSED) and any active-space convention notes.
- **Zero-sample FSR relative error vs R** under the dynamic metric: how many of the 26 geoms are
  already within the 5% DFT-level tolerance with ZERO samples, the min error and its R, and where
  it saturates the 40-sample budget.
- The same under the **DC-included** metric (so we see how much the permanent dipole flatters it).
- A one-line verdict: does LiH give a clean zero-sample mid-bond region like H4 (7/26), or not?

This determines whether the dipole section is LiH-headline (main) or stays H4 (main/appendix).
Local Claude is writing the prose in parallel with a `\color{red}` placeholder for this result —
so the cleaner and more honest the numbers, the better.
