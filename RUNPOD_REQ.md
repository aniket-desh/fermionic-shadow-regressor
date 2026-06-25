# RunPod Briefing #5 — calculated equilibrium bond lengths (documented level)

For the cross-molecule figures we normalize R by each molecule's equilibrium R_eq.
Luis: calculated values are fine if we state the method. Our minimal STO-3G active-space
energy minimum is NOT usable (gives ~2x-too-long bonds), so compute proper equilibria.

## Run (cheap, ~minutes)
```bash
cd <repo> && git pull origin dev
cd models && python -m scripts.compute_req_ccsdt
```
It scans each molecule's bond length, evaluates **CCSD(T)/cc-pVTZ** total energy, and
fits a parabola for the minimum (full molecule, not the active space). Prints
`R_EQ_CALC = {...}` for h4/lih/beh2/n2.

## Report back
- The printed `R_EQ_CALC` dict (and flag if any "min at scan edge" warning fires -- if so,
  widen that molecule's `SCANS` window and rerun).
- Sanity: expect roughly N2 ~1.10, LiH ~1.60, BeH2 ~1.33 Ang (close to experimental);
  H4 (symmetric chain) ~1.0-1.1. If anything is wildly off, say so.

That's all -- no GPU, no datagen. Paste the dict back and Local Claude drops it into the
figure scripts; the paper will cite "equilibrium bond lengths from CCSD(T)/cc-pVTZ".
(If Luis wants a different level/basis, change `BASIS` in compute_req_ccsdt.py.)
