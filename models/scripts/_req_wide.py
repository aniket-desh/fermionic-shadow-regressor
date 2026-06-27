"""Corrected R_eq scan: WIDE ranges so argmin is interior (not a boundary hit).
The committed scripts/compute_req.py used narrow windows; all 4 molecules hit the
upper bound, so its R_eq values are unreliable. Same definition (argmin_R E0)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fermionic_pipeline.data.generate_shadows import build_molecule_hamiltonian

SCANS = {  # wide enough to bracket the active-space minimum
    "h4":   (0.60, 3.00, 0.02),
    "lih":  (1.20, 3.60, 0.02),
    "beh2": (1.00, 3.40, 0.02),
    "n2":   (0.90, 2.80, 0.02),
}

def e0(mol, R):
    H, _ = build_molecule_hamiltonian(mol, R)
    return float(np.linalg.eigvalsh(H.toarray()).min())

out = {}
for mol, (a, b, s) in SCANS.items():
    Rs = np.round(np.arange(a, b + 1e-9, s), 3)
    E = np.array([e0(mol, float(R)) for R in Rs])
    i = int(np.argmin(E))
    boundary = (i == 0 or i == len(Rs) - 1)
    out[mol] = float(Rs[i])
    print(f"{mol:5s} R_eq = {Rs[i]:.2f} Ang  E0={E[i]:.6f}  scan[{a},{b}]  "
          f"{'** STILL AT BOUNDARY **' if boundary else 'interior-min OK'}")
print("\nR_EQ =", {k: round(v, 2) for k, v in out.items()})
