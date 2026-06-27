"""B9 3A (E1) — finite-shot sampler + the CRITICAL convergence self-check.

The matchgate-shadow target per Q is an analytic function of the per-qubit MARGINALS of the
rotated state probabilities. Finite-shot = draw S Born samples from those probabilities and use
EMPIRICAL marginals. This validates the sampler: as S->inf the empirical marginals (hence the
finite-shot target) converge to the exact (stored, S=inf) target, with error ~ 1/sqrt(S).
If this passes, the full finite-shot dataset = the same sampling inside the datagen's per-(Q,R,t)
loop (feasible on a COARSE grid; the fine grid is too expensive). Code pending Aniket review.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fermionic_pipeline.data.regression_dataset import (
    _precompute_decomposition, _apply_decomposition, _build_bit_array, majorana_2pt_keys,
)
from fermionic_pipeline.data.exact_conditional_dataset import (
    sample_signed_permutation_library, signed_permutation_to_matrix,
)
from fermionic_pipeline.data.generate_shadows import (
    build_molecule_hamiltonian, prepare_initial_state, molecule_n_electrons, time_evolve,
)

def exact_marginals(probs, bit_array):
    return (1.0 - bit_array).T @ probs            # (n_qubits,) P(qubit=0)

def empirical_marginals(probs, bit_array, S, rng):
    idx = rng.choice(len(probs), size=S, p=probs / probs.sum())
    return (1.0 - bit_array[idx]).mean(axis=0)    # (n_qubits,)

def main(mol="lih"):
    rng = np.random.default_rng(0)
    H, nq = build_molecule_hamiltonian(mol, 2.0)
    psi0, _ = prepare_initial_state(H, nq, n_electrons=molecule_n_electrons(mol))
    states = time_evolve(H, psi0, np.array([0.0, 5.0, 50.0]))
    perms, signs = sample_signed_permutation_library(2 * nq, 8, seed=42)
    decomps = [_precompute_decomposition(signed_permutation_to_matrix(perms[i], signs[i]).astype(np.float64)) for i in range(8)]
    bit_array = _build_bit_array(nq)
    print(f"[{mol}] convergence of empirical vs exact per-qubit marginals (avg over 3 t x 8 Q):")
    prev = None
    for S in [10, 100, 1000, 10000, 100000]:
        errs = []
        for t in [0.0, 5.0, 50.0]:
            psi = np.asarray(states[t], dtype=np.complex128).reshape(-1)
            for q in range(8):
                rot = _apply_decomposition(decomps[q], psi, nq)
                probs = np.abs(rot) ** 2
                ex = exact_marginals(probs, bit_array)
                em = empirical_marginals(probs, bit_array, S, rng)
                errs.append(np.mean(np.abs(em - ex)))
        e = float(np.mean(errs))
        rate = "" if prev is None else f"  (x{prev/e:.2f} for x10 S; sqrt-law -> ~3.16)"
        print(f"  S={S:>6}: mean|emp-exact marginal| = {e:.5f}{rate}")
        prev = e
    print("  PASS if error decreases ~3.16x per 10x S (1/sqrt(S)).")

if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
