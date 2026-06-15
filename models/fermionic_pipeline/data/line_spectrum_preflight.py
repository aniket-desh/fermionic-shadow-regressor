"""Line-spectrum pre-flight for the sampling-grid design of a new molecule.

The propositions tie temporal sampling to the molecule's observable line
spectrum: you can only resolve two Bohr lines if t_max >= 2*pi / (line spacing),
and you only avoid aliasing if dt <= pi / w_max. H2's failure (sub-resolution
line pairs at t_max=300) institutionalized this check — run it BEFORE training
any new molecule.

This computes the EXACT, observable-weighted line spectrum (not a population
proxy): the 120 degree-2 Majorana signals y_a(t) = <psi(t)| gamma_mu gamma_nu |psi(t)>
have lines at the Bohr frequencies w_ij = E_j - E_i with aggregate power
    P_ij = |c_i|^2 |c_j|^2 * sum_{mu<nu} |<i| gamma_mu gamma_nu |j>|^2 ,
c = V^dagger psi_0 the eigenbasis amplitudes of the SAME HF + symmetry-breaking
initial state the pipeline prepares. Reusing prepare_initial_state / the JW
gamma application keeps this faithful to the trained signals.

Usage (run from models/):
    python -m fermionic_pipeline.data.line_spectrum_preflight \
        --molecule lih --r-start 1.0 --r-end 3.2 --r-step 0.05 \
        --t-max 1200 --dt 0.25
    python -m fermionic_pipeline.data.line_spectrum_preflight --molecule h4 \
        --r-start 0.5 --r-end 3.0 --r-step 0.1 --t-max 300 --dt 0.05   # validation
"""

from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np

from fermionic_pipeline.data.generate_shadows import (
    build_molecule_hamiltonian,
    molecule_n_electrons,
    prepare_initial_state,
)
from fermionic_pipeline.data.exact_conditional_dataset import _apply_single_majorana


# ── molecule build: delegate to the shared registry (single source of truth) ──

def build_molecule(molecule, R):
    """Return (H_sparse, n_qubits, n_electrons) for the molecule at R, using the
    SAME builder/registry the datagen pipeline uses (generate_shadows)."""
    H_sparse, n_qubits = build_molecule_hamiltonian(molecule, R)
    return H_sparse, n_qubits, molecule_n_electrons(molecule)


def build_majorana_matrices(n_qubits):
    """The 2n Majorana operators as dense matrices, via the pipeline's JW
    convention (_apply_single_majorana). R-independent: build once."""
    dim = 2 ** n_qubits
    n_modes = 2 * n_qubits
    basis = np.eye(dim, dtype=np.complex128)
    mats = []
    for mu in range(n_modes):
        cols = [_apply_single_majorana(basis[:, k], mu, n_qubits) for k in range(dim)]
        mats.append(np.stack(cols, axis=1))
    return mats  # list of (dim, dim)


def line_spectrum(H_sparse, n_electrons, gammas, weight_thresh=1e-12):
    """Exact observable-weighted line spectrum of the 120-channel signal.

    Returns dict with sorted (freq, weight) arrays and aggregate stats.
    """
    n_qubits = int(np.log2(H_sparse.shape[0]))
    psi_0, _ = prepare_initial_state(H_sparse, n_qubits, n_electrons)

    H_dense = H_sparse.toarray()
    E, V = np.linalg.eigh(H_dense)
    c = V.conj().T @ psi_0
    pop = np.abs(c) ** 2  # eigenstate populations

    # Prune negligibly-populated eigenstates: lines need a populated pair on
    # BOTH ends, so this is exact to weight_thresh for the dominant analysis.
    keep = np.where(pop > weight_thresh) [0]
    Vk = V[:, keep]
    Ek = E[keep]
    popk = pop[keep]

    # GV[mu] = gamma_mu V  (columns gamma_mu|i>); <i|g_mu g_nu|j> = (GV_mu^H GV_nu)[i,j]
    GV = [g @ Vk for g in gammas]
    nk = len(keep)
    W = np.zeros((nk, nk))
    for mu, nu in combinations(range(len(gammas)), 2):
        M = GV[mu].conj().T @ GV[nu]
        W += np.abs(M) ** 2

    # AC lines from i<j pairs; freq |E_j - E_i|, weight pop_i pop_j W_ij
    iu, ju = np.triu_indices(nk, k=1)
    freqs = np.abs(Ek[ju] - Ek[iu])
    wts = popk[iu] * popk[ju] * W[iu, ju]
    return {
        "freqs": freqs, "weights": wts,
        "n_eigs_kept": nk, "n_eigs_total": len(E),
    }


def aggregate_lines(freqs, weights, round_dp):
    """Bin lines by rounded frequency; return sorted-by-freq (freq, weight)."""
    fr = np.round(freqs, round_dp)
    order = np.argsort(fr)
    fr, w = fr[order], weights[order]
    out_f, out_w = [], []
    i = 0
    while i < len(fr):
        j = i
        while j < len(fr) and fr[j] == fr[i]:
            j += 1
        out_f.append(fr[i]); out_w.append(w[i:j].sum())
        i = j
    return np.array(out_f), np.array(out_w)


def dominant_stats(fr, w, bw_mass=0.99, strong_frac=0.05):
    """Two distinct, physically-separate constraints:

    - bw99: the 99%-power bandwidth EDGE (lines sorted by frequency, cumulative
      power -> bw_mass). This is the highest frequency carrying real power and
      sets the Nyquist (aliasing) constraint. ~ compute_omega_op's omega.
    - dw_strong: the minimum spacing among STRONG lines only — lines each
      individually carrying >= strong_frac of total power. Two CO-DOMINANT lines
      closer than 2*pi/t_max is the H2 catastrophic-failure signature; a tail of
      weak unresolved lines just adds harmless ripple, so it is excluded.
    """
    total = w.sum()
    if total <= 0:
        return dict(bw99=0.0, dw_strong=np.inf, n_strong=0, top=[])
    # bandwidth edge: cumulative power vs frequency
    o = np.argsort(fr)
    f_sorted, w_sorted = fr[o], w[o]
    cum = np.cumsum(w_sorted) / total
    bw99 = float(f_sorted[np.searchsorted(cum, bw_mass)])
    # strong lines for resolution
    strong = fr[w >= strong_frac * total]
    n_strong = int(strong.size)
    sf = np.sort(strong)
    dw_strong = float(np.min(np.diff(sf))) if n_strong > 1 else np.inf
    top = sorted(zip(fr, w / total), key=lambda x: -x[1])[:6]
    return dict(bw99=bw99, dw_strong=dw_strong, n_strong=n_strong, top=top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", required=True)
    ap.add_argument("--r-start", type=float, required=True)
    ap.add_argument("--r-end", type=float, required=True)
    ap.add_argument("--r-step", type=float, default=0.05)
    ap.add_argument("--round-dp", type=int, default=4, help="freq rounding decimals")
    ap.add_argument("--bw-mass", type=float, default=0.99, help="bandwidth-edge power fraction")
    ap.add_argument("--strong-frac", type=float, default=0.05,
                    help="a line is 'strong' (must be resolved) if it carries >= this power fraction")
    ap.add_argument("--t-max", type=float, default=None, help="candidate grid horizon to test")
    ap.add_argument("--dt", type=float, default=None, help="candidate grid step to test")
    ap.add_argument("--nyquist-safety", type=float, default=2.5)
    args = ap.parse_args()

    Rs = np.round(np.arange(args.r_start, args.r_end + args.r_step / 2, args.r_step), 6)

    # operators are R-independent — build once from the first geometry's size
    _, n_qubits0, _ = build_molecule(args.molecule, float(Rs[0]))
    print(f"[{args.molecule}] n_qubits={n_qubits0}  n_modes={2*n_qubits0}  "
          f"channels={len(list(combinations(range(2*n_qubits0), 2)))}")
    gammas = build_majorana_matrices(n_qubits0)

    res_grid = 2 * np.pi / args.t_max if args.t_max else None
    nyq = np.pi / args.dt if args.dt else None
    if args.t_max and args.dt:
        print(f"candidate grid: t_max={args.t_max} (resolution 2pi/t_max={res_grid:.4f})  "
              f"dt={args.dt} (Nyquist pi/dt={nyq:.3f})\n")

    hdr = f"{'R':>6} {'bw99':>8} {'dw_str':>8} {'n_str':>5} {'t_need':>8} {'dt_safe':>8}"
    if args.t_max and args.dt:
        hdr += "  verdict"
    print(hdr)

    all_bw, all_dws, fails = [], [], []
    for R in Rs:
        H_sparse, nq, ne = build_molecule(args.molecule, float(R))
        assert nq == n_qubits0, f"qubit count changed at R={R}"
        ls = line_spectrum(H_sparse, ne, gammas)
        fr, w = aggregate_lines(ls["freqs"], ls["weights"], args.round_dp)
        st = dominant_stats(fr, w, bw_mass=args.bw_mass, strong_frac=args.strong_frac)
        bw99, dws = st["bw99"], st["dw_strong"]
        all_bw.append(bw99)
        all_dws.append(dws)
        t_need = 2 * np.pi / dws if np.isfinite(dws) and dws > 0 else 0.0  # 0 => no strong pair to resolve
        dt_safe = np.pi / (args.nyquist_safety * bw99) if bw99 > 0 else np.inf
        row = f"{R:6.2f} {bw99:8.4f} {dws:8.4f} {st['n_strong']:5d} {t_need:8.1f} {dt_safe:8.3f}"
        if args.t_max and args.dt:
            res_fail = np.isfinite(dws) and dws < res_grid   # co-dominant pair sub-resolution
            nyq_fail = bw99 > nyq
            ok = not (res_fail or nyq_fail)
            row += f"  {'PASS' if ok else 'FAIL'}"
            if not ok:
                fails.append((R, res_fail, nyq_fail))
        print(row)

    print("\n=== box summary ===")
    print(f"  max bw99 over box    = {max(all_bw):.4f}  -> need dt <= pi/bw99 = {np.pi/max(all_bw):.3f}")
    finite_dw = [d for d in all_dws if np.isfinite(d)]
    if finite_dw:
        binding = min(finite_dw)
        print(f"  min dw_strong (codom)= {binding:.4f}  -> need t_max >= 2pi/dw = {2*np.pi/binding:.1f}")
    else:
        print("  no co-dominant strong pairs anywhere -> resolution not binding in this box")
    if args.t_max and args.dt:
        if not fails:
            print(f"  ALL {len(Rs)} geometries PASS at t_max={args.t_max}, dt={args.dt}")
        else:
            print(f"  {len(fails)}/{len(Rs)} FAIL: "
                  f"{sum(f[1] for f in fails)} unresolved co-dominant (dw<res), "
                  f"{sum(f[2] for f in fails)} aliased (bw99>Nyquist)")
            print("  failing R:", [round(float(f[0]), 2) for f in fails])


if __name__ == "__main__":
    main()
