"""Audit and align the geometry-dependent active-orbital gauge.

Canonical MOs are tracked by maximum overlap, sign-fixed, and Procrustes-aligned
inside near-degenerate subspaces.  The degree-two Majorana covariance is then
rotated into this tracked basis and written as a training-compatible HDF5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

from fermionic_pipeline.data.generate_shadows import (
    _MOLECULE_SPECS, _molecule_coordinates, _parse_molecule, molecule_n_electrons,
)
from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.experiments.common import (
    channels_to_covariance, covariance_to_channels, write_json,
)


def active_mos(molecule, R):
    try:
        from pyscf import gto, scf
    except ModuleNotFoundError as exc:
        raise RuntimeError("orbital gauge alignment requires pyscf") from exc
    kind, payload = _parse_molecule(molecule)
    if kind == "hchain":
        atom = "; ".join(f"H {i * R} 0 0" for i in range(payload))
        charge, spin, n_active = 0, 0, payload
    else:
        spec = _MOLECULE_SPECS[payload]
        xyz = _molecule_coordinates(spec, R).reshape(-1, 3)
        atom = "; ".join(f"{s} {x} {y} {z}" for s, (x, y, z) in zip(spec["symbols"], xyz))
        charge, spin, n_active = spec["charge"], spec["mult"] - 1, spec["ao"]
        # Match generate_shadows.compute_hf_orbital_energies.
    mol = gto.M(atom=atom, basis="sto-3g", charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(mol); mf.kernel()
    start = 0 if kind == "hchain" else max(
        0, (mol.nelectron - molecule_n_electrons(molecule)) // 2,
    )
    end = start + n_active
    return mol, mf.mo_coeff[:, start:end].copy(), mf.mo_energy[start:end].copy()


def degenerate_blocks(energies, gap):
    blocks = []
    start = 0
    for i in range(len(energies) - 1):
        if abs(energies[i + 1] - energies[i]) > gap:
            if i + 1 - start > 1:
                blocks.append(np.arange(start, i + 1))
            start = i + 1
    if len(energies) - start > 1:
        blocks.append(np.arange(start, len(energies)))
    return blocks


def track_orbitals(molecule, R_values, degeneracy_gap):
    from pyscf import gto
    records = [active_mos(molecule, float(R)) for R in R_values]
    transforms = [np.eye(records[0][1].shape[1])]
    aligned = [records[0][1]]
    diagnostics = [{"R": float(R_values[0]), "permutation": list(range(len(transforms[0]))),
                    "sign_flips": 0, "min_matched_overlap": 1.0,
                    "offdiag_overlap_norm": 0.0, "procrustes_blocks": []}]
    for i in range(1, len(records)):
        prev_mol, _, _ = records[i - 1]
        mol, C, eps = records[i]
        Sx = gto.intor_cross("int1e_ovlp", prev_mol, mol)
        overlap = aligned[-1].T @ Sx @ C
        row, col = linear_sum_assignment(-np.abs(overlap))
        perm = col[np.argsort(row)]
        P = np.eye(len(perm))[:, perm]
        T = P.copy(); Cwork = C @ T; eps_work = eps[perm]
        diag = np.diag(aligned[-1].T @ Sx @ Cwork)
        signs = np.where(diag < 0, -1.0, 1.0)
        T = T @ np.diag(signs); Cwork = C @ T
        blocks_used = []
        for block in degenerate_blocks(eps_work, degeneracy_gap):
            M = aligned[-1][:, block].T @ Sx @ Cwork[:, block]
            U, _, Vt = np.linalg.svd(M)
            rot = Vt.T @ U.T
            T[:, block] = T[:, block] @ rot
            Cwork[:, block] = Cwork[:, block] @ rot
            blocks_used.append(block.tolist())
        final_overlap = aligned[-1].T @ Sx @ Cwork
        transforms.append(T); aligned.append(Cwork)
        diagnostics.append({
            "R": float(R_values[i]), "permutation": perm.tolist(),
            "sign_flips": int(np.sum(signs < 0)),
            "min_matched_overlap": float(np.min(np.abs(np.diag(final_overlap)))),
            "offdiag_overlap_norm": float(np.linalg.norm(final_overlap - np.diag(np.diag(final_overlap)))),
            "procrustes_blocks": blocks_used,
        })
    energies = []
    for (_, _, eps), T in zip(records, transforms):
        energies.append(np.diag(T.T @ np.diag(eps) @ T))
    return np.asarray(transforms), np.asarray(energies), diagnostics


def rotate_channels(expectations, keys, n_modes, spatial_transform):
    # PennyLane spin-orbital order is (orb0 alpha, orb0 beta, orb1 alpha, ...).
    spin = np.kron(spatial_transform, np.eye(2))
    majorana = np.kron(spin, np.eye(2))
    cov = channels_to_covariance(expectations, keys, n_modes)
    aligned = np.einsum("pa,tpq,qb->tab", majorana, cov, majorana, optimize=True)
    return covariance_to_channels(aligned, keys)


def smoothness(expectations, R):
    grad = np.gradient(expectations, R, axis=0)
    return np.mean(np.linalg.norm(grad, axis=1), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--degeneracy_gap", type=float, default=0.05,
                    help="Hartree threshold for within-subspace Procrustes alignment.")
    args = ap.parse_args()
    h = RegressionDatasetHandle(args.data_path)
    with h5py.File(args.data_path, "r") as src:
        molecule = src.attrs.get("molecule", f"h{h.n_atoms}")
        if isinstance(molecule, bytes): molecule = molecule.decode()
    transforms, aligned_eps, diagnostics = track_orbitals(
        molecule, h.R_values, args.degeneracy_gap,
    )
    aligned_D = np.empty_like(h.expectations)
    for i, T in enumerate(transforms):
        aligned_D[i] = rotate_channels(h.expectations[i], h.observable_keys, h.n_modes, T)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data_path, "r") as src, h5py.File(args.output, "w") as dst:
        for key, value in src.attrs.items(): dst.attrs[key] = value
        dst.attrs["orbital_gauge"] = "maximum-overlap-sign-procrustes"
        dst.attrs["gauge_source"] = args.data_path
        dst.attrs["gauge_degeneracy_gap"] = args.degeneracy_gap
        for key in src.keys():
            if key not in {"expectations", "hf_orbital_energies"}:
                src.copy(key, dst)
        dst.create_dataset("expectations", data=aligned_D, compression="lzf",
                           chunks=(1, len(h.times), h.n_observables))
        dst.create_dataset("hf_orbital_energies", data=aligned_eps)
        group = dst.create_group("orbital_gauge_alignment")
        group.create_dataset("canonical_to_aligned", data=transforms)
    raw_s = smoothness(h.expectations, h.R_values)
    aligned_s = smoothness(aligned_D, h.R_values)
    report = {
        "molecule": molecule, "data_path": args.data_path, "aligned_data_path": args.output,
        "degeneracy_gap": args.degeneracy_gap, "neighbor_diagnostics": diagnostics,
        "smoothness": [{"R": float(R), "raw": float(a), "aligned": float(b),
                        "ratio": float(b / max(a, 1e-30))}
                       for R, a, b in zip(h.R_values, raw_s, aligned_s)],
        "mean_smoothness_raw": float(raw_s.mean()),
        "mean_smoothness_aligned": float(aligned_s.mean()),
    }
    write_json(args.report, report)
    print(f"[done] aligned dataset -> {args.output}")
    print(f"[done] gauge report -> {args.report}")


if __name__ == "__main__":
    main()
