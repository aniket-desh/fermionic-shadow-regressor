"""Low-rank, dictionary-usage, and one-particle physicality audits."""
from __future__ import annotations

import argparse

import numpy as np
import torch
import h5py

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle
from fermionic_pipeline.data.generate_shadows import molecule_n_electrons
from fermionic_pipeline.eval.omega_source import OmegaOpSource
from fermionic_pipeline.eval.regressor_eval import initial_values_for, predict_signal_matrix
from fermionic_pipeline.experiments.common import channels_to_covariance, write_json
from fermionic_pipeline.training.regressor_trainer import load_checkpoint_model


def one_rdm_from_majorana(D, keys, n_modes):
    """D: (T,K) of -i<gamma_p gamma_q>; return (T,n,n) complex 1-RDM."""
    A = channels_to_covariance(D, keys, n_modes)
    T = len(D); n = n_modes // 2
    corr = 1j * A.astype(np.complex128)
    diag = np.arange(n_modes); corr[:, diag, diag] = 1.0
    rdm = np.empty((T, n, n), dtype=np.complex128)
    for p in range(n):
        xp, yp = 2 * p, 2 * p + 1
        for q in range(n):
            xq, yq = 2 * q, 2 * q + 1
            rdm[:, p, q] = 0.25 * (corr[:, xp, xq] + 1j * corr[:, xp, yq]
                                           - 1j * corr[:, yp, xq] + corr[:, yp, yq])
    return rdm


def physicality(D, keys, n_modes, expected_particles):
    rdm = one_rdm_from_majorana(D, keys, n_modes)
    herm_err = np.max(np.abs(rdm - np.swapaxes(rdm.conj(), 1, 2)), axis=(1, 2))
    eig = np.linalg.eigvalsh(0.5 * (rdm + np.swapaxes(rdm.conj(), 1, 2)))
    trace = np.real(np.trace(rdm, axis1=1, axis2=2))
    return {
        "max_hermiticity_error": float(herm_err.max()),
        "mean_trace_error": float(np.mean(np.abs(trace - expected_particles))),
        "max_trace_error": float(np.max(np.abs(trace - expected_particles))),
        "fraction_eigenvalues_outside_0_1": float(np.mean((eig < -1e-6) | (eig > 1 + 1e-6))),
        "min_eigenvalue": float(eig.min()), "max_eigenvalue": float(eig.max()),
    }


@torch.no_grad()
def frequency_usage(model, R, orb, omega_op, device):
    cfg = model.config
    if not cfg.explicit_amplitude:
        return None
    rt = torch.tensor([[R, 0.0]], dtype=torch.float32, device=device)
    x = torch.tensor(orb[None], dtype=torch.float32, device=device) if orb is not None else rt[:, :1]
    if cfg.standardize_orb_energies and orb is not None:
        x = (x - model.orb_mean) / model.orb_std
    amp = model.amp_net(x)[0]
    K, N, r = cfg.n_fourier, cfg.n_observables, cfg.amp_rank
    if r > 0 and r < min(K, N):
        aU, aV, bU, bV = torch.split(amp, [K*r, r*N, K*r, r*N])
        a = aU.view(K, r) @ aV.view(r, N)
        b = bU.view(K, r) @ bV.view(r, N)
    else:
        a, b = amp.view(2, K, N)
    energy = (a.square() + b.square()).sum(dim=1)
    frac = energy / energy.sum().clamp_min(1e-30)
    effective = torch.exp(-(frac * torch.log(frac.clamp_min(1e-30))).sum())
    # Reproduce the adaptive frequency path without synthesizing all outputs.
    if cfg.adaptive_bandwidth:
        floor = cfg.omega_op_floor
        w_eff = floor + torch.nn.functional.softplus(
            torch.tensor([omega_op - floor], device=device), beta=cfg.soft_omega_beta,
        ) if cfg.soft_omega_floor and floor > 0 else torch.tensor([max(omega_op, floor)], device=device)
        omega = w_eff[:, None] * torch.sigmoid(model.freq_net(x))
    else:
        omega = model.omega_base[None] + (model.freq_net(x) if model.freq_net is not None else 0)
    order = torch.argsort(frac, descending=True)[:10]
    spacing = torch.diff(torch.sort(omega[0]).values)
    collision_tol = 1e-3 * omega[0].max().clamp_min(1e-12)
    return {"effective_frequency_count": float(effective.cpu()),
            "active_above_1pct": int((frac > 0.01).sum().cpu()),
            "min_frequency_spacing": float(spacing.min().cpu()),
            "near_collisions": int((spacing < collision_tol).sum().cpu()),
            "top_frequencies": omega[0, order].cpu().tolist(),
            "top_amplitude_fractions": frac[order].cpu().tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    h = RegressionDatasetHandle(args.data_path)
    model, payload = load_checkpoint_model(args.checkpoint, device=device)
    src = OmegaOpSource("train-interp", handle=h, payload=payload)
    with h5py.File(args.data_path, "r") as f:
        molecule = f.attrs.get("molecule", f"h{h.n_atoms}")
        if isinstance(molecule, bytes): molecule = molecule.decode()
    expected_particles = molecule_n_electrons(molecule)
    rows = []
    for i in payload.get("test_r_indices", range(len(h.R_values))):
        R = float(h.R_values[i]); orb = h.hf_orbital_energies[i] if h.hf_orbital_energies is not None else None
        omega_op = src.value(r_idx=i)
        d0 = initial_values_for(h, model)
        pred = predict_signal_matrix(model, R, h.times, device, orb, omega_op, d0).T
        ref = h.expectations[i]
        s = np.linalg.svd(ref - ref.mean(axis=0, keepdims=True), compute_uv=False)
        power = s ** 2 / max(float(np.sum(s ** 2)), 1e-30)
        rank99 = int(np.searchsorted(np.cumsum(power), 0.99) + 1)
        rows.append({"R": R, "rank_99pct": rank99, "leading_singular_values": s[:20].tolist(),
                     "exact_physicality": physicality(ref, h.observable_keys, h.n_modes, expected_particles),
                     "model_physicality": physicality(pred, h.observable_keys, h.n_modes, expected_particles),
                     "frequency_usage": frequency_usage(model, R, orb, omega_op, device)})
        print(f"[R={R:.4f}] rank99={rank99} active_freq="
              f"{rows[-1]['frequency_usage']['active_above_1pct'] if rows[-1]['frequency_usage'] else 'n/a'}")
    write_json(args.output, {"data_path": args.data_path, "checkpoint": args.checkpoint,
                             "expected_particles": expected_particles, "results": rows})
    print(f"[done] architecture audit -> {args.output}")


if __name__ == "__main__":
    main()
