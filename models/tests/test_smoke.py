"""Smoke tests: model forward pass, checkpoint round-trip, omega_op computation,
and the non-oracle property of the train-interp omega_op source.

Run from the models/ directory:  python -m pytest tests/ -q
"""

import numpy as np
import pytest
import torch

from fermionic_pipeline.models.observable_regressor import init_observable_regressor
from fermionic_pipeline.data.compute_omega_op import compute_omega_op
from fermionic_pipeline.eval.omega_source import OmegaOpSource

N_OBS, N_ORB, K_FOURIER = 12, 3, 8


def _tiny_model():
    torch.manual_seed(0)
    return init_observable_regressor(
        n_observables=N_OBS, d_hidden=32, n_layers=2, n_fourier=K_FOURIER,
        conditioned_frequencies=True, freq_net_hidden=16, freq_net_layers=2,
        n_orb_features=N_ORB, adaptive_bandwidth=True, omega_op_floor=2.0,
        soft_omega_floor=True, standardize_orb_energies=True,
        explicit_amplitude=True, amp_rank=4,
    )


def _tiny_inputs(batch=5):
    torch.manual_seed(1)
    rt = torch.rand(batch, 2) * torch.tensor([2.5, 300.0]) + torch.tensor([0.5, 0.0])
    orb = torch.randn(batch, N_ORB)
    omega_op = torch.full((batch,), 1.2)
    return rt, orb, omega_op


def test_forward_shape_and_finite():
    model = _tiny_model()
    rt, orb, omega_op = _tiny_inputs()
    out = model(rt, orb_energies=orb, omega_op=omega_op)
    assert out.shape == (5, N_OBS)
    assert torch.isfinite(out).all()


def test_checkpoint_roundtrip_identical():
    model = _tiny_model()
    rt, orb, omega_op = _tiny_inputs()
    ref = model(rt, orb_energies=orb, omega_op=omega_op)

    state = model.state_dict()
    model2 = _tiny_model()
    # perturb to prove the load matters
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(0.1)
    model2.load_state_dict(state)
    out = model2(rt, orb_energies=orb, omega_op=omega_op)
    assert torch.allclose(ref, out, atol=0, rtol=0)


def test_compute_omega_op_single_tone():
    t = np.arange(0.0, 100.0, 0.05)
    w0 = 1.5
    y = np.sin(w0 * t)[None, :, None] * np.ones((4, 1, 6))  # (n_R, n_t, K)
    omega_op = compute_omega_op(y, t, frac=0.99)
    assert omega_op.shape == (4,)
    # 99% of the Hann-windowed energy of a single tone sits at the tone
    assert np.all(np.abs(omega_op - w0) < 0.2)


def test_compute_omega_op_orders_two_tones():
    t = np.arange(0.0, 200.0, 0.05)
    slow = np.sin(0.5 * t)
    fast = 0.2 * np.sin(3.0 * t)
    y = np.stack([np.tile((slow)[:, None], (1, 6)),
                  np.tile((slow + fast)[:, None], (1, 6))])  # (2, n_t, 6)
    omega_op = compute_omega_op(y, t, frac=0.999)
    # adding high-frequency content must raise the ceiling
    assert omega_op[1] > omega_op[0]


class _FakeHandle:
    def __init__(self, R_values, omega_op):
        self.R_values = np.asarray(R_values, dtype=float)
        self.omega_op = np.asarray(omega_op, dtype=float)


def test_train_interp_is_non_oracle():
    """Corrupting omega_op at HELD-OUT geometries must not change train-interp values."""
    R = np.linspace(0.5, 3.0, 11)
    w_clean = 2.0 - 0.4 * R  # smooth ground truth
    train_idx = [0, 1, 2, 4, 5, 7, 8, 10]
    test_idx = [3, 6, 9]
    payload = {"train_r_indices": train_idx}

    src_clean = OmegaOpSource("train-interp", handle=_FakeHandle(R, w_clean), payload=payload)
    vals_clean = [src_clean.value(r_idx=i) for i in test_idx]

    w_poisoned = w_clean.copy()
    w_poisoned[test_idx] = 1e9  # oracle values destroyed
    src_pois = OmegaOpSource("train-interp", handle=_FakeHandle(R, w_poisoned), payload=payload)
    vals_pois = [src_pois.value(r_idx=i) for i in test_idx]

    assert np.allclose(vals_clean, vals_pois)
    # and the interpolation tracks the smooth ground truth closely
    assert np.allclose(vals_clean, w_clean[test_idx], atol=1e-6)


def test_dataset_mode_matches_legacy():
    R = np.linspace(0.5, 3.0, 6)
    w = np.linspace(2.0, 0.7, 6)
    src = OmegaOpSource("dataset", handle=_FakeHandle(R, w))
    assert src.value(r_idx=2) == pytest.approx(w[2])
