import json

import h5py
import numpy as np
import torch

from fermionic_pipeline.data.regression_dataset import RegressionDatasetHandle, RegressionTorchDataset
from fermionic_pipeline.experiments.common import channels_to_covariance, covariance_to_channels
from fermionic_pipeline.data.majorana_observables import exact_majorana_channels
from fermionic_pipeline.experiments.geometry_splits import make_manifests
from fermionic_pipeline.experiments.line_interpolation_baseline import design
from fermionic_pipeline.experiments.spectral_metrics import compare_spectra
from fermionic_pipeline.training.regressor_trainer import resolve_geometry_split
from fermionic_pipeline.models.observable_regressor import init_observable_regressor


def tiny_h5(path):
    with h5py.File(path, "w") as f:
        f.attrs["n_atoms"] = 1; f.attrs["n_qubits"] = 1
        f.attrs["n_modes"] = 2; f.attrs["n_observables"] = 1
        f.create_dataset("R_values", data=np.arange(5.0))
        f.create_dataset("times", data=np.arange(4.0))
        f.create_dataset("observable_keys", data=np.array([[0, 1]]))
        f.create_dataset("expectations", data=np.arange(20.0).reshape(5, 4, 1))
        f.create_dataset("eigvals", data=np.zeros((5, 2)))


def test_dataset_time_subset(tmp_path):
    p = tmp_path / "x.h5"; tiny_h5(p)
    d = RegressionTorchDataset(RegressionDatasetHandle(p), r_indices=[2], t_indices=[1, 3])
    assert len(d) == 2
    assert d[0][0].tolist() == [2.0, 1.0]
    assert d[1][0].tolist() == [2.0, 3.0]


def test_explicit_split_and_complement(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"train": [0, 2, 4]}))
    train, test = resolve_geometry_split(5, 42, 0.2, train_file=p)
    assert train.tolist() == [0, 2, 4]
    assert test.tolist() == [1, 3]


def test_geometry_manifests_are_disjoint():
    R = np.arange(0.5, 1.51, 0.01)
    manifests = make_manifests(R, [5, 10], [0.05], [0.1], [1.0])
    assert manifests
    for _, train, test, _ in manifests:
        assert not len(np.intersect1d(train, test))
        assert len(np.union1d(train, test)) == len(R)


def test_exact_majorana_one_qubit_z():
    # -i gamma_0 gamma_1 = Z.
    psi = np.array([[1, 0], [0, 1]], dtype=complex).T  # columns |0>, |1>
    out = exact_majorana_channels(psi, [(0, 1)], 1)
    assert np.allclose(out[:, 0], [1, -1])


def test_covariance_channel_roundtrip():
    rng = np.random.default_rng(0)
    keys = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    x = rng.normal(size=(3, len(keys)))
    assert np.allclose(covariance_to_channels(channels_to_covariance(x, keys, 4), keys), x)


def test_line_design_and_spectral_identity():
    t = np.linspace(0, 100, 1001)
    w = np.array([0.7, 1.4])
    X = design(t, w)
    assert X.shape == (len(t), 5)
    D = np.stack([np.sin(0.7 * t), np.cos(1.4 * t)])
    m = compare_spectra(D, D, t, tolerance=0.02)
    assert m["spectral_wasserstein"] == 0.0
    assert m["recall"] == 1.0


def test_initial_condition_is_exact_and_ordered_bank_is_monotone():
    model = init_observable_regressor(
        n_observables=6, d_hidden=16, n_layers=2, n_fourier=8,
        conditioned_frequencies=True, n_orb_features=2, adaptive_bandwidth=True,
        explicit_amplitude=True, amp_rank=2, enforce_initial_condition=True,
        ordered_frequencies=True,
    )
    rt = torch.tensor([[1.0, 0.0], [1.2, 0.0]])
    orb = torch.randn(2, 2); ceiling = torch.tensor([2.0, 3.0])
    d0 = torch.randn(2, 6)
    out = model(rt, orb, ceiling, d0)
    assert torch.allclose(out, d0, atol=1e-5)
    increments = torch.nn.functional.softplus(model.freq_net(orb))
    sigma = torch.cumsum(increments, -1) / (increments.sum(-1, keepdim=True) + 1)
    assert torch.all(torch.diff(sigma, dim=-1) > 0)
