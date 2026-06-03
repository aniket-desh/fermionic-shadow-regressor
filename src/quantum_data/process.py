import numpy as np
import sys
import os

_PRED_FERM_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "symmetry_adjusted_classical_shadows_main",
    "fermion_shadows",
    "prediction",
)
_PRED_PAULI_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "symmetry_adjusted_classical_shadows_main",
    "pauli_shadows",
    "prediction",
)
if _PRED_FERM_DIR not in sys.path:
    sys.path.insert(0, _PRED_FERM_DIR)
if _PRED_PAULI_DIR not in sys.path:
    sys.path.insert(0, _PRED_PAULI_DIR)


def fermionic_signals(outcomes, pauli_terms) -> np.ndarray:
    # outcomes: list of (permutation, bitstring) from fermionic_shadows()
    # pauli_terms: list of (coeff, pauli_dict) from get_qubit_hamiltonian()
    # returns: 1d array of estimated expectation values, one per observable
    pass


def shadow_postprocess(signals, times, r=5, window=True, eps=1e-8) -> dict:
    # signals: array (n_observables, n_times) of estimated expectation values
    # times: 1d array of time points (uniform spacing)
    # r: number of dominant eigenvectors to keep
    # window: apply hann window before fft to reduce spectral leakage
    # returns: dict with keys omega, I (spectral intensity), C, eigvals, eigvecs_r
    K, T = signals.shape
    dt = float(times[1] - times[0])

    # standardize each signal
    x = signals - np.mean(signals, axis=1, keepdims=True)
    std = np.std(signals, axis=1, ddof=1, keepdims=True)
    std[std < eps] = 1.0
    D = x / std  # shape is (K, T), matrix D with rows as time series
    C = D.T @ D / K  # shape is (T, T), real symmetric

    # dominant eigenvectors of C, maximize average overlap with signals
    w, V = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    w, V = w[idx], V[:, idx]
    r = min(r, T)
    V_r = V[:, :r]  # shape is (T, r), keeping r dominant eigenvectors, signal subspace

    # shadow spectrum via spectral cross-correlation of the r eigenvectors
    # fourier transforming the dominant eigenvectors and taking
    # dominant singular value of cross-correlation at each frequency
    if window:
        win = np.hanning(T)
        Y = V_r.T * win  # shape (r, T)
    else:
        Y = V_r.T  # shape (r, T)

    F = np.fft.rfft(Y, axis=1)  # shape (r, F complex amplitudes)
    omega = (
        2 * np.pi * np.fft.rfftfreq(T, d=dt)  # angular frequencies
    )

    # for each freq bin m, build cross-correlation matrix
    # and take sv max (S) = spectral intensity at omega[m]

    intensity = np.sum(
        np.abs(F) ** 2, axis=0
    ).real  # shape (F,), robust estimator of spectral intensity I(omega)

    return {
        "C": C,
        "eigenvals": w,
        "eigenvecs_r": V_r,
        "omega": omega,
        "I": intensity,
    }
