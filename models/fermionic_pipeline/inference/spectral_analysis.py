"""
Downstream evaluation: matchgate estimator + Chan et al. spectral analysis.

Takes synthetic shadow snapshots (Q, b) and:
  1. Feeds them into the matchgate estimator to get Majorana expectations
  2. Assembles data matrix D in R^{N_o x N_T}
  3. Runs the spectral analysis pipeline (standardize, covariance, FFT)
"""

import os
import sys

import numpy as np
from itertools import combinations
from scipy.special import comb as binom

# Vendored prediction code (from Zhao-Miyake submodule)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.join(_SCRIPT_DIR, "..", "vendor")
if os.path.abspath(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, os.path.abspath(_VENDOR_DIR))


# ── Differentiable observable loss support ────────────────────────────

def compute_obs_matrix_elements(perm, signs, n_qubits, k=1):
    """Compute shadow estimator matrix elements for all bitstrings and operators.

    For each diagonal operator (2i, 2i+1), maps through permutation to get
    the target operator key, parity sign, and sign factor. The matrix element
    for bitstring b is: combined_sign * (-1)^{b_i}.

    Returns:
        obs_keys: sorted list of all C(2n, 2) operator tuples
        M: (2^n, n_obs) float array — M[b_idx, mu] = <b|estimator for mu|b>
           (without the shadow coefficient f_k^{-1})
    """
    n_modes = 2 * n_qubits
    dim = 1 << n_qubits
    perm = np.asarray(perm, dtype=np.intp)
    sgn = np.asarray(signs, dtype=np.float64) if signs is not None else np.ones(n_modes)

    obs_keys = list(combinations(range(n_modes), 2))
    key_to_idx = {k: i for i, k in enumerate(obs_keys)}
    n_obs = len(obs_keys)

    M = np.zeros((dim, n_obs), dtype=np.float64)

    # Precompute bitstring array
    bits = np.zeros((dim, n_qubits), dtype=np.int64)
    for b in range(dim):
        for p in range(n_qubits):
            bits[b, p] = (b >> (n_qubits - 1 - p)) & 1

    # Inverse permutation
    inv_perm = np.empty(n_modes, dtype=np.intp)
    inv_perm[perm] = np.arange(n_modes)

    diag_ops, degrees = _precompute_diagonal_ops(n_qubits, k)

    for diag_op, deg in zip(diag_ops, degrees):
        permuted = perm[diag_op]
        target = tuple(sorted(permuted))
        idx = key_to_idx.get(target)
        if idx is None:
            continue

        inv_applied = inv_perm[list(target)]
        inversions = sum(
            1 for i in range(len(inv_applied))
            for j in range(i + 1, len(inv_applied))
            if inv_applied[i] > inv_applied[j]
        )
        parity_sign = (-1) ** inversions
        sign_factor = float(np.prod(sgn[diag_op]))
        combined = sign_factor * parity_sign

        qubit_indices = diag_op[::2] // 2
        diag_vals = (-1.0) ** bits[:, qubit_indices].sum(axis=1)  # (dim,)
        M[:, idx] += combined * diag_vals

    return obs_keys, M


def batch_obs_matrix_elements(perms, signs, n_qubits, k=1):
    """Compute matrix elements for a batch of permutations.

    Args:
        perms: (B, 2n) int array
        signs: (B, 2n) int array (or None for unsigned)
        n_qubits: number of qubits

    Returns:
        obs_keys: list of operator tuples (same for all samples)
        M: (B, 2^n, n_obs) float array
    """
    B = len(perms)
    obs_keys = list(combinations(range(2 * n_qubits), 2))
    dim = 1 << n_qubits
    n_obs = len(obs_keys)
    M = np.zeros((B, dim, n_obs), dtype=np.float64)

    for s in range(B):
        sgn = signs[s] if signs is not None else None
        _, M_s = compute_obs_matrix_elements(perms[s], sgn, n_qubits, k)
        M[s] = M_s

    return obs_keys, M


# ── Vectorized batch estimator ────────────────────────────────────────

def _stack_outcomes(outcomes):
    """Stack snapshot data, supporting signed and unsigned permutations."""
    perms = np.array([o[0] for o in outcomes], dtype=np.intp)
    bits = np.array([o[1] for o in outcomes], dtype=np.intp)

    if len(outcomes[0]) >= 3:
        signs = np.array([o[2] for o in outcomes], dtype=np.intp)
    else:
        signs = np.ones_like(perms, dtype=np.intp)

    return perms, bits, signs

def _precompute_diagonal_ops(n_modes, k):
    """Precompute diagonal Majorana operator indices as numpy arrays.

    Returns:
        diag_ops: list of (2j,) int arrays — diagonal operator indices
        degrees: list of int — degree j for each operator
    """
    diag_ops = []
    degrees = []
    for j in range(1, k + 1):
        for P in combinations(range(n_modes), j):
            idx = []
            for p in P:
                idx.extend([2 * p, 2 * p + 1])
            diag_ops.append(np.array(idx, dtype=np.intp))
            degrees.append(j)
    return diag_ops, degrees


def _batch_parity(indices_batch):
    """Compute permutation parity for a batch of index sequences.

    Args:
        indices_batch: (B, L) int array — each row is a sequence of indices

    Returns:
        parities: (B,) int array — 0 (even) or 1 (odd)
    """
    B, L = indices_batch.shape
    # Count inversions: pairs (i,j) with i<j but val[i]>val[j]
    parities = np.zeros(B, dtype=np.intp)
    for i in range(L):
        for j in range(i + 1, L):
            parities += (indices_batch[:, i] > indices_batch[:, j]).astype(np.intp)
    return parities % 2


def _batch_diagonal_element(diag_op, bitstrings):
    """Compute <b|Gamma_diag|b> for a batch of bitstrings.

    Args:
        diag_op: (2j,) int array — diagonal Majorana indices
        bitstrings: (S, n) int array

    Returns:
        values: (S,) array of +1/-1
    """
    # For diagonal ops, only the even-indexed entries matter (qubit indices)
    qubit_indices = diag_op[::2] // 2
    # Product of (-1)^{b[q]} for each qubit q in the operator
    signs = bitstrings[:, qubit_indices]  # (S, j)
    # Each 1-bit flips the sign
    return (-1) ** signs.sum(axis=1)


def estimate_majorana_expectations_batch(outcomes, k=2):
    """Vectorized Majorana expectation estimator.

    Same interface as estimate_majorana_expectations but ~10-50x faster
    by replacing Python loops with numpy batch operations.

    Args:
        outcomes: list of [permutation, bitstring] pairs
        k: maximum fermionic locality (degree 2k)

    Returns:
        expectations: dict {tuple -> float}
    """
    # Stack into arrays
    perms, bits, signs = _stack_outcomes(outcomes)
    S = len(outcomes)
    n_modes = bits.shape[1]

    # Precompute inverse permutations: (S, 2n)
    inv_perms = np.zeros_like(perms)
    idx = np.arange(perms.shape[1])
    for s in range(S):
        inv_perms[s, perms[s]] = idx

    # Precompute diagonal ops and shadow coefficients
    diag_ops, degrees = _precompute_diagonal_ops(n_modes, k)
    shadow_coeffs = {}
    for j in range(1, k + 1):
        shadow_coeffs[j] = binom(2 * n_modes, 2 * j, exact=True) / binom(n_modes, j, exact=True)

    expectations = {}

    for op_idx, (diag_op, deg) in enumerate(zip(diag_ops, degrees)):
        # Target op for each sample: sort(perm[diag_op_indices])
        # diag_op has indices into the 2n Majorana modes
        permuted = perms[:, diag_op]  # (S, 2j) — permuted indices
        target_sorted = np.sort(permuted, axis=1)  # (S, 2j)

        # Parity: count inversions in inv_perm[target_op]
        # inv_perm applied to target_op indices
        inv_applied = np.take_along_axis(
            inv_perms,
            target_sorted,
            axis=1,
        )  # (S, 2j)
        parities = _batch_parity(inv_applied)  # (S,)
        parity_signs = (-1) ** parities  # (S,)
        permutation_signs = np.prod(np.take(signs, diag_op, axis=1), axis=1)

        # Diagonal matrix elements
        diag_vals = _batch_diagonal_element(diag_op, bits)  # (S,)

        # Accumulate per unique target op
        vals = permutation_signs * parity_signs * diag_vals  # (S,)

        # Group by target op key
        for s in range(S):
            key = tuple(target_sorted[s])
            if key not in expectations:
                expectations[key] = 0.0
            expectations[key] += vals[s]

    # Apply shadow coefficients and normalize
    for key in expectations:
        j = len(key) // 2
        expectations[key] *= shadow_coeffs[j] / S

    return expectations


# ── Parallel signal matrix construction ───────────────────────────────

def _estimate_at_time(args):
    """Worker function for parallel signal matrix construction."""
    t, outcomes, k = args
    return t, estimate_majorana_expectations_batch(outcomes, k=k)


def build_signal_matrix(shadow_outcomes_by_time, times, observable_keys=None,
                        k=2, n_workers=None):
    """Build data matrix D from shadow snapshots at multiple time points.

    D[i, j] = <O_i>(t_j) estimated from shadows at time t_j.

    Args:
        shadow_outcomes_by_time: dict mapping t -> list of [perm, bits]
        times: sorted array of time values
        observable_keys: list of Majorana operator tuples to track.
            If None, uses all operators from the first time point.
        k: maximum fermionic locality
        n_workers: number of parallel workers (None = serial)

    Returns:
        D: (N_o, N_T) signal matrix
        obs_keys: list of observable keys used
    """
    N_T = len(times)

    # Discover observable keys from first time point
    first_expectations = estimate_majorana_expectations_batch(
        shadow_outcomes_by_time[times[0]], k=k
    )
    if observable_keys is None:
        observable_keys = sorted(first_expectations.keys())

    N_o = len(observable_keys)
    key_to_idx = {key: i for i, key in enumerate(observable_keys)}
    D = np.zeros((N_o, N_T))

    # Fill first column from already-computed result
    for key, val in first_expectations.items():
        if key in key_to_idx:
            D[key_to_idx[key], 0] = val

    # Remaining time points
    remaining_args = [
        (t, shadow_outcomes_by_time[t], k) for t in times[1:]
    ]

    if n_workers is not None and n_workers > 1:
        from multiprocessing import Pool
        with Pool(n_workers) as pool:
            for i, (t, exp) in enumerate(pool.imap(_estimate_at_time, remaining_args)):
                j = i + 1  # offset by 1 since we already did times[0]
                for key, val in exp.items():
                    if key in key_to_idx:
                        D[key_to_idx[key], j] = val
                if (j + 1) % 500 == 0:
                    print(f"    signal matrix: {j + 1}/{N_T} time points", flush=True)
    else:
        for i, (t, outcomes, _k) in enumerate(remaining_args):
            exp = estimate_majorana_expectations_batch(outcomes, k=k)
            j = i + 1
            for key, val in exp.items():
                if key in key_to_idx:
                    D[key_to_idx[key], j] = val
            if (j + 1) % 500 == 0:
                print(f"    signal matrix: {j + 1}/{N_T} time points", flush=True)

    return D, observable_keys


def _ljung_box_screen(D, p_threshold=0.06):
    """Pre-screen D matrix rows using the Ljung-Box autocorrelation test.

    Filters out observables whose time series are indistinguishable from
    white noise. Chan et al. Eq. (5) / Section II.C: only signals that
    pass a Ljung-Box test (p < p_threshold) contain statistically
    significant temporal structure and should contribute to C.

    The Ljung-Box Q statistic at lag h for a length-T series is:
        Q(h) = T(T+2) sum_{k=1}^{h} r_k^2 / (T-k)
    where r_k is the sample autocorrelation at lag k. Under H0 (white
    noise), Q(h) ~ chi2(h). We reject H0 (keep the signal) if p < threshold.

    Args:
        D: (N_o, N_T) signal matrix (already standardized)
        p_threshold: maximum p-value to keep (Chan uses 0.06)

    Returns:
        D_screened: (N_kept, N_T) filtered signal matrix
        kept_mask: (N_o,) boolean mask of retained rows
    """
    from scipy.stats import chi2

    N_o, N_T = D.shape
    kept = np.zeros(N_o, dtype=bool)
    n_lags = min(10, N_T // 4)

    for i in range(N_o):
        row = D[i]
        if np.std(row) < 1e-12:
            continue

        # Sample autocorrelation at lags 1..n_lags
        mean = row.mean()
        centered = row - mean
        var = np.dot(centered, centered) / N_T
        if var < 1e-20:
            continue

        # Ljung-Box Q statistic
        Q = 0.0
        for k in range(1, n_lags + 1):
            r_k = np.dot(centered[:N_T - k], centered[k:]) / (N_T * var)
            Q += r_k ** 2 / (N_T - k)
        Q *= N_T * (N_T + 2)

        # p-value from chi2 distribution with n_lags degrees of freedom
        p_value = 1.0 - chi2.cdf(Q, df=n_lags)
        if p_value < p_threshold:
            kept[i] = True

    return D[kept], kept


def spectral_analysis(D, times, r=None, window="hann", ljung_box_p=None):
    """Chan et al. spectral analysis pipeline.

    1. Standardize each row of D
    2. (Optional) Ljung-Box pre-screen to remove noise-dominated rows
    3. Form covariance C = D^T D
    4. Extract dominant eigenvectors
    5. FFT to get spectral peaks (unweighted, per Chan)

    Args:
        D: (N_o, N_T) signal matrix
        times: (N_T,) time values
        r: number of dominant eigenvectors to use (default: min(10, N_T//2))
        window: window function for FFT ('hann', 'hamming', or None)
        ljung_box_p: p-value threshold for Ljung-Box screening.
            If None, no screening is applied. Chan uses 0.06.

    Returns:
        omega: frequency axis (positive only)
        spectrum: spectral intensity I(omega)
        eigvals: eigenvalues of C
    """
    N_o, N_T = D.shape

    if r is None:
        r = min(10, N_T // 2)

    # Standardize rows (Chan Eq. 5)
    mu = D.mean(axis=1, keepdims=True)
    sigma = D.std(axis=1, ddof=1, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    D_std = (D - mu) / sigma

    # Ljung-Box pre-screening: remove noise-dominated observables
    if ljung_box_p is not None:
        D_std, kept = _ljung_box_screen(D_std, p_threshold=ljung_box_p)
        n_kept = D_std.shape[0]
        print(f"    Ljung-Box screening: {n_kept}/{N_o} observables retained (p < {ljung_box_p})", flush=True)
        if n_kept == 0:
            # All observables are noise — return flat spectrum
            dt = times[1] - times[0] if len(times) > 1 else 1.0
            omega = 2 * np.pi * np.fft.rfftfreq(N_T, d=dt)
            return omega, np.zeros_like(omega), np.array([])

    # Covariance matrix (time × time)
    K = D_std.shape[0]
    C = D_std.T @ D_std  # (N_T, N_T)

    # Eigen decomposition — top-r eigenvectors
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1][:r]
    V = eigvecs[:, idx]  # (N_T, r)

    # FFT of windowed eigenvectors (Chan Section II.C)
    dt = times[1] - times[0] if len(times) > 1 else 1.0

    if window == "hann":
        w = np.hanning(N_T)
    elif window == "hamming":
        w = np.hamming(N_T)
    else:
        w = np.ones(N_T)

    # Apply window and rfft (positive frequencies only)
    Y = V.T * w  # (r, N_T)
    F = np.fft.rfft(Y, axis=1)  # (r, N_T//2+1) complex
    omega = 2 * np.pi * np.fft.rfftfreq(N_T, d=dt)

    # Unweighted sum of |FFT|^2 across eigenvectors (Chan / notebook)
    spectrum = np.sum(np.abs(F) ** 2, axis=0).real

    return omega, spectrum, eigvals


def extract_peaks(omega, spectrum, n_peaks=5, min_height_frac=0.1):
    """Extract peak frequencies from spectrum.

    Args:
        omega: frequency axis
        spectrum: spectral intensity
        n_peaks: maximum number of peaks to return
        min_height_frac: minimum peak height as fraction of max

    Returns:
        peak_freqs: array of peak frequencies
        peak_heights: array of peak intensities
    """
    from scipy.signal import find_peaks

    threshold = min_height_frac * spectrum.max()
    peaks, properties = find_peaks(spectrum, height=threshold)

    if len(peaks) == 0:
        return np.array([]), np.array([])

    # Sort by height
    heights = properties["peak_heights"]
    order = np.argsort(heights)[::-1][:n_peaks]

    return omega[peaks[order]], heights[order]
