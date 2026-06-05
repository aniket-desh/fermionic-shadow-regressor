"""Bayesian optimization with a user-supplied function prior.

This module models an unknown target function ``g(x)`` as

    g(x) = f(x) + h(x)

where ``f(x)`` is a deterministic prior/educated guess and ``h(x)`` is a
zero-mean Gaussian-process residual. Observations update the posterior over
``g`` by learning only the residual ``g(x) - f(x)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.stats import norm

Acquisition = Literal[
    "expected_improvement",
    "probability_improvement",
    "confidence_bound",
    "posterior_variance",
]
Objective = Literal["maximize", "minimize"]


def _as_2d_x(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 0:
        return x.reshape(1, 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim == 2:
        return x
    raise ValueError(f"Expected x to be scalar, 1D, or 2D; got shape {x.shape}.")


def _as_1d_y(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        y = y.reshape(-1)
    return y


def _evaluate_mean(mean_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray) -> np.ndarray:
    values = np.asarray(mean_fn(x), dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    return values.reshape(-1)


def rbf_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
    *,
    length_scale: float | np.ndarray = 1.0,
    signal_variance: float = 1.0,
) -> np.ndarray:
    """Squared-exponential/RBF kernel.

    Args:
        x1: First input array of shape ``(n, d)``.
        x2: Second input array of shape ``(m, d)``.
        length_scale: Positive scalar or per-dimension length scale.
        signal_variance: Positive kernel amplitude squared.

    Returns:
        Kernel matrix of shape ``(n, m)``.
    """

    x1 = _as_2d_x(x1)
    x2 = _as_2d_x(x2)
    length_scale = np.asarray(length_scale, dtype=float)
    if np.any(length_scale <= 0):
        raise ValueError("length_scale must be positive.")
    if signal_variance <= 0:
        raise ValueError("signal_variance must be positive.")

    scaled_x1 = x1 / length_scale
    scaled_x2 = x2 / length_scale
    cross_term = np.einsum("id,jd->ij", scaled_x1, scaled_x2, optimize=True)
    sqdist = (
        np.sum(scaled_x1**2, axis=1)[:, None]
        + np.sum(scaled_x2**2, axis=1)[None, :]
        - 2.0 * cross_term
    )
    return signal_variance * np.exp(-0.5 * np.maximum(sqdist, 0.0))


@dataclass
class FunctionPriorGP:
    """Gaussian-process posterior using ``prior_mean(x)`` as the mean function.

    The posterior is exact GP regression with Gaussian observation noise.
    Use ``suggest_next`` to run Bayesian optimization over a finite candidate
    set, or ``sample_posterior`` to draw functions from the posterior.
    """

    prior_mean: Callable[[np.ndarray], np.ndarray]
    length_scale: float | np.ndarray = 1.0
    signal_variance: float = 1.0
    noise_variance: float = 1e-6
    jitter: float = 1e-10
    x_train: np.ndarray | None = field(default=None, init=False)
    y_train: np.ndarray | None = field(default=None, init=False)
    _cho_factor: tuple[np.ndarray, bool] | None = field(default=None, init=False)
    _alpha: np.ndarray | None = field(default=None, init=False)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FunctionPriorGP":
        """Fit the posterior to observations of ``g(x)``.

        Args:
            x: Inputs with shape ``(n,)`` for 1D or ``(n, d)`` for d-dimensional
                points.
            y: Observed values of the true function ``g`` at ``x``.
        """

        x = _as_2d_x(x)
        y = _as_1d_y(y)
        if len(x) != len(y):
            raise ValueError(f"x and y must have the same length: {len(x)} != {len(y)}")
        if self.noise_variance < 0:
            raise ValueError("noise_variance must be non-negative.")
        if self.jitter <= 0:
            raise ValueError("jitter must be positive.")

        self.x_train = x.copy()
        self.y_train = y.copy()
        self._refactor()
        return self

    def update(self, x_new: np.ndarray, y_new: np.ndarray) -> "FunctionPriorGP":
        """Append new observations and recompute the posterior."""

        x_new = _as_2d_x(x_new)
        y_new = _as_1d_y(y_new)
        if len(x_new) != len(y_new):
            raise ValueError(
                f"x_new and y_new must have the same length: {len(x_new)} != {len(y_new)}"
            )
        if self.x_train is None:
            return self.fit(x_new, y_new)

        self.x_train = np.vstack([self.x_train, x_new])
        self.y_train = np.concatenate([self.y_train, y_new])
        self._refactor()
        return self

    def predict(
        self,
        x: np.ndarray,
        *,
        return_cov: bool = False,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict posterior mean and uncertainty for ``g(x)``."""

        x = _as_2d_x(x)
        prior_at_x = _evaluate_mean(self.prior_mean, x)

        if self.x_train is None or self._cho_factor is None or self._alpha is None:
            cov = self._kernel(x, x)
            return self._format_prediction(prior_at_x, cov, return_cov, return_std)

        k_x_train = self._kernel(x, self.x_train)
        residual_mean = np.einsum("ij,j->i", k_x_train, self._alpha, optimize=True)
        posterior_mean = prior_at_x + residual_mean

        if not return_cov and not return_std:
            return posterior_mean

        lower = self._cho_factor[1]
        factor = self._cho_factor[0]
        v = solve_triangular(factor, k_x_train.T, lower=lower, check_finite=False)
        cov = self._kernel(x, x) - np.einsum("ij,jk->ik", v.T, v, optimize=True)
        cov = 0.5 * (cov + cov.T)
        return self._format_prediction(posterior_mean, cov, return_cov, return_std)

    def sample_posterior(
        self,
        x: np.ndarray,
        *,
        n_samples: int = 1,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw samples from the posterior over ``g(x)``.

        Returns an array with shape ``(n_samples, len(x))``.
        """

        mean, cov = self.predict(x, return_cov=True)
        cov = 0.5 * (cov + cov.T) + self.jitter * np.eye(len(mean))
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        transform = eigvecs * np.sqrt(eigvals)[None, :]
        rng = np.random.default_rng(random_state)
        draws = rng.standard_normal(size=(n_samples, len(mean)))
        residual_samples = np.einsum("sd,xd->sx", draws, transform, optimize=True)
        return mean[None, :] + residual_samples

    def acquisition(
        self,
        candidates: np.ndarray,
        *,
        kind: Acquisition = "expected_improvement",
        objective: Objective = "maximize",
        xi: float = 0.01,
        beta: float = 2.0,
    ) -> np.ndarray:
        """Score candidate points with a Bayesian-optimization acquisition."""

        candidates = _as_2d_x(candidates)
        mean, std = self.predict(candidates, return_std=True)
        std = np.maximum(std, 0.0)

        if kind == "posterior_variance":
            return std**2

        sign = _objective_sign(objective)
        scaled_mean = sign * mean

        if kind == "confidence_bound":
            return scaled_mean + beta * std

        if self.y_train is None:
            return np.full(len(candidates), np.inf)

        best = np.max(sign * self.y_train)
        improvement = scaled_mean - best - xi
        safe_std = np.maximum(std, 1e-15)
        z = improvement / safe_std

        if kind == "expected_improvement":
            score = improvement * norm.cdf(z) + safe_std * norm.pdf(z)
            return np.where(std > 0, score, 0.0)
        if kind == "probability_improvement":
            return norm.cdf(z)

        raise ValueError(f"Unknown acquisition kind: {kind}")

    def suggest_next(
        self,
        candidates: np.ndarray,
        *,
        kind: Acquisition = "expected_improvement",
        objective: Objective = "maximize",
        xi: float = 0.01,
        beta: float = 2.0,
        exclude_observed: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Return the best candidate and its acquisition score.

        This works over a finite candidate set. For continuous domains, first
        create candidates with ``random_candidates`` or your own design/grid.
        """

        candidates = _as_2d_x(candidates)
        scores = self.acquisition(
            candidates,
            kind=kind,
            objective=objective,
            xi=xi,
            beta=beta,
        )

        if exclude_observed and self.x_train is not None:
            observed = _observed_mask(candidates, self.x_train)
            scores = scores.copy()
            scores[observed] = -np.inf

        if np.all(np.isneginf(scores)):
            raise ValueError("No valid candidate remains after excluding observations.")

        best_idx = int(np.argmax(scores))
        return candidates[best_idx].copy(), float(scores[best_idx])

    def log_marginal_likelihood(self) -> float:
        """Return the GP residual log marginal likelihood."""

        if self.x_train is None or self.y_train is None or self._cho_factor is None:
            raise ValueError("Fit the model before computing log marginal likelihood.")
        residual = self.y_train - _evaluate_mean(self.prior_mean, self.x_train)
        factor = self._cho_factor[0]
        alpha = cho_solve(self._cho_factor, residual, check_finite=False)
        return float(
            -0.5 * residual @ alpha
            - np.sum(np.log(np.diag(factor)))
            - 0.5 * len(self.x_train) * np.log(2.0 * np.pi)
        )

    def random_candidates(
        self,
        bounds: np.ndarray,
        *,
        n_candidates: int,
        random_state: int | np.random.Generator | None = None,
    ) -> np.ndarray:
        """Sample uniform candidate points inside ``bounds``.

        Args:
            bounds: Array of shape ``(d, 2)`` with lower/upper bounds.
            n_candidates: Number of points to sample.
        """

        bounds = np.asarray(bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (d, 2).")
        if np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("Each upper bound must be greater than its lower bound.")
        rng = np.random.default_rng(random_state)
        low = bounds[:, 0]
        high = bounds[:, 1]
        return rng.uniform(low, high, size=(n_candidates, len(bounds)))

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return rbf_kernel(
            x1,
            x2,
            length_scale=self.length_scale,
            signal_variance=self.signal_variance,
        )

    def _refactor(self) -> None:
        if self.x_train is None or self.y_train is None:
            return
        residual = self.y_train - _evaluate_mean(self.prior_mean, self.x_train)
        if len(residual) != len(self.x_train):
            raise ValueError("prior_mean(x) must return one scalar value per input row.")

        train_cov = self._kernel(self.x_train, self.x_train)
        train_cov += (self.noise_variance + self.jitter) * np.eye(len(self.x_train))
        self._cho_factor = cho_factor(train_cov, lower=True, check_finite=False)
        self._alpha = cho_solve(self._cho_factor, residual, check_finite=False)

    @staticmethod
    def _format_prediction(
        mean: np.ndarray,
        cov: np.ndarray,
        return_cov: bool,
        return_std: bool,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if return_cov and return_std:
            raise ValueError("Use only one of return_cov=True or return_std=True.")
        if return_cov:
            return mean, cov
        if return_std:
            var = np.maximum(np.diag(cov), 0.0)
            return mean, np.sqrt(var)
        return mean


def _objective_sign(objective: Objective) -> float:
    if objective == "maximize":
        return 1.0
    if objective == "minimize":
        return -1.0
    raise ValueError(f"Unknown objective: {objective}")


def _observed_mask(candidates: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return np.any(
        np.all(np.isclose(candidates[:, None, :], observed[None, :, :]), axis=2),
        axis=1,
    )
