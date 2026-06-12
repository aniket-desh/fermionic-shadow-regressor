"""Sources for the adaptive-bandwidth ceiling omega_op(R) at evaluation time.

The dataset's ``omega_op`` field is computed from each geometry's exact target
signal (see ``data/compute_omega_op.py``). Feeding ``handle.omega_op[r_idx]``
to the model at a *held-out* geometry therefore leaks a summary statistic of
the very signal being predicted. ``OmegaOpSource("train-interp", ...)``
removes that leak: the ceiling at an evaluated geometry is obtained by 1-D
interpolation, in R, over the ceilings of the checkpoint's TRAINING geometries
only. Out-of-range R is clamped to the edge training values, so extended-grid
(extrapolation) evaluations receive a conservative, non-oracle ceiling.

Modes:
    dataset       legacy behaviour: the evaluated geometry's own dataset value
                  (an oracle input at held-out geometries).
    train-interp  non-oracle: interpolated from training geometries only.
"""

from __future__ import annotations

import numpy as np

MODES = ("dataset", "train-interp")


class OmegaOpSource:
    """Resolves omega_op for an evaluated geometry.

    Args:
        mode: one of ``MODES``.
        handle: the dataset handle being EVALUATED (used for ``dataset`` mode
            and to look up R when only an index is given).
        payload: checkpoint payload; ``train-interp`` requires its
            ``train_r_indices``.
        train_handle: the dataset the checkpoint was TRAINED on, used as the
            interpolation table. Defaults to ``handle``; pass explicitly when
            evaluating on a different grid (e.g. the extrapolation heatmap).
    """

    def __init__(self, mode, handle=None, payload=None, train_handle=None):
        if mode not in MODES:
            raise ValueError(f"omega_op source mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.handle = handle
        if mode == "train-interp":
            src = train_handle if train_handle is not None else handle
            if payload is None or "train_r_indices" not in payload:
                raise ValueError(
                    "train-interp requires a checkpoint payload containing train_r_indices"
                )
            if src is None or src.omega_op is None:
                raise ValueError("train-interp requires a training dataset with an omega_op field")
            idx = np.asarray(payload["train_r_indices"], dtype=int)
            R_train = np.asarray(src.R_values, dtype=float)[idx]
            w_train = np.asarray(src.omega_op, dtype=float)[idx]
            order = np.argsort(R_train)
            self._R = R_train[order]
            self._w = w_train[order]

    def value(self, r_idx=None, R=None):
        """omega_op for geometry ``r_idx`` of ``handle`` (or explicit ``R``)."""
        if self.mode == "dataset":
            if self.handle is None or self.handle.omega_op is None:
                return None
            return float(self.handle.omega_op[r_idx])
        if R is None:
            R = float(self.handle.R_values[r_idx])
        return float(np.interp(R, self._R, self._w))
