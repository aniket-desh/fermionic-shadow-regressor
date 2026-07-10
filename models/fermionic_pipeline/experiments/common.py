from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def load_split(path):
    with open(path) as f:
        obj = json.load(f)
    train = np.asarray(obj["train"], dtype=int)
    test = np.asarray(obj["test"], dtype=int)
    if len(np.intersect1d(train, test)):
        raise ValueError(f"overlapping split in {path}")
    return train, test, obj


def temporal_pearsons(pred, target):
    """Per-channel Pearson for arrays shaped (channels, times)."""
    p = pred - pred.mean(axis=1, keepdims=True)
    t = target - target.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    out = np.full(len(p), np.nan)
    good = denom > 1e-12
    out[good] = np.sum(p[good] * t[good], axis=1) / denom[good]
    return out


def channels_to_covariance(values, keys, n_modes):
    """Convert (..., K) unique-pair channels to (..., n_modes, n_modes)."""
    values = np.asarray(values)
    out = np.zeros(values.shape[:-1] + (n_modes, n_modes), dtype=values.dtype)
    for k, (p, q) in enumerate(keys):
        out[..., p, q] = values[..., k]
        out[..., q, p] = -values[..., k]
    return out


def covariance_to_channels(cov, keys):
    return np.stack([cov[..., p, q] for p, q in keys], axis=-1)
