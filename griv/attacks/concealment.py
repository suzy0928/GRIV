"""Prototype-directed concealment (Eq. (1), Section 3.3).

q = Pi((1 - alpha) x + alpha mu_ben(reporter)), where Pi is the integer and relation projection
(constraints.project): directional counts and the seven additive sums per direction become non-negative integers
(round half up), mean durations stay non-negative, and a zero count zeroes that direction's sums and duration.
alpha = 0 leaves the report unchanged.
"""
from __future__ import annotations

import numpy as np

from .constraints import project


def conceal(X: np.ndarray, MU: np.ndarray, alpha: float, feats: list) -> np.ndarray:
    a = float(alpha)
    if a == 0.0:
        return np.array(X, dtype=np.float64, copy=True)
    q = (1.0 - a) * np.asarray(X, np.float64) + a * np.asarray(MU, np.float64)
    return project(q, feats)


def mu_matrix(devices, mu_table: dict, mu_global: np.ndarray) -> np.ndarray:
    """Benign reference vector of each report's reporter (pooled median when the reporter has none)."""
    return np.vstack([mu_table.get(str(d), mu_global) for d in devices]) if len(devices) else \
        np.zeros((0, len(mu_global)))
