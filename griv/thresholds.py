"""Threshold rule (Section 5.4).

tau_m = np.nanquantile(S_m[VAL_THR & benign & covered], 0.99, method="linear"); alert iff S_m > tau_m (strict).
"""
from __future__ import annotations

import numpy as np

from .errors import ValidityError

METHOD = "linear"


def tau_population(node_label: np.ndarray, covered: np.ndarray) -> np.ndarray:
    return (np.asarray(node_label).astype(int) == 0) & np.asarray(covered).astype(bool)


def tau(score: np.ndarray, node_label: np.ndarray, covered: np.ndarray, q: float = 0.99) -> dict:
    """Returns tau, the population size and the alert rate achieved on that population (S > tau)."""
    m = tau_population(node_label, covered)
    x = np.asarray(score, np.float64)[m]
    if not np.isfinite(x).any():
        raise ValidityError(f"tau population (VAL_THR benign-covered) is empty or non-finite (n={int(m.sum())})")
    t = float(np.nanquantile(x, q, method=METHOD))
    fin = np.isfinite(x); n = int(m.sum())
    return {"tau": t, "n_val_thr_benign_covered": n, "n_nonfinite_score": int((~fin).sum()),
            "val_thr_achieved_alert_rate": float(np.sum(x[fin] > t) / n) if n else float("nan"),
            "population": "VAL_THR & node_label==0 & covered",
            "rate_definition": "score exceedance S > tau (GRIV: p_G > tau_G, before NodeReportIDS gating)"}


def alert(score: np.ndarray, t: float, covered: np.ndarray) -> np.ndarray:
    s = np.asarray(score, np.float64)
    with np.errstate(invalid="ignore"):
        return np.asarray(covered).astype(bool) & np.nan_to_num(s > t, nan=False).astype(bool) & np.isfinite(s)
