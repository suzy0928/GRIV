"""Adaptive concealment (Sections 3.3 and 5.3).

Attacker knowledge: the method's decision rule and threshold, the classifier and calibration models, its own benign
reference, and (evaluation oracle) gateway availability, the realized gateway probability and, for Feature Residual,
the gateway feature vector. Gateway observations cannot be modified.
Search: alpha in {0, step, ..., 1} on the concealment path of Eq. (1), separately for each targeted report and method.
  - the smallest alpha with NodeReportIDS(submitted) == benign and no integrity alert (simultaneous evasion; alpha = 0 means the
    report already evades without modification);
  - otherwise the smallest alpha that evades the node-report IDS alone;
  - otherwise alpha = 1.
All methods use the same grid and projection. Deterministic.
"""
from __future__ import annotations

import numpy as np

from ..methods import METHODS
from .concealment import conceal

EXTRA_ROWS = ["NodeReportIDS", "NodeOrGatewayIDS"]      # reference rows, each searched as its own target


def alpha_grid(step: float) -> np.ndarray:
    n = int(round(1.0 / float(step)))
    g = np.round(np.arange(n + 1) * float(step), 10)
    assert g[0] == 0.0 and abs(g[-1] - 1.0) < 1e-12
    g[-1] = 1.0
    return g


def search(det, X, MU, devices, covered, Gx, pG, feats, step):
    """For the targeted reports X (n x 18): selected alpha and the resulting submitted report, per method."""
    grid = alpha_grid(step)
    n, K = len(X), len(grid)
    if n == 0:
        return {m: dict(alpha=np.zeros(0), X=np.zeros((0, len(feats))), pred=np.zeros(0, int), p_N=np.zeros(0),
                        score=np.zeros(0), I=np.zeros(0, bool), case=np.zeros(0, dtype=object))
                for m in list(METHODS) + EXTRA_ROWS}, grid
    Q = np.concatenate([conceal(X, MU, a, feats) for a in grid], axis=0)          # (K n) x 18, blocks ordered by alpha
    dev_r = np.tile(np.asarray(devices), K); cov_r = np.tile(covered, K)
    Gx_r = np.tile(Gx, (K, 1)); pG_r = np.tile(pG, K)
    nb = det.nb
    raw = nb.raw_p(Q); pred = nb.predict(Q); pN = nb.cal_N.transform(raw)
    _, _, S = det.scores(Q, dev_r, cov_r, Gx_r, pN=pN, pG=pG_r)
    I = dict(det.alerts(S, cov_r, pred))
    AG = det.gateway_alert(pG_r, cov_r)
    I["NodeReportIDS"] = np.zeros(len(pred), bool)                 # node-report IDS alone: evading the node is enough
    I["NodeOrGatewayIDS"] = AG                                   # node-report IDS | gateway IDS
    S = dict(S)
    S["NodeReportIDS"] = raw; S["NodeOrGatewayIDS"] = pG_r
    pred_k = pred.reshape(K, n); res = {}
    for m in list(METHODS) + EXTRA_ROWS:
        Ik = I[m].reshape(K, n)
        evade = (pred_k == 0) & ~Ik
        node_ok = pred_k == 0
        first_e = np.where(evade.any(axis=0), evade.argmax(axis=0), -1)
        first_n = np.where(node_ok.any(axis=0), node_ok.argmax(axis=0), -1)
        k_sel = np.where(first_e >= 0, first_e, np.where(first_n >= 0, first_n, K - 1))
        case = np.where(first_e >= 0, "evade", np.where(first_n >= 0, "node_only", "none"))
        flat = k_sel * n + np.arange(n)
        res[m] = dict(alpha=grid[k_sel], k=k_sel, X=Q[flat], pred=pred[flat], p_N=pN[flat], raw_pN=raw[flat],
                      score=S[m][flat], I=I[m][flat], case=case)
    return res, grid
