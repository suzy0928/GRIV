"""Feature Residual statistics (Section 5.5).

Residual e = r - o between an honest node report r and its sampled gateway observation o, on covered TRAIN pairs.
Per reporter and feature: mean and sample standard deviation (reporters with at least ``min_reports`` covered pairs);
other reporters use the pooled values. Standard deviations below ``floor`` are replaced by the pooled value and then
floored at ``floor``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KEYS = ["device_id", "window"]


@dataclass
class FeatureResidualStats:
    mu: dict
    sigma: dict
    global_mu: np.ndarray
    global_sigma: np.ndarray
    eps: float
    fallback_devices: list = field(default_factory=list)

    def get_mu(self, device: str) -> np.ndarray:
        return self.mu.get(str(device), self.global_mu)

    def get_sigma(self, device: str) -> np.ndarray:
        return self.sigma.get(str(device), self.global_sigma)


def feature_residual_stats(reports: pd.DataFrame, gateway: pd.DataFrame, feats: list, floor: float,
                           min_reports: int) -> FeatureResidualStats:
    o = gateway[KEYS + feats].rename(columns={c: f"__o_{c}" for c in feats})
    m = reports[KEYS + feats].merge(o, on=KEYS, how="left", validate="many_to_one")
    ocols = [f"__o_{c}" for c in feats]
    m = m[~m[ocols].isna().any(axis=1)]
    if m.empty:
        raise ValueError("Feature Residual statistics: no covered TRAIN pairs")
    res = m[KEYS].copy()
    res[feats] = m[feats].to_numpy(np.float64) - m[ocols].to_numpy(np.float64)
    E = res[feats].to_numpy(np.float64)
    gmu = E.mean(axis=0)
    gsd = E.std(axis=0, ddof=1)
    gsd = np.where(~np.isfinite(gsd), floor, gsd)
    gsd = np.where(gsd < floor, floor, gsd)
    mu, sigma, fallback = {}, {}, []
    for dev, grp in res.groupby("device_id"):
        X = grp[feats].to_numpy(np.float64)
        if len(X) >= min_reports:
            sd = X.std(axis=0, ddof=1)
            sd = np.where(~np.isfinite(sd), gsd, sd)
            sd = np.where(sd < floor, gsd, sd)
            sd = np.where(sd < floor, floor, sd)
            mu[str(dev)] = X.mean(axis=0)
            sigma[str(dev)] = sd
        else:
            fallback.append(str(dev))
    return FeatureResidualStats(mu=mu, sigma=sigma, global_mu=gmu, global_sigma=gsd, eps=floor,
                                fallback_devices=sorted(fallback))
