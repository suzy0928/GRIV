"""Compromised reporters and targeted reports (Sections 3.2 and 5.3).

  - eligible pool: monitored TEST reporters with at least one attack-bearing TEST report
  - ranking: TRAIN attack-bearing report count, then TRAIN byte volume, then reporter id; fixed across runs and methods
  - compromised set at beta: the first round_half_up(beta * N) reporters of the ranking (nested in beta)
Targeted reports = attack-bearing TEST reports of compromised reporters, fixed before any method-specific search.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from .config import TYPES
from .provenance import json_hash


def n_compromised(beta: float, N: int) -> int:
    """round_half_up(beta * N) in exact decimal arithmetic (e.g. 0.7 * 45 = 31.5 -> 32), clamped to [0, N]."""
    k = int((Decimal(str(beta)) * Decimal(int(N))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return max(0, min(k, int(N)))


def eligible_reporters(o_all: pd.DataFrame, test_windows, test_reporters) -> list[str]:
    test = {int(w) for w in test_windows}
    nodes = {str(n) for n in test_reporters}
    t = o_all[o_all["window"].astype("int64").isin(test) & o_all["device_id"].astype(str).isin(nodes)]
    return sorted(t[t["node_label"].astype(int) == 1]["device_id"].astype(str).unique())


def ranking(o_all: pd.DataFrame, train_windows, eligible) -> list[str]:
    trn = {int(w) for w in train_windows}; vn = set(map(str, eligible))
    t = o_all[o_all["window"].astype("int64").isin(trn) & o_all["device_id"].astype(str).isin(vn)]
    dev = t["device_id"].astype(str)
    atk = t.assign(_d=dev).groupby("_d")["node_label"].sum()
    bcols = [c for c in o_all.columns if "bytes" in c]
    byt = t.assign(_d=dev).groupby("_d")[bcols].sum().sum(axis=1)
    return sorted(map(str, eligible), key=lambda d: (-float(atk.get(d, 0)), -float(byt.get(d, 0)), d))


class Compromise:
    def __init__(self, d, sp, betas):
        self.eligible = eligible_reporters(d.o_all, sp.windows["TEST"], sp.reporters["TEST"])
        self.ranking = ranking(d.o_all, sp.windows["TRAIN"], self.eligible) if self.eligible else []
        self.rank = {r: i for i, r in enumerate(self.ranking)}
        self.N = len(self.ranking)
        self.betas = list(betas)
        self.k = {b: n_compromised(b, self.N) for b in self.betas}

    def compromised(self, beta) -> set:
        return set(self.ranking[:n_compromised(beta, self.N)])

    def rank_of(self, devices) -> np.ndarray:
        return np.array([self.rank.get(str(x), -1) for x in devices], dtype=int)

    def targeted(self, reports: pd.DataFrame, beta) -> np.ndarray:
        """Targeted at beta: compromised reporter (rank < k(beta)) and attack-bearing report."""
        rk = self.rank_of(reports["device_id"].to_numpy())
        k = len(self.compromised(beta))
        return (rk >= 0) & (rk < k) & (reports["node_label"].astype(int).to_numpy() == 1)

    def target_hash(self, reports: pd.DataFrame, beta=1.0) -> str:
        m = self.targeted(reports, beta)
        keys = sorted(zip(reports.loc[m, "device_id"].astype(str), reports.loc[m, "window"].astype(int)))
        return json_hash(keys)[:16]

    def sna_table(self, reports: pd.DataFrame, attack_type: np.ndarray, seed: int) -> pd.DataFrame:
        """Targeted-report counts per (seed, attack type, beta) before any attack; 0 = structurally not applicable."""
        rows = []
        for b in self.betas:
            tg = self.targeted(reports, b)
            for t in TYPES:
                n = int((tg & (attack_type == t)).sum())
                rows.append(dict(seed=seed, attack_type=t, beta=b, n_targeted=n, k_compromised=self.k[b],
                                 N_eligible=self.N, structural_not_applicable=(n == 0)))
        return pd.DataFrame(rows)
