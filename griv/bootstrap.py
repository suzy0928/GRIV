"""Cluster bootstrap for CW attack metrics (Section 5.6).

Clusters = (reporter, 60-minute clock block), block = window // 60. Strata = attack types: a cluster with attack reports
of several types is assigned to the type with the most attack windows (ties: alphabetical). Clusters are resampled with
replacement within each stratum, and the drawn weight applies to all reports of that cluster. Each replicate recomputes
cell rate -> mean over applicable beta -> 7-type macro -> mean over runs. All methods share the same draws (paired).
Runs are not resampled. B = 10,000 percentile intervals, rng = np.random.default_rng(20261002).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as MT
from .config import TYPES
from .errors import ValidityError


def _cluster_ids(f: pd.DataFrame, cluster: str) -> np.ndarray:
    if cluster == "reporter_block60":
        return (f["device_id"].astype(str) + "|" + (f["window"].astype(np.int64) // 60).astype(str)).to_numpy()
    if cluster == "reporter":
        return f["device_id"].astype(str).to_numpy()
    raise ValueError(cluster)


def cluster_strata(f: pd.DataFrame, cluster: str) -> tuple[dict, int]:
    """Cluster -> assigned stratum (type with most attack windows, ties alphabetical) and the number of multi-type clusters."""
    a = f[(f["node_label"] == 1) & f["attack_type"].isin(TYPES)].copy()
    a["_c"] = _cluster_ids(a, cluster)
    cnt = a.drop_duplicates(subset=["_c", "attack_type", "window"]).groupby(["_c", "attack_type"]).size()
    assign, overlap = {}, 0
    for c, g in cnt.groupby(level=0):
        g = g.droplevel(0)
        if len(g) > 1:
            overlap += 1
        mx = g.max()
        assign[c] = sorted(t for t, v in g.items() if v == mx)[0]
    return assign, overlap


def check_seeds(f: pd.DataFrame, expected_seeds, what: str):
    """The runs present in the shards must equal the configured runs."""
    have = sorted(int(x) for x in f["seed"].unique()); want = sorted(int(x) for x in expected_seeds)
    if have != want:
        raise ValidityError(f"{what}: seed set {have} != configured {want}")
    return len(have)


def prepare(f: pd.DataFrame, betas, value_cols: dict, cluster: str, adaptive=True, require_all_types=True,
            expected_seeds=None):
    """value_cols: {"GRIV": "<0/1 column>", ...} per targeted report. Returns clusters, strata and per-type count arrays."""
    f = f.reset_index(drop=True)
    if expected_seeds is not None:
        check_seeds(f, expected_seeds, "bootstrap input")
    seeds = sorted(f["seed"].unique())
    N_of = {sd: MT.n_eligible(f[f["seed"] == sd]) for sd in seeds}
    assign, overlap = cluster_strata(f, cluster)
    clusters = sorted(assign)
    ci = {c: i for i, c in enumerate(clusters)}
    C, S, Bn = len(clusters), len(seeds), len(betas)
    cid_all = _cluster_ids(f, cluster)
    present = set()
    for sd in seeds:
        g = f[f["seed"] == sd]
        tb1 = MT.targeted_at(g, 1.0, N_of[sd])
        present |= set(g.loc[tb1 & g["attack_type"].isin(TYPES).to_numpy(), "attack_type"])
    present = sorted(present)
    if require_all_types and present != sorted(TYPES):
        raise ValidityError(f"bootstrap input lacks attack types: {sorted(set(TYPES) - set(present))}")
    per_type = {}
    for t in TYPES:
        ft = f[(f["attack_type"] == t) & (f["node_label"] == 1)]
        if not len(ft):
            continue
        idx = np.array([ci[c] for c in cid_all[ft.index.to_numpy()]], dtype=int)
        N = np.zeros((S, C, Bn)); V = {m: np.zeros((S, C, Bn)) for m in value_cols}
        for si, sd in enumerate(seeds):
            sel = (ft["seed"] == sd).to_numpy()
            g = ft[sel]; gi = idx[sel]
            for bi, b in enumerate(betas):
                tb = MT.targeted_at(g, b, N_of[sd])
                if not adaptive:
                    tb = tb & g["was_modified"].to_numpy().astype(bool) & g["attack_applicable"].to_numpy().astype(bool)
                elif "attack_applicable" in g.columns:
                    tb = tb & g["attack_applicable"].to_numpy().astype(bool)
                N[si, :, bi] = np.bincount(gi[tb], minlength=C)
                for m, col in value_cols.items():
                    V[m][si, :, bi] = np.bincount(gi[tb], weights=g[col].to_numpy().astype(float)[tb], minlength=C)
        per_type[t] = dict(N=N, V=V)
    strata = {t: np.array([ci[c] for c in clusters if assign[c] == t], dtype=int) for t in TYPES}
    return dict(clusters=clusters, stratum_of=assign, strata=strata, per_type=per_type, seeds=seeds,
                n_overlap_clusters=int(overlap), cluster=cluster, n_seeds=len(seeds))


def _weights(prep, B, rng):
    """Resample clusters within each stratum -> weights (B x C), applied to all reports of a cluster."""
    C = len(prep["clusters"])
    W = np.zeros((B, C))
    for t in sorted(prep["strata"]):
        ids = prep["strata"][t]
        if len(ids):
            W[:, ids] = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)), size=B)
    return W


def _stat(W, prep, m_list):
    """W: (B x C). Returns {m: (B,)} = mean over applicable beta -> 7-type macro -> mean over runs."""
    res = {m: [] for m in m_list}
    for t in TYPES:
        P = prep["per_type"].get(t)
        if P is None:
            continue
        den = np.einsum("bc,scx->bsx", W, P["N"])
        for m in m_list:
            num = np.einsum("bc,scx->bsx", W, P["V"][m])
            with np.errstate(invalid="ignore", divide="ignore"):
                cell = np.where(den > 0, num / den, np.nan)
                ok = np.isfinite(cell).any(axis=2)
                mean = np.where(ok, np.nansum(cell, axis=2) / np.maximum(np.isfinite(cell).sum(axis=2), 1), np.nan)
            res[m].append(mean)                              # B x S
    out = {}
    for m in m_list:
        arr = np.stack(res[m], axis=2)                       # B x S x types
        tm = np.where(np.isfinite(arr).all(axis=2), np.nan_to_num(arr).mean(axis=2), np.nan)   # NaN if a type is missing
        cnt = np.isfinite(tm).sum(axis=1)
        out[m] = np.where(cnt > 0, np.nansum(tm, axis=1) / np.maximum(cnt, 1), np.nan)
    return out


def level_cis(prep, methods, B=10000, rng_seed=20261002, ci=0.95) -> dict:
    """Point value and percentile CI for each method, all methods evaluated on the same bootstrap draws."""
    rng = np.random.default_rng(rng_seed)
    pt = _stat(np.ones((1, len(prep["clusters"]))), prep, list(methods))
    bs = _stat(_weights(prep, B, rng), prep, list(methods))
    out = {}
    for m in methods:
        v = bs[m]; fin = np.isfinite(v)
        lo, hi = np.percentile(v[fin], [100 * (1 - ci) / 2, 100 * (1 + ci) / 2]) if fin.any() else (np.nan, np.nan)
        out[m] = dict(point=float(pt[m][0]), ci_lo=float(lo), ci_hi=float(hi), B_requested=int(B), B_finite=int(fin.sum()),
                      rng_seed=int(rng_seed), cluster=prep["cluster"], n_clusters=int(len(prep["clusters"])))
    return out


def benign_ci(f: pd.DataFrame, col: str, cluster: str, B=10000, rng_seed=20261002, ci=0.95) -> dict:
    """Cluster-bootstrap CI of a benign-report rate (mean of `col` over unmodified benign reports, mean over runs).
    Clusters as above, without strata; every run uses the same cluster weights."""
    g = f[f["node_label"] == 0].reset_index(drop=True)
    cid = _cluster_ids(g, cluster)
    clusters = sorted(set(cid.tolist())); ci_ = {c: i for i, c in enumerate(clusters)}
    idx = np.array([ci_[c] for c in cid]); C = len(clusters)
    seeds = sorted(g["seed"].unique())
    num = np.zeros((len(seeds), C)); den = np.zeros((len(seeds), C))
    vals = g[col].to_numpy().astype(float); sd = g["seed"].to_numpy()
    for si, s in enumerate(seeds):
        m = sd == s
        num[si] = np.bincount(idx[m], weights=vals[m], minlength=C); den[si] = np.bincount(idx[m], minlength=C)
    rng = np.random.default_rng(rng_seed)
    W = rng.multinomial(C, np.full(C, 1.0 / C), size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (W @ num.T) / (W @ den.T)
    est = (num.sum(1) / den.sum(1)).mean()
    vm = v.mean(axis=1); fin = np.isfinite(vm)
    lo, hi = np.percentile(vm[fin], [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return dict(point=float(est), ci_lo=float(lo), ci_hi=float(hi), B_requested=int(B), B_finite=int(fin.sum()), n_clusters=C)
