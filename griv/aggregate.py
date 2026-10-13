"""Aggregation (Section 5.6).

CW: per (seed, beta, attack type) cell -> mean over applicable beta within each attack type -> equal-weight mean over
the seven attack types -> mean over runs (with between-run SD). Micro (report-weighted) is available as well.
RD: attack metrics pool the reports of the attack-bearing folds within each attack type, then macro-average over types
and average over runs; benign metrics pool all folds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as MT

ATTACK_METRICS = ["operational_TPR", "covered_TPR", "simultaneous_evasion", "tampered_report_coverage",
                  "tampered_report_attribution_TPR"]


def load_shards(root: Path, experiment=None, split=None, attack=None, param=None, variant=None, classifier=None,
                s=None) -> pd.DataFrame:
    root = Path(root)
    files = sorted((root / "shards").rglob("*.parquet"))
    out = []
    for p in files:
        parts = p.relative_to(root / "shards").parts      # experiment / split / fold / seed= / s= / variant / file
        E, sp, fold, seed, ss, var, fn = parts
        if fn.count("__") != 2:                 # side tables (cost.parquet) are not report shards
            continue
        clf, atk, prm = fn[:-8].split("__")
        if experiment and E != experiment:
            continue
        if split and sp != split:
            continue
        if attack and atk != attack:
            continue
        if param and prm != param:
            continue
        if variant and var != variant:
            continue
        if classifier and clf != classifier:
            continue
        if s is not None and float(ss.split("=")[1]) != float(s):
            continue
        if not p.with_suffix(".COMPLETE.json").exists():
            continue
        out.append(pd.read_parquet(p))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def cells(f: pd.DataFrame, betas, adaptive: bool, pooled_folds: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per seed: attack cells (beta, type, row, stratum) and benign metrics (beta, row). pooled_folds=True (RD) pools folds."""
    A, U = [], []
    for seed, g in f.groupby("seed"):
        for b in betas:
            if pooled_folds:
                # RD: the compromised set of each fold is its held-out reporter; targets are computed per fold, then pooled
                tb_list = []
                for fold, gf in g.groupby("fold"):
                    tb_list.append(gf.assign(_tb=MT.targeted_at(gf, b)))
                gg = pd.concat(tb_list)
                a = _attack_metrics_pre_tb(gg, b, adaptive)
                u = _untampered_pre_tb(gg, b, adaptive)
            else:
                a = MT.attack_metrics(g, b, adaptive)
                u = MT.untampered_metrics(g, b, adaptive)
            A.append(a.assign(seed=seed)); U.append(u.assign(seed=seed))
    return (pd.concat(A, ignore_index=True) if A else pd.DataFrame(),
            pd.concat(U, ignore_index=True) if U else pd.DataFrame())


def _attack_metrics_pre_tb(gg, b, adaptive):
    """attack_metrics on pooled RD folds, using the per-fold targets in _tb."""
    g2 = gg.copy()
    # rank 0 for targeted rows, -1 otherwise, with N = 1, so that targeted_at(., 1.0) reproduces _tb
    g2["compromise_rank"] = np.where(g2["_tb"].to_numpy(), 0, -1); g2["n_eligible"] = 1
    return MT.attack_metrics(g2, 1.0, adaptive).assign(beta=b)


def _untampered_pre_tb(gg, b, adaptive):
    g2 = gg.copy()
    g2["compromise_rank"] = np.where(g2["_tb"].to_numpy(), 0, -1); g2["n_eligible"] = 1
    nw = int(g2["n_test_windows"].iloc[0]) if "n_test_windows" in g2.columns else int(g2["window"].nunique())
    return MT.untampered_metrics(g2, 1.0, adaptive, n_test_windows=nw).assign(beta=b)


class CellCountError(RuntimeError):
    """Targeted-report counts in the shards differ from the pre-attack counts in sna.parquet (missing or duplicated rows)."""


def na_cells(A: pd.DataFrame) -> pd.DataFrame:
    """Cells in state SNA / ONA / ZERO_EFFECT (stratum all): their metrics are NA and excluded from macros."""
    if A.empty or "state" not in A.columns:
        return A.iloc[0:0]
    x = A[(A["stratum"] == "all") & (A["state"] != "OK")]
    keep = [c for c in ("seed", "fold", "beta", "attack_type", "row", "state", "n_targeted", "n_applicable", "n_modified",
                        "mode") if c in x.columns]
    return x[keep]


def check_against_sna(A: pd.DataFrame, sna: pd.DataFrame) -> pd.DataFrame:
    """Cell n_targeted (shards) vs pre-attack sna.parquet; returns mismatching cells (empty = pass). RD compares fold sums."""
    if A.empty or sna.empty:
        return pd.DataFrame()
    a = (A[A["stratum"] == "all"].drop_duplicates(subset=["seed", "beta", "attack_type"])
         [["seed", "beta", "attack_type", "n_targeted"]])
    b = sna.groupby(["seed", "beta", "attack_type"], as_index=False)["n_targeted"].sum().rename(columns={"n_targeted": "n_sna"})
    m = a.merge(b, on=["seed", "beta", "attack_type"], how="left")
    m["n_sna"] = m["n_sna"].fillna(0).astype(int)
    return m[m["n_targeted"] != m["n_sna"]]


def macro(A: pd.DataFrame, metric: str, betas=None) -> pd.DataFrame:
    """beta macro -> 7-type macro -> mean over runs. Returns row, stratum, value, seed_sd, n_seeds, n_types, n_cells_sum."""
    x = A[(A["n"] > 0) & (A["state"] == "OK" if "state" in A.columns else True)].copy()
    if betas is not None:
        x = x[x["beta"].isin(betas)]
    if metric not in A.columns:
        return pd.DataFrame()
    if x.empty:
        return finalize_n(_keep_all_strata(A, pd.DataFrame(columns=["row", "stratum", "value", "seed_sd", "n_seeds", "n_types", "n"]),
                                           metric), x)
    x = x[np.isfinite(x[metric])]
    if "n_targeted" not in x.columns:
        x["n_targeted"] = x["n"]
    # n = denominator of the metric (attribution TPR -> n_attr, covered TPR -> n_covered)
    if metric == "tampered_report_attribution_TPR" and "n_attr" in x.columns:
        x["n"] = x["n_attr"].fillna(0)
    elif metric == "covered_TPR" and "n_covered" in x.columns:
        x["n"] = x["n_covered"]
    t = x.groupby(["seed", "row", "stratum", "attack_type"]).agg(v=(metric, "mean"), n=("n", "sum"), nt=("n_targeted", "sum"),
                                                                  n_beta=("beta", "nunique")).reset_index()
    s = t.groupby(["seed", "row", "stratum"]).agg(v=("v", "mean"), n=("n", "sum"), nt=("nt", "sum"),
                                                  n_types=("attack_type", "nunique")).reset_index()
    r = s.groupby(["row", "stratum"]).agg(value=("v", "mean"), seed_sd=("v", "std"), n_seeds=("seed", "nunique"),
                                          n_types=("n_types", "mean"), n=("n", "sum"), n_targeted=("nt", "sum")).reset_index()
    r["metric"] = metric
    return finalize_n(_keep_all_strata(A, r, metric), x)


def per_type(A: pd.DataFrame, metric: str, betas=None) -> pd.DataFrame:
    """Per attack type: mean over applicable beta -> mean over runs, per row, stratum and type."""
    x = A[(A["n"] > 0) & (A["state"] == "OK" if "state" in A.columns else True)].copy()
    if betas is not None:
        x = x[x["beta"].isin(betas)]
    if metric not in A.columns or x.empty:
        return pd.DataFrame()
    x = x[np.isfinite(x[metric])]
    if metric == "tampered_report_attribution_TPR" and "n_attr" in x.columns:
        x["n"] = x["n_attr"].fillna(0)
    elif metric == "covered_TPR" and "n_covered" in x.columns:
        x["n"] = x["n_covered"]
    t = x.groupby(["seed", "row", "stratum", "attack_type"]).agg(v=(metric, "mean"), n=("n", "sum")).reset_index()
    r = t.groupby(["row", "stratum", "attack_type"]).agg(value=("v", "mean"), seed_sd=("v", "std"), n_seeds=("seed", "nunique"),
                                                         n=("n", "sum")).reset_index()
    r["metric"] = metric; r["level"] = "type:" + r["attack_type"]; r["n_types"] = 1
    return finalize_n(r, x, keys=("row", "stratum", "attack_type"))


def _beta1_per_seed(x: pd.DataFrame, keys) -> pd.DataFrame:
    """Mean over runs of the beta = 1 denominator (summed over types)."""
    b1 = x[np.isclose(x["beta"].astype(float), 1.0)]
    if b1.empty:
        return pd.DataFrame(columns=list(keys) + ["n_reports_beta1_per_seed"])
    s = b1.groupby(["seed"] + list(keys))["n"].sum().reset_index()
    return s.groupby(list(keys))["n"].mean().rename("n_reports_beta1_per_seed").reset_index()


def finalize_n(r: pd.DataFrame, x: pd.DataFrame, keys=("row", "stratum")) -> pd.DataFrame:
    """Rename n -> n_cells_sum and add n_reports_beta1_per_seed."""
    r = r.rename(columns={"n": "n_cells_sum"})
    b = _beta1_per_seed(x, keys)
    r = r.merge(b, on=list(keys), how="left")
    r["n_reports_beta1_per_seed"] = r["n_reports_beta1_per_seed"].fillna(0.0)
    return r


def _keep_all_strata(A, r, metric):
    """Keep (row, stratum) pairs with n = 0 (value NaN) so that all strata always appear."""
    allc = A[["row", "stratum"]].drop_duplicates()
    r = allc.merge(r, on=["row", "stratum"], how="left")
    r["n"] = r["n"].fillna(0).astype(int); r["metric"] = metric
    # the type macro averages applicable types only; the number of types is reported as "k-type macro"
    if "n_types" in r.columns:
        r["n_types"] = r["n_types"].fillna(0)
        r["level"] = r["n_types"].map(lambda k: f"{int(round(k))}-type macro" if k == k and k > 0 else "no applicable type")
    else:
        r["level"] = "micro"
    return r


def micro(A: pd.DataFrame, metric: str, betas=None) -> pd.DataFrame:
    """Report-weighted ratio over beta and type cells -> mean over runs."""
    x = A[(A["n"] > 0) & (A["state"] == "OK" if "state" in A.columns else True)].copy()
    if betas is not None:
        x = x[x["beta"].isin(betas)]
    if metric not in A.columns:
        return pd.DataFrame()
    if x.empty:
        return finalize_n(_keep_all_strata(A, pd.DataFrame(columns=["row", "stratum", "value", "seed_sd", "n_seeds", "n"]), metric), x)
    nn = ("n_covered" if metric == "covered_TPR" else
          "n_attr" if (metric == "tampered_report_attribution_TPR" and "n_attr" in x.columns) else "n")
    x = x[np.isfinite(x[metric])]
    x["num"] = x[metric] * x[nn]
    s = x.groupby(["seed", "row", "stratum"]).agg(num=("num", "sum"), den=(nn, "sum"),
                                                  n_types=("attack_type", "nunique")).reset_index()
    s["v"] = s["num"] / s["den"]
    r = s.groupby(["row", "stratum"]).agg(value=("v", "mean"), seed_sd=("v", "std"), n_seeds=("seed", "nunique"),
                                          n=("den", "sum"), n_types=("n_types", "max")).reset_index()
    r = _keep_all_strata(A, r, metric)
    r["level"] = "micro"
    x2 = x.copy(); x2["n"] = x2[nn]
    return finalize_n(r, x2)


def untampered_summary(U: pd.DataFrame, beta=1.0) -> pd.DataFrame:
    """Benign metrics do not depend on beta (benign reports are never modified): mean over runs at the given beta."""
    x = U[U["beta"] == beta]
    num = [c for c in x.columns if c not in ("beta", "row", "seed") and pd.api.types.is_numeric_dtype(x[c])]
    g = x.groupby("row")[num].mean().reset_index()
    sd = x.groupby("row")[num].std().add_suffix("_seed_sd").reset_index(drop=True)
    return pd.concat([g, sd], axis=1).assign(beta=beta)


def summarize(A: pd.DataFrame, U: pd.DataFrame, betas) -> pd.DataFrame:
    """Summary per metric x row x stratum: beta macro, beta = 1 only, and micro."""
    rows = []
    for m in ATTACK_METRICS:
        for agg, df in (("beta_macro", macro(A, m, betas)), ("beta1_only", macro(A, m, [1.0])),
                        ("micro", micro(A, m, betas))):
            if len(df):
                rows.append(df.assign(aggregation=agg))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
