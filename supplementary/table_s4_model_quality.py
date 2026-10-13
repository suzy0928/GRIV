"""Supplementary Table S4 and Fig. S1 (Section 5.4): descriptive quality of the shared classifier in the two views.

Honest CW TEST reports at s = 0.3, 20 runs (complete-concealment shards; only honest columns are used):
  - node view: calibrated p_N of every TEST report; gateway view: calibrated p_G of covered reports;
  - AUROC, Brier score and ECE (10 equal-width probability bins) per run, then the mean over runs;
  - reliability curves (Fig. S1): per bin, mean predicted probability and observed attack fraction, pooled over runs;
  - feature-wise two-sample Kolmogorov-Smirnov distance between node reports and gateway observations of covered
    TEST reports (per run, then the mean over runs). Gateway observations are rebuilt from the data with the run's
    sampling hash; no model is trained.
Usage: python supplementary/table_s4_model_quality.py results/<run_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import common as CM  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from griv import config as C  # noqa: E402
from griv import data as D  # noqa: E402
from griv import models as M  # noqa: E402
from griv import splits as S  # noqa: E402
from griv.features import FEATS  # noqa: E402

BINS = 10


def _bin(p):
    e = np.linspace(0, 1, BINS + 1)
    return e, np.clip(np.digitize(p, e[1:-1], right=True), 0, BINS - 1)


def ece(y, p) -> float:
    _, idx = _bin(p)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(BINS) if (idx == b).any()))


def quality(y, p) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score
    m = np.isfinite(p); y, p = y[m].astype(int), p[m].astype(float)
    return dict(n=int(len(y)), AUROC=float(roc_auc_score(y, p)), Brier=float(brier_score_loss(y, p)), ECE=ece(y, p))


def reliability(y, p) -> pd.DataFrame:
    m = np.isfinite(p); y, p = y[m].astype(int), p[m].astype(float)
    e, idx = _bin(p)
    return pd.DataFrame([dict(bin=b, lo=e[b], hi=e[b + 1], n=int((idx == b).sum()), sum_pred=float(p[idx == b].sum()),
                              n_pos=int(y[idx == b].sum())) for b in range(BINS)])


def calibration(run) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = CM.load_setting(run, "complete_concealment", "CW", CM.CONCEALMENT, "alpha=1.0")
    q, rel = [], []
    for seed, g in res.f.groupby("seed"):
        y = g["node_label"].to_numpy().astype(int); cov = g["covered_honest"].to_numpy().astype(bool)
        for view, p, m in (("node", g["p_N_honest"].to_numpy(float), np.ones(len(g), bool)),
                           ("gateway", g["p_G"].to_numpy(float), cov)):
            q.append(dict(seed=seed, view=view, **quality(y[m], p[m])))
            rel.append(reliability(y[m], p[m]).assign(seed=seed, view=view))
    return pd.DataFrame(q), pd.concat(rel, ignore_index=True)


def feature_shift(run) -> pd.DataFrame:
    from scipy.stats import ks_2samp
    rc = CM.run_config(run); cfg = C.load(rc["_path"]); d = D.load(cfg)
    cw = S.make(d, cfg, "CW")
    s = cfg["gateway"]["s_main"]
    rows = []
    for seed in C.cw_seeds(cfg):
        te = M.honest_reports(d, cw.windows["TEST"], cw.reporters["TEST"])
        go = M.gateway_obs(d.flow_store, cw.reporters["TEST"], cw.windows["TEST"], s, seed, stream=cfg["gateway"]["stream"])
        pair = te[M.KEYS + FEATS].merge(go[M.KEYS + FEATS], on=M.KEYS, suffixes=("_node", "_gw"))   # covered reports
        for f in FEATS:
            r = ks_2samp(pair[f"{f}_node"].to_numpy(float), pair[f"{f}_gw"].to_numpy(float))
            rows.append(dict(seed=seed, feature=f, KS=float(r.statistic), n_covered=int(len(pair))))
        print(f"KS seed {seed}: {len(pair)} covered reports", flush=True)
    return pd.DataFrame(rows)


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, "supplementary")
    Q, rel = calibration(run)
    T = Q.groupby("view")[["AUROC", "Brier", "ECE", "n"]].mean().reindex(["node", "gateway"]).reset_index()
    T = T.merge(Q.groupby("view")[["AUROC", "Brier", "ECE"]].std().add_suffix("_run_sd").reset_index(), on="view")
    R = rel.groupby(["view", "bin", "lo", "hi"], as_index=False)[["n", "sum_pred", "n_pos"]].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        R["mean_pred"] = R["sum_pred"] / R["n"]; R["frac_pos"] = R["n_pos"] / R["n"]
    K = feature_shift(run)
    KT = K.groupby("feature", sort=False)[["KS", "n_covered"]].mean().reset_index()
    outs = [CM.write_csv(T, out, "table_s4a_model_quality"), CM.write_csv(Q, out, "table_s4a_model_quality_by_run"),
            CM.write_csv(R.drop(columns=["sum_pred"]), out, "fig_s1_reliability"),
            CM.write_csv(KT, out, "table_s4b_feature_ks"), CM.write_csv(K, out, "table_s4b_feature_ks_by_run")]
    CM.write_provenance(run, out, "supplementary/table_s4_model_quality.py",
                        ["shards complete_concealment CW alpha=1.0 (honest columns)",
                         "honest TEST reports and gateway observations rebuilt from the data (CW, every run, s_main)"], outs,
                        {"ece_bins": BINS, "KS_range": [float(KT["KS"].min()), float(KT["KS"].max())]})
    print(T.round(4).to_string(index=False))
    print(f"KS range over features: {KT['KS'].min():.4f} - {KT['KS'].max():.4f}")


if __name__ == "__main__":
    main(sys.argv)
