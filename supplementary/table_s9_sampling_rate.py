"""Supplementary Table S9 (Section 6.2): validation check of the reference sampling rate and detailed cost measurements.

(1) Validation benefit-cost check (TEST is not used). For every CW run and sampling rate s, the CW models are rebuilt
    with the same code and seed; on VAL_THR, complete concealment (alpha = 1) is applied to every attack-bearing report
    and GRIV's Operational TPR is the benefit B(s) (micro, and attack-type macro). Costs: C_flow(s) = fraction of the
    VAL_THR reporter flows retained by the sampler; C_time(s) = median time of the whole GRIV online path (cost
    experiment). Per run, J(s) = minmax(B) - minmax(C) over the s grid; reported: the mean J per s, the s with the highest
    mean J, and how often each s is the per-run maximum. The reference rate s = 0.3 was fixed before the final evaluation;
    this check is descriptive.
(2) Cost table: every stage of the GRIV path per s (median and p95 time, per-report latency, throughput, peak RSS of the
    whole-TEST replay, retained flows).
Usage: python supplementary/table_s9_sampling_rate.py results/<run_id>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import common as CM  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from griv.sampling import keep_uniform  # noqa: E402
from griv import config as C  # noqa: E402
from griv import data as D  # noqa: E402
from griv import methods as ME  # noqa: E402
from griv import models as M  # noqa: E402
from griv import splits as S  # noqa: E402
from griv.attacks import concealment as CON  # noqa: E402
from griv.features import FEATS  # noqa: E402


def validation_benefit(run, cfg_path) -> pd.DataFrame:
    cfg = C.load(cfg_path); d = D.load(cfg)
    cw = S.make(d, cfg, "CW")
    vw = set(map(int, cw.windows["VAL_THR"])); vr = set(map(str, cw.reporters["VAL_THR"]))
    fs = d.flow_store
    sub = fs[fs["window"].isin(vw) & (fs["src_ip"].astype(str).isin(vr) | fs["dst_ip"].astype(str).isin(vr))]
    rows = []
    for seed in C.cw_seeds(cfg):
        t0 = time.time()
        nb = M.build_node(d, cfg, cw, seed, classifier="rf", cw=cw)
        vt = nb.H["VAL_THR"].reset_index(drop=True)
        X = vt[FEATS].to_numpy(np.float64); atk = vt["node_label"].to_numpy() == 1
        devs = vt["device_id"].astype(str).to_numpy()
        Q = X.copy(); Q[atk] = CON.conceal(X[atk], CON.mu_matrix(devs[atk], nb.mu_table, nb.mu_global), 1.0, FEATS)
        typ = d.window_type(vt["window"].to_numpy())
        u = keep_uniform(sub["raw_flow_id"].to_numpy(), seed, cfg["gateway"]["stream"])
        for s in cfg["s_grid"]:
            gb = M.attach_gateway(d, cfg, nb, s)
            det = ME.Detector(cfg, nb, gb)
            O = det.evaluate(vt, X=Q)["O_GRIV"]
            stored = Path(run) / "tables" / "CW" / "-" / f"seed={seed}" / f"s={s}" / "base" / "rf" / "thresholds.parquet"
            tau_diff = np.nan
            if stored.exists():
                th = pd.read_parquet(stored)
                tau_diff = abs(float(th.loc[th["method"] == "GRIV", "tau"].iloc[0]) - det.tau[("GRIV", det.q_main)])
            rows.append(dict(seed=seed, s=s, B_micro=float(O[atk].mean()),
                             B_type_macro=float(pd.Series(O[atk]).groupby(typ[atk]).mean().mean()),
                             C_flow=float((u < s).mean()) if s < 1 else 1.0, n_attack_reports=int(atk.sum()),
                             tau_GRIV=det.tau[("GRIV", det.q_main)], tau_abs_diff_vs_run=tau_diff))
        print(f"seed {seed}: {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def J(R, b, c):
    out = []
    for sd, g in R.groupby("seed"):
        g = g.sort_values("s")
        nb_ = (g[b] - g[b].min()) / (g[b].max() - g[b].min())
        nc = (g[c] - g[c].min()) / (g[c].max() - g[c].min())
        out.append(pd.DataFrame(dict(seed=sd, s=g["s"], J=nb_ - nc)))
    return pd.concat(out)


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, "supplementary"); rc = CM.run_config(run)
    cost = CM.cost_table(run)
    R = validation_benefit(run, rc["_path"])
    tm = cost[cost["stage"] == "E_griv_online_path"].set_index("s")["median_ns"] if len(cost) else pd.Series(dtype=float)
    R["C_time"] = R["s"].map(tm)
    rows, summ = [], []
    for b in ("B_micro", "B_type_macro"):
        for c in ("C_flow", "C_time"):
            if R[c].isna().all():
                continue
            j = J(R, b, c); mj = j.groupby("s")["J"].mean()
            arg = j.loc[j.groupby("seed")["J"].idxmax()]["s"].value_counts().sort_index()
            for s, v in mj.items():
                rows.append(dict(benefit=b, cost=c, s=s, mean_J=v, n_runs_maximal=int(arg.get(s, 0))))
            summ.append(dict(benefit=b, cost=c, s_with_highest_mean_J=float(mj.idxmax())))
    Jt = pd.DataFrame(rows); St = pd.DataFrame(summ)
    ct = cost.copy()
    if len(ct):
        ct["peak_rss_MiB"] = ct["stageE_subprocess_peak_rss_bytes"] / 2 ** 20
        ct["flow_retention_pct"] = 100 * ct["n_flows_kept"] / ct["n_flows_test"]
        ct = ct[["s", "stage", "seed", "warmup", "repeats", "n_reports", "n_flows_test", "n_flows_kept", "flow_retention_pct", "median_ns",
                 "p95_ns", "per_report_median_us", "throughput_reports_per_s", "peak_rss_MiB", "python", "platform", "processor",
                 "cpu_count", "numpy", "pandas", "sklearn", "omp_num_threads", "mkl_num_threads"]].sort_values(["s", "stage"])
    outs = [CM.write_csv(R, out, "table_s9_validation_benefit_cost_by_run"), CM.write_csv(Jt, out, "table_s9_validation_J"),
            CM.write_csv(St, out, "table_s9_validation_J_argmax"), CM.write_csv(ct, out, "cost_detail")]
    CM.write_provenance(run, out, "supplementary/table_s9_sampling_rate.py",
                        ["models rebuilt from the data (CW, every run and s; VAL_THR only)", "shards cost CW (cost.parquet)"], outs,
                        {"tau_abs_diff_vs_run_max": float(np.nanmax(R["tau_abs_diff_vs_run"])) if R["tau_abs_diff_vs_run"].notna().any() else None})
    print(R.groupby("s")[["B_micro", "B_type_macro", "C_flow", "C_time"]].mean().round(4).to_string())
    print(St.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
