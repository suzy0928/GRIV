"""Supplementary Table S5: benign false alerts, integrity confusion counts, prevalence scenarios and nominal-exposure rates
(complete concealment alpha = 1, s = 0.3; CW 20 runs, RD 10 runs).

Benign population: all unmodified benign TEST reports, including uncovered reports (honest values). Benign reports are
never modified, so the benign metrics do not depend on the compromise ratio.
(a) CW benign-report alert burden: FPR of the Node-report IDS (intrusion alert), of the GRIV integrity alert, and of the combined
    alert (Node-report IDS + GRIV), per run then the mean over runs, with 95 % cluster-bootstrap CIs (reporter x 60-minute
    blocks, unstratified; Section S5.2); alerts per 1,000 benign reports; GRIV integrity FPR on covered benign reports.
(b) CW integrity-alert confusion counts at beta = 1: positives = modified targeted reports (submitted values),
    negatives = unmodified reports (honest values); mean counts per run. Integrity micro TPR = TP / positives (mean over
    runs); the beta = 1 type macro and the main CW value (compromise-ratio cells within type, then types) of Operational
    TPR are listed for comparison.
(c) Precision under an assumed report-level tampering prevalence pi: per run,
    pi * a / (pi * a + (1 - pi) * f) with a = integrity micro TPR and f = integrity false-alert rate of (b); mean over runs.
(d) Workload illustrations: benign integrity FPR (CW, RD) multiplied by an assumed number of benign reports per
    device-day; RD Node-report IDS and combined FPR (folds pooled within each run).
(e) Benign integrity alerts per nominal device-day at beta = 1: GRIV integrity alerts on the honest values of the
    unmodified reports (all benign at beta = 1) divided by reporters x TEST windows x 60 s, in days. RD sums alerts and
    exposure over the 17 held-out folds within each run. Per run, then the mean and range over runs.
Usage: python supplementary/table_s5_alert_workload.py results/<run_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import common as CM  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from griv import aggregate as AG  # noqa: E402
from griv import bootstrap as BS  # noqa: E402
from griv import metrics as MT  # noqa: E402

PREVALENCE = (0.01, 0.05, 0.10)
VOLUMES = (100, 1000)
ALERTS = (("Node-report IDS", "A_NodeReportIDS_honest"), ("GRIV integrity", "I_GRIV_honest"), ("Combined (Node-report IDS + GRIV)", "O_GRIV_honest"))


def targeted(f) -> np.ndarray:
    """Targeted reports at beta = 1 (per fold for RD)."""
    tb = np.zeros(len(f), bool)
    for _, g in f.groupby(["seed", "fold"]):
        tb[g.index.to_numpy()] = MT.targeted_at(g, 1.0)
    return tb


def burden(f, B, rs) -> pd.DataFrame:
    if not np.array_equal(f["O_GRIV_honest"].to_numpy(bool), f["A_NodeReportIDS_honest"].to_numpy(bool) | f["I_GRIV_honest"].to_numpy(bool)):
        raise AssertionError("combined alert != Node-report IDS alert | GRIV integrity alert")
    if (f["A_NodeReportIDS_honest"].to_numpy(bool) & f["I_GRIV_honest"].to_numpy(bool)).any():
        raise AssertionError("Node-report IDS alert and GRIV integrity alert are not disjoint")
    rows = []
    for name, col in ALERTS:
        c = BS.benign_ci(f.assign(_v=f[col].astype(float)), "_v", "reporter_block60", B=B, rng_seed=rs)
        rows.append(dict(alert=name, column=col, FPR_pct=100 * c["point"], ci_lo_pct=100 * c["ci_lo"], ci_hi_pct=100 * c["ci_hi"],
                         per_1000_benign_reports=1000 * c["point"], B=c["B_requested"], n_clusters=c["n_clusters"]))
    R = pd.DataFrame(rows)
    ben = f["node_label"].to_numpy() == 0; cov = f["covered_honest"].to_numpy(bool)
    cv = f[ben & cov].groupby("seed")["I_GRIV_honest"].mean().mean()
    R["GRIV_integrity_FPR_covered_benign_pct"] = 100 * cv
    R["GRIV_relative_increment_over_Node_IDS_pct"] = 100 * R.loc[1, "FPR_pct"] / R.loc[0, "FPR_pct"]
    R["n_benign_reports_per_run"] = f[ben].groupby("seed").size().mean()
    return R


def confusion(f, tb) -> pd.DataFrame:
    rows = []
    for seed, g in f.groupby("seed"):
        t = tb[g.index.to_numpy()]
        T = MT.tampered_mask(g, t, "fixed")
        I = np.where(T, g["I_GRIV"].to_numpy(bool), g["I_GRIV_honest"].to_numpy(bool))
        rows.append(dict(seed=seed, TP=int((I & T).sum()), FN=int((~I & T).sum()), FP=int((I & ~T).sum()), TN=int((~I & ~T).sum()),
                         n_pos=int(T.sum()), n_neg=int((~T).sum()), n_neg_benign=int((~T & (g["node_label"].to_numpy() == 0)).sum())))
    C = pd.DataFrame(rows)
    C["integrity_micro_TPR"] = C["TP"] / C["n_pos"]; C["integrity_FPR"] = C["FP"] / C["n_neg"]
    for p in PREVALENCE:
        a, fr = C["integrity_micro_TPR"], C["integrity_FPR"]
        C[f"precision_at_{p}"] = p * a / (p * a + (1 - p) * fr)
    return C


def device_day(f, tb, split) -> pd.DataFrame:
    rows = []
    for seed, g in f.groupby("seed"):
        t = tb[g.index.to_numpy()]
        T = MT.tampered_mask(g, t, "fixed"); unt = ~T
        nl = g["node_label"].to_numpy(); I = g["I_GRIV_honest"].to_numpy(bool)
        fold = g.groupby("fold").agg(n_rep=("n_test_reporters", "first"), nw=("n_test_windows", "first"))
        exposure = float((fold["n_rep"] * fold["nw"]).sum()) * 60 / 86400
        rows.append(dict(split=split, seed=seed, reporter_windows=int((fold["n_rep"] * fold["nw"]).sum()), nominal_device_days=exposure,
                         n_unmodified_benign=int((unt & (nl == 0)).sum()), n_unmodified_attack=int((unt & (nl == 1)).sum()),
                         integrity_alerts=int(I[unt].sum()), alerts_per_nominal_device_day=float(I[unt].sum()) / exposure))
    return pd.DataFrame(rows)


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, "supplementary"); cfg = CM.run_config(run)
    B, rs = int(cfg["bootstrap"]["B"]), int(cfg["bootstrap"]["rng_seed"])
    res = {sp: CM.load_setting(run, "complete_concealment", sp, CM.CONCEALMENT, "alpha=1.0") for sp in ("CW", "RD")}
    f = {sp: r.f.reset_index(drop=True) for sp, r in res.items()}
    tb = {sp: targeted(x) for sp, x in f.items()}

    A = burden(f["CW"], B, rs)
    ben = res["CW"].benign()
    checks = {"burden_vs_untampered_summary_max_abs_diff_pct": float(max(
        abs(A.loc[1, "FPR_pct"] - 100 * CM.value(ben, "GRIV", "benign_integrity_FPR")),
        abs(A.loc[2, "FPR_pct"] - 100 * CM.value(ben, "GRIV", "benign_operational_FPR")),
        abs(A.loc[0, "FPR_pct"] - 100 * CM.value(ben, "A:NodeReportIDS", "benign_operational_FPR"))))}

    Cr = confusion(f["CW"], tb["CW"])
    num = [c for c in Cr.columns if c != "seed"]
    Cm = Cr[num].mean().to_frame().T
    Cm["beta1_type_macro_operational_TPR"] = CM.value(AG.macro(res["CW"].A, "operational_TPR", [1.0]).query("stratum == 'all'"), "GRIV")
    Cm["main_CW_operational_TPR"] = CM.value(res["CW"].macro("operational_TPR", ["GRIV"]), "GRIV")
    Cm["node_ids_detected_targets"] = int(sum(((g["post_rf"].to_numpy() == 1) & MT.tampered_mask(g, tb["CW"][g.index.to_numpy()], "fixed")).sum()
                                               for _, g in f["CW"].groupby("seed")))
    Cb = pd.concat([Cr.assign(seed=Cr["seed"].astype(str)), Cm.assign(seed="mean")], ignore_index=True)
    Pc = pd.DataFrame([dict(assumed_prevalence=p, mean_integrity_precision_pct=100 * Cr[f"precision_at_{p}"].mean()) for p in PREVALENCE])

    W = []
    for sp in ("CW", "RD"):
        b = res[sp].benign()
        for v in VOLUMES:
            W.append(dict(split=sp, assumed_benign_reports_per_device_day=v,
                          integrity_false_alerts=v * CM.value(b, "GRIV", "benign_integrity_FPR"),
                          benign_integrity_FPR_pct=100 * CM.value(b, "GRIV", "benign_integrity_FPR"),
                          node_ids_FPR_pct=100 * CM.value(b, "A:NodeReportIDS", "benign_operational_FPR"),
                          combined_FPR_pct=100 * CM.value(b, "GRIV", "benign_operational_FPR")))
    Wd = pd.DataFrame(W)

    Dr = pd.concat([device_day(f[sp], tb[sp], sp) for sp in ("CW", "RD")], ignore_index=True)
    Ds = Dr.groupby("split").agg(n_runs=("seed", "nunique"), reporter_windows=("reporter_windows", "first"),
                                 nominal_device_days=("nominal_device_days", "first"),
                                 n_unmodified_benign=("n_unmodified_benign", "mean"), n_unmodified_attack=("n_unmodified_attack", "max"),
                                 mean_integrity_alerts=("integrity_alerts", "mean"), min_integrity_alerts=("integrity_alerts", "min"),
                                 max_integrity_alerts=("integrity_alerts", "max"),
                                 mean_alerts_per_nominal_device_day=("alerts_per_nominal_device_day", "mean"),
                                 min_alerts_per_nominal_device_day=("alerts_per_nominal_device_day", "min"),
                                 max_alerts_per_nominal_device_day=("alerts_per_nominal_device_day", "max")).reset_index()
    if (Ds["n_unmodified_attack"] != 0).any():
        raise AssertionError("unmodified attack reports at beta = 1: the device-day numerator is not benign-only")
    checks["device_day_vs_untampered_summary_max_abs_diff"] = float(max(
        abs(Ds.set_index("split").loc[sp, "mean_alerts_per_nominal_device_day"]
            - CM.value(res[sp].benign(), "GRIV", "integrity_alerts_per_device_day_equivalent")) for sp in ("CW", "RD")))

    outs = [CM.write_csv(A, out, "table_s5a_benign_alert_burden"), CM.write_csv(Cb, out, "table_s5b_integrity_confusion"),
            CM.write_csv(Pc, out, "table_s5c_prevalence_precision"), CM.write_csv(Wd, out, "table_s5d_workload"),
            CM.write_csv(Ds, out, "table_s5e_device_day"), CM.write_csv(Dr, out, "table_s5e_device_day_by_run")]
    CM.write_provenance(run, out, "supplementary/table_s5_alert_workload.py",
                        ["shards complete_concealment CW alpha=1.0", "shards complete_concealment RD alpha=1.0"], outs,
                        dict(bootstrap_B=B, rng_seed=rs, cluster="reporter_block60 (unstratified)", **checks))
    pd.set_option("display.width", 250)
    for T in (A, Cm, Pc, Wd, Ds):
        print(T.round(4).to_string(index=False))
    print(checks)


if __name__ == "__main__":
    main(sys.argv)
