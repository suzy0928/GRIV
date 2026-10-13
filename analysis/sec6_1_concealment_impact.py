"""Section 6.1, Fig. 3: Operational TPR of Node-report IDS, Gateway IDS and Node-report IDS + GRIV under prototype-directed concealment
(CW, s = 0.3). (a) concealment strength alpha at beta = 1; (b) enlargement of alpha = 0.9 - 1.0 (same values as (a));
(c) compromise ratio beta at alpha = 1.

Per report: Node-report IDS = class prediction on the submitted report (honest report if not targeted); Gateway IDS =
covered & p_G > tau_G; Node-report IDS + GRIV = Node-report IDS | I_GRIV. TPR = detected / all attack-bearing TEST reports of the run
(micro), then the mean over 20 runs. Panel (c) adds Node-report IDS on targeted reports only and Node-report IDS on honest reports.
alpha = 0 uses the honest columns of the same shards (Eq. (1) leaves reports unchanged at alpha = 0).
Also reported: benign-report Operational FPR of the three detectors (Section 6.1, last paragraph).
Usage: python analysis/sec6_1_concealment_impact.py results/<run_id>
"""
from __future__ import annotations

import sys

import common as CM
import numpy as np
import pandas as pd

from griv import metrics as MT

SECTION = "sec6_1_concealment_impact"
ZOOM = 0.9


def per_report(f, beta):
    tg = np.zeros(len(f), bool)
    for _, g in f.groupby("seed"):
        tg[g.index.to_numpy()] = MT.targeted_at(g, beta)
    node = np.where(tg, f["post_rf"].to_numpy() == 1, f["pre_rf"].to_numpy() == 1)
    gw = np.where(tg, f["A_GatewayIDS"].to_numpy(bool), f["A_GatewayIDS_honest"].to_numpy(bool))
    op = np.where(tg, f["O_GRIV"].to_numpy(bool), f["O_GRIV_honest"].to_numpy(bool))
    if not np.array_equal(op, node | gw):
        raise AssertionError("Node-report IDS + GRIV != Node-report IDS | Gateway IDS")
    return dict(tg=tg, atk=f["node_label"].to_numpy() == 1, node=node, gw=gw, op=op, honest=f["pre_rf"].to_numpy() == 1)


def curves(f, panel, alpha, beta):
    r = per_report(f, beta); seed = f["seed"].to_numpy(); out = []
    for sd in sorted(set(seed)):
        m = (seed == sd) & r["atk"]
        cs = [("Node-report IDS", r["node"], m), ("Gateway IDS", r["gw"], m), ("Node-report IDS + GRIV", r["op"], m)]
        if panel == "beta":
            cs += [("Node-report IDS (targeted only)", r["node"], m & r["tg"]), ("Node-report IDS (honest reports)", r["honest"], m)]
        for name, hit, den in cs:
            out.append(dict(panel=panel, alpha=alpha, beta=beta, seed=int(sd), curve=name, TP=int((hit & den).sum()),
                            denominator=int(den.sum()), TPR=(hit & den).sum() / den.sum(),
                            targeted_fraction=float((m & r["tg"]).sum() / m.sum())))
    return out


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, SECTION); cfg = CM.run_config(run); s = cfg["gateway"]["s_main"]
    e1 = CM.shards(run, experiment="complete_concealment", split="CW", attack=CM.CONCEALMENT, param="alpha=1.0", s=s)
    rows = []
    f0 = e1.copy()                                   # alpha = 0: honest columns
    f0["post_rf"] = f0["pre_rf"]; f0["O_GRIV"] = f0["O_GRIV_honest"]; f0["A_GatewayIDS"] = f0["A_GatewayIDS_honest"]
    rows += curves(f0, "alpha", 0.0, 1.0)
    for a in cfg["attacks"]["concealment"]["alpha_grid"]:
        f = CM.shards(run, experiment="concealment_strength", split="CW", attack=CM.CONCEALMENT, param=f"alpha={a}", s=s)
        if len(f):
            rows += curves(f.reset_index(drop=True), "alpha", float(a), 1.0)
    for b in cfg["compromise"]["beta_grid"]:
        rows += curves(e1, "beta", 1.0, float(b))
    R = pd.DataFrame(rows)
    Sm = R.groupby(["panel", "alpha", "beta", "curve"], sort=False).agg(
        TPR=("TPR", "mean"), TPR_seed_sd=("TPR", "std"), n_runs=("seed", "nunique"), denominator=("denominator", "mean"),
        targeted_fraction=("targeted_fraction", "mean")).reset_index()
    Sm["TPR_pct"] = CM.pct(Sm["TPR"])
    res = CM.load_setting(run, "complete_concealment", "CW", CM.CONCEALMENT, "alpha=1.0")
    ben = res.benign()
    fpr = pd.DataFrame([dict(row=r, display=CM.DISPLAY.get(r, r), benign_operational_FPR_pct=100 * CM.value(ben, r, "benign_operational_FPR"))
                        for r in ("A:NodeReportIDS", "GRIV", "A:GatewayIDS")])
    outs = [CM.write_csv(R, out, "fig3_curves_by_run"), CM.write_csv(Sm, out, "fig3_curves"),
            CM.write_csv(fpr, out, "benign_operational_FPR")]
    CM.write_provenance(run, out, "analysis/sec6_1_concealment_impact.py",
                        ["shards complete_concealment CW alpha=1.0", "shards concealment_strength CW alpha grid"], outs)
    print(Sm.pivot_table(index=["panel", "alpha", "beta"], columns="curve", values="TPR_pct", sort=False).round(2).to_string())
    print(fpr.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
