"""Supplementary Table S2 (a)-(c) and the RD fold design of (d): partition dates, report counts, attack-type composition, exclusion ledger and held-out reporters (Section 5.2).

Reads the partition tables written by the runner (results/<run_id>/partitions/). The exclusion ledger counts, for every
window assignment (four partitions and three purge steps), the windows and the reports of the monitored reporters;
reports of monitored reporters outside the CW reporter set (reporters without both TRAIN and TEST reports) are counted
separately. Network-wide window labels count attack flows of all devices (n_attack_windows of the partition
table counts windows with attack-bearing reports of the CW reporters). The ledger reads the node-window reports
(configs/local_paths.yaml) and rebuilds the CW reporter set.
Usage: python supplementary/table_s2_partitions.py results/<run_id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import common as CM  # noqa: E402
import pandas as pd  # noqa: E402

from griv import config as C  # noqa: E402
from griv import data as D  # noqa: E402
from griv import splits as S  # noqa: E402
from griv.config import TYPES  # noqa: E402

PURGE = {"PURGED:stratum_purge": "stratum boundary purge", "PURGED:valcal_valthr_purge": "VAL_CAL / VAL_THR gap",
         "PURGED:xpurge": "cross-label purge (TEST)"}


def ledger(run) -> pd.DataFrame:
    """Windows and reports per window assignment; the four partitions first, then the purge steps, then the total."""
    asg = pd.read_parquet(run / "partitions" / "CW__-" / "partitions.parquet")
    rc = CM.run_config(run); cfg = C.load(rc["_path"]); d = D.load(cfg)
    cw = S.make(d, cfg, "CW")
    for p in S.PARTS:
        if sorted(asg.loc[asg["partition"] == p, "window"]) != sorted(map(int, cw.windows[p])):
            raise AssertionError(f"rebuilt CW windows differ from partitions.parquet ({p})")
    rep = set(cw.reporters["TRAIN"])
    o = d.o.merge(asg[["window", "partition"]], on="window", how="inner")
    o["cw_reporter"] = o["device_id"].isin(rep)
    gl = asg["window"].map(d.gl).fillna(0).astype(int)
    rows = []
    for p in S.PARTS + list(PURGE):
        w = asg["partition"] == p; x = o[o["partition"] == p]; xc = x[x["cw_reporter"]]
        rows.append(dict(assignment=p if p in S.PARTS else "excluded: " + PURGE[p], windows=int(w.sum()),
                         attack_free_windows_network_wide=int((w & (gl == 0)).sum()),
                         attack_windows_network_wide=int((w & (gl == 1)).sum()),
                         reports_cw_reporters=int(len(xc)), attack_bearing_reports_cw_reporters=int((xc["node_label"] == 1).sum()),
                         reports_other_monitored_reporters=int((~x["cw_reporter"]).sum())))
    L = pd.DataFrame(rows)
    tot = L.drop(columns="assignment").sum(); tot["assignment"] = "total"
    L = pd.concat([L, tot.to_frame().T], ignore_index=True)
    if int(tot["windows"]) != len(asg) or int(tot["reports_cw_reporters"] + tot["reports_other_monitored_reporters"]) != len(o):
        raise AssertionError("ledger does not add up")
    return L


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, "supplementary")
    pdir = run / "partitions"
    cw = pd.read_csv(pdir / "CW__-" / "partition_table.csv")
    meta = json.loads((pdir / "CW__-" / "split_meta.json").read_text(encoding="utf-8"))
    xp = pd.DataFrame([dict(item="TEST windows removed by the cross-label purge (benign / attack)",
                            value=f"{meta['xpurge_test_windows_excluded']['benign']} / {meta['xpurge_test_windows_excluded']['attack']}"),
                       dict(item="TEST attack-window loss from the cross-label purge (%)", value=round(100 * meta["test_attack_window_loss"], 2))])
    rows = []
    for d in sorted(pdir.glob("RD__*")):
        t = pd.read_csv(d / "partition_table.csv")
        te = t[t["partition"] == "TEST"].iloc[0]; tr = t[t["partition"] == "TRAIN"].iloc[0]
        types = [ty for ty in TYPES if te[f"attack_reports_{ty}"] > 0]
        rows.append(dict(held_out_reporter=te["fold"], test_reports=int(te["n_reports"]), test_benign_reports=int(te["n_benign_reports"]),
                         test_attack_bearing_reports=int(te["n_attack_reports"]), test_attack_types=";".join(types),
                         attack_metric_fold=bool(te["n_attack_reports"] > 0), training_reporters=int(tr["n_reporters"])))
    rd = pd.DataFrame(rows)
    L = ledger(run)
    outs = [CM.write_csv(cw, out, "table_s2ab_cw_partitions"), CM.write_csv(xp, out, "table_s2c_cross_label_purge"),
            CM.write_csv(rd, out, "table_s2d_rd_fold_design"), CM.write_csv(L, out, "table_s2c_exclusion_ledger")]
    CM.write_provenance(run, out, "supplementary/table_s2_partitions.py",
                        ["partitions/*/partition_table.csv", "partitions/CW__-/split_meta.json", "partitions/CW__-/partitions.parquet",
                         "o_it_R2.parquet (node-window reports)"],
                        outs, {"n_rd_folds": len(rd), "n_rd_attack_folds": int(rd["attack_metric_fold"].sum()) if len(rd) else 0})
    print(cw.to_string(index=False)); print(xp.to_string(index=False)); print(rd.to_string(index=False)); print(L.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
