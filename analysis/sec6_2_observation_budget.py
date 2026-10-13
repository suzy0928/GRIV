"""Section 6.2, Fig. 4: gateway observation budget and processing cost under complete concealment (alpha = 1, CW).

(a) GRIV Operational TPR and gateway-observation availability (tampered-report coverage) among actually modified target
    reports, per sampling rate s (compromise-ratio and attack-type macro, mean over 20 runs);
(b) median processing latency per report of the whole GRIV online path; (c) peak RSS of a fresh process replaying the
    whole TEST partition (one run). The flow-retention fraction is reported with the cost table.
Usage: python analysis/sec6_2_observation_budget.py results/<run_id>
"""
from __future__ import annotations

import sys

import common as CM
import pandas as pd

SECTION = "sec6_2_observation_budget"


def detection(run, cfg) -> pd.DataFrame:
    rows = []
    for s in cfg["s_grid"]:
        res = CM.load_setting(run, "sampling_rate", "CW", CM.CONCEALMENT, "alpha=1.0", s=s)
        tpr = res.macro("operational_TPR", ["GRIV"]); cov = res.macro("tampered_report_coverage", ["GRIV"])
        rows.append(dict(s=s, GRIV_operational_TPR_pct=100 * CM.value(tpr, "GRIV"), seed_sd_pct=100 * CM.value(tpr, "GRIV", "seed_sd"),
                         availability_pct=100 * CM.value(cov, "GRIV"), availability_seed_sd_pct=100 * CM.value(cov, "GRIV", "seed_sd"),
                         n_runs=int(CM.value(tpr, "GRIV", "n_seeds"))))
    return pd.DataFrame(rows)


def cost(run) -> pd.DataFrame:
    c = CM.cost_table(run)
    if c.empty:
        return c
    e = c[c["stage"] == "E_griv_online_path"].copy()
    e["median_latency_us_per_report"] = e["per_report_median_us"]
    e["peak_rss_MiB"] = e["stageE_subprocess_peak_rss_bytes"] / 2 ** 20
    e["flow_retention_pct"] = 100 * e["n_flows_kept"] / e["n_flows_test"]
    return e[["s", "seed", "n_reports", "n_flows_test", "n_flows_kept", "flow_retention_pct", "median_latency_us_per_report",
              "peak_rss_MiB"]].sort_values("s").reset_index(drop=True)


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, SECTION); cfg = CM.run_config(run)
    det = detection(run, cfg); co = cost(run)
    outs = [CM.write_csv(det, out, "fig4a_detection_availability"), CM.write_csv(co, out, "fig4bc_cost")]
    CM.write_provenance(run, out, "analysis/sec6_2_observation_budget.py",
                        ["shards sampling_rate CW alpha=1.0 (all s)", "shards cost CW (cost.parquet, one run)"], outs)
    print(det.round(2).to_string(index=False)); print(co.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
