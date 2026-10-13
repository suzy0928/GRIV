"""Supplementary Fig. S2 (Section 6.3; CW, complete concealment alpha = 1, s = 0.3, 20 runs).

Fig. S2 (a): Operational TPR per attack type for GRIV and the five baselines (compromise-ratio macro within
the type, mean over runs).
Fig. S2 (b): GRIV Covered TPR and target coverage (gateway-observation availability among
modified targets) per attack type.
Usage: python supplementary/fig_s2_attack_types.py results/<run_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import common as CM  # noqa: E402
import pandas as pd  # noqa: E402


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, "supplementary")
    res = CM.load_setting(run, "complete_concealment", "CW", CM.CONCEALMENT, "alpha=1.0")
    rows = []
    for metric in ("operational_TPR", "covered_TPR", "tampered_report_coverage"):
        pt = res.per_type(metric, CM.ROWS)
        for _, r in pt.iterrows():
            rows.append(dict(attack_type=r["attack_type"], row=r["row"], display=CM.DISPLAY[r["row"]], metric=metric,
                             value_pct=100 * r["value"], seed_sd_pct=100 * r["seed_sd"], n_runs=int(r["n_seeds"])))
    T = pd.DataFrame(rows)
    outs = [CM.write_csv(T, out, "fig_s2_attack_types")]
    CM.write_provenance(run, out, "supplementary/fig_s2_attack_types.py", ["shards complete_concealment CW alpha=1.0"], outs)
    print(T[T.metric == "operational_TPR"].pivot_table(index="attack_type", columns="display", values="value_pct").round(1).to_string())
    print(T[(T.row == "GRIV") & (T.metric != "operational_TPR")].pivot_table(index="attack_type", columns="metric", values="value_pct").round(1).to_string())


if __name__ == "__main__":
    main(sys.argv)
