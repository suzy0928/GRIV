"""Section 6.3: comparison with integrity-verification baselines under fixed-strength concealment (CW, s = 0.3).

Fig. 5: Operational TPR vs concealment strength alpha for GRIV and the five baselines (compromise-ratio macro ->
seven-type macro -> mean over 20 runs); panel (b) enlarges alpha = 0.9 - 1.0; inset = complete concealment (alpha = 1).
Table: complete concealment (alpha = 1) Operational TPR, Covered TPR and tampered-report coverage for all rows, and the
benign integrity FPR (integrity alerts over all unmodified benign TEST reports) and Operational FPR.
Usage: python analysis/sec6_3_baselines.py results/<run_id>
"""
from __future__ import annotations

import sys

import common as CM
import pandas as pd

SECTION = "sec6_3_baselines"
ZOOM = 0.9


def alpha_sweep(run, cfg) -> pd.DataFrame:
    rows = []
    for a in cfg["attacks"]["concealment"]["alpha_grid"]:
        res = CM.load_setting(run, "concealment_strength", "CW", CM.CONCEALMENT, f"alpha={a}")
        m = res.macro("operational_TPR", CM.INTEGRITY_ROWS)
        for r in CM.INTEGRITY_ROWS:
            rows.append(dict(alpha=float(a), row=r, display=CM.DISPLAY[r], operational_TPR_pct=100 * CM.value(m, r),
                             seed_sd_pct=100 * CM.value(m, r, "seed_sd"), n_runs=int(CM.value(m, r, "n_seeds"))))
    return pd.DataFrame(rows)


def complete(run) -> pd.DataFrame:
    res = CM.load_setting(run, "complete_concealment", "CW", CM.CONCEALMENT, "alpha=1.0")
    t = res.macro("operational_TPR", CM.ROWS); c = res.macro("covered_TPR", CM.ROWS); v = res.macro("tampered_report_coverage", CM.ROWS)
    b = res.benign()
    rows = []
    for r in CM.ROWS:
        rows.append(dict(row=r, display=CM.DISPLAY[r], operational_TPR_pct=100 * CM.value(t, r), seed_sd_pct=100 * CM.value(t, r, "seed_sd"),
                         covered_TPR_pct=100 * CM.value(c, r), tampered_report_coverage_pct=100 * CM.value(v, r),
                         benign_integrity_FPR_pct=100 * CM.value(b, r, "benign_integrity_FPR") if r in CM.INTEGRITY_ROWS else float("nan"),
                         benign_operational_FPR_pct=100 * CM.value(b, r, "benign_operational_FPR"),
                         n_benign_reports_per_run=CM.value(b, r, "n_benign")))
    return pd.DataFrame(rows)


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, SECTION); cfg = CM.run_config(run)
    sw = alpha_sweep(run, cfg); cp = complete(run)
    outs = [CM.write_csv(sw, out, "fig5_alpha_sweep"), CM.write_csv(cp, out, "complete_concealment_and_benign_FPR")]
    CM.write_provenance(run, out, "analysis/sec6_3_baselines.py",
                        ["shards concealment_strength CW (alpha grid)", "shards complete_concealment CW alpha=1.0"], outs)
    print(sw.pivot_table(index="alpha", columns="display", values="operational_TPR_pct").round(2).to_string())
    print(cp.drop(columns=["row"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
