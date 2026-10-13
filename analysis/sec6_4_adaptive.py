"""Section 6.4, Fig. 6: robustness to adaptive concealment (CW, s = 0.3).

Operational TPR under adaptive concealment = 1 - simultaneous evasion over all targeted reports (including alpha = 0
choices), compromise-ratio macro -> seven-type macro -> mean over 20 runs.
(a) search step 0.01 with 95 % cluster-bootstrap CIs (reporter x 60-minute-block clusters, attack-type strata,
    10,000 resamples, paired across methods); (b) search-step sensitivity (0.01, 0.02, 0.05).
Also: the complete-concealment (alpha = 1) values for the change from fixed to adaptive, and GRIV's remaining
simultaneous evasion.
Usage: python analysis/sec6_4_adaptive.py results/<run_id>
"""
from __future__ import annotations

import sys

import common as CM
import pandas as pd

SECTION = "sec6_4_adaptive"


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, SECTION); cfg = CM.run_config(run)
    steps = [cfg["attacks"]["adaptive"]["alpha_step"]] + list(cfg["attacks"]["adaptive"]["step_sensitivity"])
    fixed = CM.load_setting(run, "complete_concealment", "CW", CM.CONCEALMENT, "alpha=1.0").macro("operational_TPR", CM.ROWS)
    rows, main_res = [], None
    for st in steps:
        res = CM.load_setting(run, "adaptive", "CW", CM.ADAPTIVE, f"step={st}", adaptive=True)
        if st == steps[0]:
            main_res = res
        m = res.macro("simultaneous_evasion", CM.ROWS)
        for r in CM.ROWS:
            rows.append(dict(step=st, row=r, display=CM.DISPLAY[r], operational_TPR_pct=100 * (1 - CM.value(m, r)),
                             simultaneous_evasion_pct=100 * CM.value(m, r), seed_sd_pct=100 * CM.value(m, r, "seed_sd"),
                             n_runs=int(CM.value(m, r, "n_seeds"))))
    st = pd.DataFrame(rows)
    ci = CM.adaptive_bootstrap(main_res, CM.ROWS, int(cfg["bootstrap"]["B"]), int(cfg["bootstrap"]["rng_seed"]))
    a = st[st["step"] == steps[0]].merge(ci[["row", "point", "ci_lo", "ci_hi", "B_requested", "B_finite", "n_clusters"]], on="row")
    a["ci_lo_pct"] = 100 * a["ci_lo"]; a["ci_hi_pct"] = 100 * a["ci_hi"]
    a["fixed_alpha1_operational_TPR_pct"] = [100 * CM.value(fixed, r) for r in a["row"]]
    a["change_fixed_to_adaptive_pp"] = a["operational_TPR_pct"] - a["fixed_alpha1_operational_TPR_pct"]
    if (a["point"] * 100 - a["operational_TPR_pct"]).abs().max() > 1e-9:
        raise AssertionError("bootstrap point estimate differs from the aggregated value")
    a = a.drop(columns=["point", "ci_lo", "ci_hi"])
    outs = [CM.write_csv(a, out, "fig6a_adaptive_with_ci"), CM.write_csv(st, out, "fig6b_step_sensitivity")]
    CM.write_provenance(run, out, "analysis/sec6_4_adaptive.py",
                        ["shards adaptive CW step=0.01/0.02/0.05", "shards complete_concealment CW alpha=1.0"], outs,
                        {"bootstrap": "griv.bootstrap.level_cis, clusters (reporter, window // 60), strata = attack type"})
    print(a.drop(columns=["row"]).round(2).to_string(index=False))
    print(st.pivot_table(index="display", columns="step", values="operational_TPR_pct", sort=False).round(2).to_string())


if __name__ == "__main__":
    main(sys.argv)
