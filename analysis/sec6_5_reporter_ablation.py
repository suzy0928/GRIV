"""Section 6.5: reporter generalization (RD, Fig. 7) and the ablation / generalization table (Table 6).

Fig. 7 (RD, s = 0.3, 10 runs): (a) Operational TPR under complete concealment, (b) under adaptive concealment
(step 0.01), (c) benign integrity FPR with validation-q99 thresholds (dashed line = nominal 1 % reference).
Attack metrics pool the reports of the attack-bearing folds within each attack type before the type macro; benign
metrics pool all folds. Also: Operational FPR and the range of GRIV's fold-level Operational TPR.
Table 6: GRIV with the decision gate (RF + Platt), the multiplicative score p_G (1 - p_N), GRIV without Platt
calibration (both views), and GRIV with the MLP classifier (CW), and GRIV on held-out reporters (RD):
complete-concealment and adaptive Operational TPR and benign integrity FPR.
Usage: python analysis/sec6_5_reporter_ablation.py results/<run_id>
"""
from __future__ import annotations

import sys

import common as CM
import pandas as pd

SECTION = "sec6_5_reporter_ablation"


def rd(run):
    fx = CM.load_setting(run, "complete_concealment", "RD", CM.CONCEALMENT, "alpha=1.0")
    ad = CM.load_setting(run, "adaptive", "RD", CM.ADAPTIVE, "step=0.01", adaptive=True)
    t = fx.macro("operational_TPR", CM.ROWS); e = ad.macro("simultaneous_evasion", CM.ROWS); b = fx.benign()
    rows = []
    for r in CM.ROWS:
        rows.append(dict(row=r, display=CM.DISPLAY[r], fixed_operational_TPR_pct=100 * CM.value(t, r),
                         fixed_seed_sd_pct=100 * CM.value(t, r, "seed_sd"), adaptive_operational_TPR_pct=100 * (1 - CM.value(e, r)),
                         adaptive_seed_sd_pct=100 * CM.value(e, r, "seed_sd"),
                         benign_integrity_FPR_pct=100 * CM.value(b, r, "benign_integrity_FPR") if r in CM.INTEGRITY_ROWS else float("nan"),
                         benign_operational_FPR_pct=100 * CM.value(b, r, "benign_operational_FPR"),
                         n_runs=int(CM.value(t, r, "n_seeds")), n_attack_folds=int(fx.f.loc[fx.f["node_label"] == 1, "fold"].nunique()),
                         n_benign_folds=int(fx.f.loc[fx.f["node_label"] == 0, "fold"].nunique())))
    S = pd.DataFrame(rows)
    S["change_fixed_to_adaptive_pp"] = S["adaptive_operational_TPR_pct"] - S["fixed_operational_TPR_pct"]
    return S, fx, ad


def ablation(run) -> pd.DataFrame:
    rows = []
    for label, var, clf, row, fixed_exp, adapt_exp in (
            ("Decision gate, RF with Platt calibration", "base", "rf", "GRIV", "complete_concealment", "adaptive"),
            ("Multiplicative score, RF with Platt calibration", "base", "rf", "MultiplicativeScore", "complete_concealment", "adaptive"),
            ("Decision gate, RF without Platt calibration", "no_platt", "rf", "GRIV", "no_platt", "no_platt"),
            ("Decision gate, MLP with Platt calibration", "base", "mlp", "GRIV", "mlp", "mlp")):
        fx = CM.load_setting(run, fixed_exp, "CW", CM.CONCEALMENT, "alpha=1.0", variant=var, classifier=clf)
        ad = CM.load_setting(run, adapt_exp, "CW", CM.ADAPTIVE, "step=0.01", variant=var, classifier=clf, adaptive=True)
        t = fx.macro("operational_TPR", [row]); e = ad.macro("simultaneous_evasion", [row]); b = fx.benign()
        rows.append(dict(configuration=label, split="CW", row=row, fixed_operational_TPR_pct=100 * CM.value(t, row),
                         adaptive_operational_TPR_pct=100 * (1 - CM.value(e, row)),
                         benign_integrity_FPR_pct=100 * CM.value(b, row, "benign_integrity_FPR"),
                         experiments=f"{fixed_exp} / {adapt_exp}", variant=var, classifier=clf))
    return pd.DataFrame(rows)


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, SECTION)
    S, fx, ad = rd(run); P = CM.rd_per_fold(fx, ad); T = ablation(run)
    g0 = S[S["row"] == "GRIV"].iloc[0]                                      # Table 6, reporter generalization row
    T = pd.concat([T, pd.DataFrame([dict(configuration="Decision gate, RF with Platt calibration", split="RD", row="GRIV",
                                         fixed_operational_TPR_pct=g0["fixed_operational_TPR_pct"],
                                         adaptive_operational_TPR_pct=g0["adaptive_operational_TPR_pct"],
                                         benign_integrity_FPR_pct=g0["benign_integrity_FPR_pct"],
                                         experiments="complete_concealment / adaptive", variant="base", classifier="rf")])],
                  ignore_index=True)
    g = P[(P["row"] == "GRIV") & (P["quantity"] == "fixed_operational_TPR_pct")]["value"]
    notes = {"GRIV_fold_fixed_operational_TPR_range_pct": [float(g.min()), float(g.max())]}
    outs = [CM.write_csv(S, out, "fig7_rd_summary"), CM.write_csv(T, out, "table6_ablation_generalization")]
    CM.write_provenance(run, out, "analysis/sec6_5_reporter_ablation.py",
                        ["shards complete_concealment RD", "shards adaptive RD step=0.01", "shards complete_concealment/adaptive CW",
                         "shards no_platt CW", "shards mlp CW"], outs, notes)
    print(S.drop(columns=["row"]).round(2).to_string(index=False)); print(notes)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
