"""Supplementary Table S6: held-out-reporter results by attack type (RD, s = 0.3, 10 runs; Section 6.5).

(a) Operational TPR per attack type under adaptive concealment (step 0.01) for GRIV and the baselines, GRIV under
complete concealment (alpha = 1) and its tampered-report coverage, and the CW value of GRIV for reference. Attack
reports of the attack-bearing folds are pooled within each type, then averaged over runs (the aggregation of
Section 6.5). The type macro over all seven types equals the RD values of Section 6.5. The last row repeats the macro
over the types whose TEST reports all come from folds with training reports of that type.
(b) Training presence: for every attack-bearing fold and attack type, the TEST attack reports of the held-out reporter
and the TRAIN attack reports of the same type (read from the partition tables). A type has no training reports in a
fold when the held-out reporter is the only reporter with reports of that type.
Usage: python supplementary/table_s6_attack_types.py results/<run_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import common as CM  # noqa: E402
import pandas as pd  # noqa: E402

from griv.config import TYPES  # noqa: E402

ALL7 = "7-type macro"
SEEN = "macro over types with training reports in every fold"


def training_presence(run) -> pd.DataFrame:
    rows = []
    for d in sorted((Path(run) / "partitions").glob("RD__*")):
        t = pd.read_csv(d / "partition_table.csv").set_index("partition")
        for ty in TYPES:
            n_test = int(t.loc["TEST", f"attack_reports_{ty}"])
            if n_test:
                rows.append(dict(held_out_reporter=t.loc["TEST", "fold"], attack_type=ty, test_attack_reports=n_test,
                                 train_attack_reports=int(t.loc["TRAIN", f"attack_reports_{ty}"])))
    return pd.DataFrame(rows)


def per_type(res, metric, rows, adaptive):
    pt = res.per_type(metric, rows)
    v = 1 - pt["value"] if adaptive else pt["value"]
    return pd.DataFrame(dict(row=pt["row"], attack_type=pt["attack_type"], value_pct=100 * v, seed_sd_pct=100 * pt["seed_sd"],
                             n_runs=pt["n_seeds"].astype(int)))


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, "supplementary")
    rd_fx = CM.load_setting(run, "complete_concealment", "RD", CM.CONCEALMENT, "alpha=1.0")
    rd_ad = CM.load_setting(run, "adaptive", "RD", CM.ADAPTIVE, "step=0.01", adaptive=True)
    cw_ad = CM.load_setting(run, "adaptive", "CW", CM.ADAPTIVE, "step=0.01", adaptive=True)
    L = pd.concat([
        per_type(rd_ad, "simultaneous_evasion", CM.ROWS, True).assign(split="RD", quantity="adaptive_operational_TPR_pct"),
        per_type(rd_fx, "operational_TPR", ["GRIV"], False).assign(split="RD", quantity="fixed_operational_TPR_pct"),
        per_type(rd_fx, "tampered_report_coverage", ["GRIV"], False).assign(split="RD", quantity="tampered_report_coverage_pct"),
        per_type(cw_ad, "simultaneous_evasion", ["GRIV"], True).assign(split="CW", quantity="adaptive_operational_TPR_pct"),
    ], ignore_index=True)
    L["display"] = L["row"].map(CM.DISPLAY)

    P = training_presence(run)
    g = P.groupby("attack_type")
    unseen = (P["train_attack_reports"] == 0)
    pres = pd.DataFrame(dict(test_attack_reports=g["test_attack_reports"].sum(),
                             test_reports_without_training_reports_pct=100 * P[unseen].groupby("attack_type")["test_attack_reports"].sum()
                             .reindex(g.size().index, fill_value=0) / g["test_attack_reports"].sum(),
                             attack_bearing_folds=g.size()))
    seen = [ty for ty in TYPES if pres.loc[ty, "test_reports_without_training_reports_pct"] == 0]

    W = L.pivot_table(index="attack_type", columns=["split", "quantity", "display"], values="value_pct", sort=False)
    W.columns = [f"{s} {q} | {d}" for s, q, d in W.columns]
    order = [f"RD adaptive_operational_TPR_pct | {CM.DISPLAY[r]}" for r in CM.ROWS]
    W = W[order + [c for c in W.columns if c not in order]].reindex(TYPES)
    # type macro: equal to Section 6.5 for all seven types (checked below)
    W.loc[ALL7] = W.loc[TYPES].mean()
    W.loc[SEEN] = W.loc[seen].mean()
    W = W.join(pres).reset_index().rename(columns={"index": "attack_type"})
    W.loc[W["attack_type"] == ALL7, "test_attack_reports"] = pres["test_attack_reports"].sum()
    W.loc[W["attack_type"] == SEEN, "test_attack_reports"] = pres.loc[seen, "test_attack_reports"].sum()

    ref = rd_ad.macro("simultaneous_evasion", CM.ROWS)
    diff = max(abs(W.loc[W["attack_type"] == ALL7, f"RD adaptive_operational_TPR_pct | {CM.DISPLAY[r]}"].iloc[0]
                   - 100 * (1 - CM.value(ref, r))) for r in CM.ROWS)
    if not diff < 1e-9:
        raise AssertionError(f"type macro differs from Section 6.5 RD values (max {diff})")

    outs = [CM.write_csv(L, out, "table_s6_rd_attack_types_long"), CM.write_csv(W, out, "table_s6a_rd_attack_types"),
            CM.write_csv(P, out, "table_s6b_rd_training_presence")]
    CM.write_provenance(run, out, "supplementary/table_s6_attack_types.py",
                        ["shards complete_concealment RD alpha=1.0", "shards adaptive RD step=0.01", "shards adaptive CW step=0.01",
                         "partitions/RD__*/partition_table.csv"], outs,
                        {"types_with_training_reports_in_every_fold": seen, "max_abs_diff_type_macro_vs_section_6_5_pct": float(diff)})
    pd.set_option("display.width", 250)
    print(W.round(2).to_string(index=False))
    print(P[P["train_attack_reports"] == 0].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
