"""Supplementary Table S7 and Fig. S3: decision-rule ablation (how the node evidence enters the integrity decision).

Three rules share the calibrated probabilities, the coverage condition and the threshold rule (VAL_THR covered-benign
q99); only the node term differs:
  GRIV                    1[Node-report IDS = benign] * p_G   (binary node decision)
  Multiplicative score    (1 - p_N) * p_G               (continuous weight)
  Probability Difference  p_G - p_N                     (continuous subtraction; the Section 6.3 baseline)
(a) definitions; (b) Operational TPR under complete concealment (alpha = 1) and adaptive concealment (step 0.01, with
95 % cluster-bootstrap CIs for CW; steps 0.02 and 0.05), benign integrity FPR, for CW (20 runs) and RD (10 runs);
(c) evasion mechanism, CW, beta = 1, adaptive step 0.01, targeted attack-bearing reports pooled over runs (micro):
for all evading reports and for the reports that evade a continuous rule but not GRIV, the alpha chosen by the
attacker, the calibrated p_N of the submitted report, the Node-report IDS decision and coverage; and the share of the reports
evading GRIV that also evade each rule.
Fig. S3: calibrated p_N of the submitted report, (a) among all evading reports, (b) among reports that evade the
Multiplicative score or Probability Difference but not GRIV.
Usage: python supplementary/table_s7_decision_rule.py results/<run_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import common as CM  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from griv import metrics as MT  # noqa: E402

RULES = ["GRIV", "MultiplicativeScore", "ProbabilityDifference"]
DEF = {"GRIV": ("1[Node-report IDS = benign] * p_G", "binary node decision"),
       "MultiplicativeScore": ("(1 - p_N) * p_G", "continuous weight"),
       "ProbabilityDifference": ("p_G - p_N", "continuous subtraction")}
STEPS = ("0.01", "0.02", "0.05")
B, RNG = 10000, 20261002


def results(run, cfg) -> pd.DataFrame:
    rows = []
    for split in ("CW", "RD"):
        fx = CM.load_setting(run, "complete_concealment", split, CM.CONCEALMENT, "alpha=1.0")
        t = fx.macro("operational_TPR", RULES); ben = fx.benign()
        for r in RULES:
            rows.append(dict(split=split, row=r, quantity="fixed_operational_TPR_pct", value=100 * CM.value(t, r),
                             seed_sd=100 * CM.value(t, r, "seed_sd")))
            rows.append(dict(split=split, row=r, quantity="benign_integrity_FPR_pct",
                             value=100 * CM.value(ben, r, "benign_integrity_FPR")))
        for step in (STEPS if split == "CW" else STEPS[:1]):
            ad = CM.load_setting(run, "adaptive", split, CM.ADAPTIVE, f"step={step}", adaptive=True)
            e = ad.macro("simultaneous_evasion", RULES)
            ci = CM.adaptive_bootstrap(ad, RULES, B, RNG).set_index("row") if (split == "CW" and step == "0.01") else None
            for r in RULES:
                rows.append(dict(split=split, row=r, quantity=f"adaptive_operational_TPR_pct_step{step}",
                                 value=100 * (1 - CM.value(e, r)), seed_sd=100 * CM.value(e, r, "seed_sd"),
                                 ci_lo=100 * ci.loc[r, "ci_lo"] if ci is not None else np.nan,
                                 ci_hi=100 * ci.loc[r, "ci_hi"] if ci is not None else np.nan))
    R = pd.DataFrame(rows)
    R.insert(2, "display", R["row"].map(CM.DISPLAY))
    return R


def mechanism(run, s) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = CM.shards(run, experiment="adaptive", split="CW", attack=CM.ADAPTIVE, param="step=0.01", s=s).reset_index(drop=True)
    tg = np.zeros(len(f), bool)
    for _, g in f.groupby("seed"):
        tg[g.index.to_numpy()] = MT.targeted_at(g, 1.0)
    x = f[tg & (f["node_label"].to_numpy() == 1)]
    cov = x["covered"].to_numpy(bool)
    ev = {r: x[f"evaded_{r}"].to_numpy(bool) for r in RULES}
    rows, dist = [], []
    for r in RULES:
        pn = x[f"p_N_{r}"].to_numpy(float); a = x[f"adaptive_alpha_{r}"].to_numpy(float)
        for subset, m in (("all evading reports", ev[r]), ("evading this rule but not GRIV", ev[r] & ~ev["GRIV"])):
            if r == "GRIV" and subset != "all evading reports":
                continue
            rows.append(dict(row=r, display=CM.DISPLAY[r], subset=subset, n_targeted=int(len(x)), n=int(m.sum()),
                             pct_of_targeted=100 * m.mean(), chosen_alpha_median=float(np.median(a[m])),
                             alpha0_pct=100 * float((a[m] == 0).mean()), p_N_q10=float(np.quantile(pn[m], 0.1)),
                             p_N_median=float(np.median(pn[m])), p_N_q90=float(np.quantile(pn[m], 0.9)),
                             p_N_ge_0_2_pct=100 * float((pn[m] >= 0.2).mean()),
                             node_benign_pct=100 * float((x[f"post_rf_{r}"].to_numpy()[m] == 0).mean()),
                             uncovered_pct=100 * float((~cov[m]).mean()),
                             GRIV_evaders_also_evading_pct=100 * float((m & ev["GRIV"]).sum() / ev["GRIV"].sum())
                             if subset == "all evading reports" else np.nan))
            dist.append(pd.DataFrame(dict(row=r, subset=subset, p_N=pn[m], chosen_alpha=a[m])))
    return pd.DataFrame(rows), pd.concat(dist, ignore_index=True)


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, "supplementary"); cfg = CM.run_config(run)
    s = cfg["gateway"]["s_main"]
    D = pd.DataFrame([dict(row=r, display=CM.DISPLAY[r], integrity_score_if_covered=DEF[r][0], node_term=DEF[r][1],
                           threshold="VAL_THR covered-benign q99") for r in RULES])
    R = results(run, cfg)
    Mx, Dst = mechanism(run, s)
    outs = [CM.write_csv(D, out, "table_s7a_decision_rule_definitions"), CM.write_csv(R, out, "table_s7b_decision_rule_results"),
            CM.write_csv(Mx, out, "table_s7c_decision_rule_mechanism")]
    CM.write_provenance(run, out, "supplementary/table_s7_decision_rule.py",
                        ["shards complete_concealment CW/RD alpha=1.0", "shards adaptive CW step=0.01/0.02/0.05, RD step=0.01"], outs,
                        {"bootstrap_B": B, "rng_seed": RNG, "mechanism_population": "CW, beta = 1, step 0.01, targeted attack-bearing TEST reports, all runs pooled"})
    print(R.pivot_table(index=["split", "display"], columns="quantity", values="value", sort=False).round(2).to_string())
    print(Mx.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
