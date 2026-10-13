"""Supplementary Table S8: operating points and FPR-matched comparison of the integrity methods (CW, s = 0.3, 20 runs).

The integrity methods are GRIV, the five cross-view baselines and the multiplicative-score ablation. Every value is
computed from the stored per-report scores; no model is refitted and no attack is re-run.

Integrity score of a report (submitted values for modified targeted reports, honest values otherwise):
  GRIV = p_G if the gateway observation is available and the node-report IDS decision is benign, otherwise -inf (the gate is
  kept; only the threshold varies). Other methods = their integrity score if the observation is available, otherwise
  -inf. Alert = score > threshold (strict).
Metrics:
  - integrity FPR = integrity alerts over all unmodified benign TEST reports (per run, then mean over runs);
  - integrity TPR = integrity alerts over modified targeted reports under complete concealment (alpha = 1);
  - adaptive simultaneous evasion = 1 - adaptive Operational TPR.
  Attack metrics: mean over compromise ratios within each attack type -> seven-type macro -> mean over runs.
  95 % CIs: cluster bootstrap (Section 5.6; benign FPR: same clusters without strata).
Operating points: the validation-fixed point (VAL_THR q99, the reported setting) and post hoc TEST-retuned thresholds at
nominal integrity-FPR targets of 1 / 2 / 5 %: per run, the lowest threshold whose TEST benign integrity alert rate does not
exceed the target (uses TEST labels; not a deployable operating point).
The panel column follows Supplementary Table S8:
  (a) simultaneous evasion of the alpha-grid attacker, restricted to the stored concealment strengths
      alpha in {0, 0.1, ..., 0.9, 0.95, 0.99, 1.0}: a targeted report evades if, for some alpha on the grid, the node-report
      IDS decision is benign and the method raises no integrity alert. The grid is a subset of the main search grid, so
      this attacker is never stronger than the main attack at the same threshold;
  (b) achieved TEST integrity FPR at the TEST-retuned thresholds;
  (c) integrity FPR (q99) and complete-concealment integrity TPR at every operating point, with 95 % CIs, and the
      threshold-free ROC-AUC and PR-AUC (average precision) under complete concealment (mean over runs);
  text: adaptive simultaneous evasion of the main attack (search step 0.01) at q99, compared in the Section S8 text.
Usage: python supplementary/table_s8_matched_fpr.py results/<run_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import common as CM  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from griv import bootstrap as BS  # noqa: E402
from griv import metrics as MT  # noqa: E402
from griv.errors import ValidityError  # noqa: E402

METHODS = CM.INTEGRITY_ROWS + ["MultiplicativeScore"]
TARGETS = [0.01, 0.02, 0.05]
KEYS = ["seed", "device_id", "window"]
LEVEL = "7-type macro"


def score(f, submitted, m) -> np.ndarray:
    """Integrity score per report; `submitted` selects the submitted values (otherwise honest values)."""
    T = np.asarray(submitted, bool)
    s = np.where(T, f[f"score_{m}"], f[f"score_{m}_honest"]).astype(float)
    cov = np.where(T, f["covered"], f["covered_honest"]).astype(bool)
    s = np.where(cov, s, -np.inf)
    if m == "GRIV":
        node = np.where(T, f["post_rf"], f["pre_rf"]).astype(int)
        s = np.where(node == 0, s, -np.inf)
    return np.where(np.isfinite(s), s, -np.inf)


def matched_threshold(neg, target) -> tuple[float, float]:
    """Lowest threshold whose alert rate (score > threshold) on benign scores `neg` does not exceed `target`."""
    x = np.asarray(neg, float); n = len(x)
    fin = np.sort(x[np.isfinite(x)])
    if len(fin) / n <= target:                           # alerting on every finite score stays within the target
        return -np.inf, len(fin) / n
    uniq = np.unique(fin)
    cnt_gt = len(fin) - np.searchsorted(fin, uniq, side="right")
    j = int(np.argmax(cnt_gt / n <= target))
    return float(uniq[j]), float(cnt_gt[j] / n)


class Frames:
    def __init__(self, run, cfg):
        self.run, self.cfg = run, cfg
        self.betas = list(cfg["compromise"]["beta_grid"])
        self.B = int(cfg["bootstrap"]["B"]); self.rs = int(cfg["bootstrap"]["rng_seed"])
        self.full = CM.load_setting(run, "complete_concealment", "CW", CM.CONCEALMENT, "alpha=1.0").f
        self.full = self.full.sort_values(KEYS).reset_index(drop=True)
        self.grid = {}
        for a in cfg["attacks"]["concealment"]["alpha_grid"]:
            g = CM.load_setting(run, "concealment_strength", "CW", CM.CONCEALMENT, f"alpha={a}").f.sort_values(KEYS).reset_index(drop=True)
            if not g[KEYS].equals(self.full[KEYS]):
                raise ValidityError(f"alpha={a}: report keys differ from the complete-concealment shards")
            self.grid[float(a)] = g
        self.adaptive = CM.load_setting(run, "adaptive", "CW", CM.ADAPTIVE, "step=0.01", adaptive=True)
        self.ben = self.full["node_label"].to_numpy() == 0
        self.seed = self.full["seed"].to_numpy()
        T1 = np.zeros(len(self.full), bool)
        for sd, g in self.full.groupby("seed"):
            T1[g.index] = MT.tampered_mask(g, MT.targeted_at(g, 1.0), "fixed")
        self.T1 = T1

    def per_seed(self, values: dict) -> np.ndarray:
        return pd.Series(self.seed).map(values).to_numpy(float)

    def stored_tau(self, m) -> dict:
        t = self.full.groupby("seed")[f"tau_{m}"].agg(["min", "max"])
        if (t["min"] != t["max"]).any():
            raise ValidityError(f"tau_{m} is not constant within a run")
        return t["min"].to_dict()

    def fixed(self, thr_by_m: dict) -> dict:
        """Integrity FPR [CI] and complete-concealment integrity TPR [CI] for per-run thresholds of every method."""
        f = self.full.copy(); cols = {}
        fpr = {}
        for m in METHODS:
            thr = self.per_seed(thr_by_m[m])
            f[f"_tpr_{m}"] = (score(f, self.T1, m) > thr).astype(float)
            f[f"_fpr_{m}"] = (score(f, np.zeros(len(f), bool), m) > thr).astype(float)
            cols[m] = f"_tpr_{m}"
            fpr[m] = BS.benign_ci(f, f"_fpr_{m}", "reporter_block60", B=self.B, rng_seed=self.rs)
        prep = BS.prepare(f, self.betas, cols, "reporter_block60", adaptive=False)
        tpr = BS.level_cis(prep, METHODS, B=self.B, rng_seed=self.rs)
        return {m: dict(FPR=fpr[m], TPR=tpr[m]) for m in METHODS}, f

    def grid_evasion(self, thr_by_m: dict) -> dict:
        """Adaptive simultaneous evasion [CI] of the alpha-grid attacker."""
        f = self.full.copy(); cols = {}
        for m in METHODS:
            thr = self.per_seed(thr_by_m[m])
            nb = f["pre_rf"].to_numpy() == 0
            ev = nb & ~(score(f, np.zeros(len(f), bool), m) > thr)                     # alpha = 0
            for g in self.grid.values():
                nb = g["post_rf"].to_numpy() == 0
                ev |= nb & ~(score(g, np.ones(len(g), bool), m) > thr)
            f[f"_det_{m}"] = (~ev).astype(float); cols[m] = f"_det_{m}"
        prep = BS.prepare(f, self.betas, cols, "reporter_block60", adaptive=True)
        ci = BS.level_cis(prep, METHODS, B=self.B, rng_seed=self.rs)
        return {m: dict(point=1 - v["point"], ci_lo=1 - v["ci_hi"], ci_hi=1 - v["ci_lo"]) for m, v in ci.items()}, f


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, "supplementary"); cfg = CM.run_config(run)
    F = Frames(run, cfg)
    rows, thr_rows, checks = [], [], {}

    # ---- (a) validation-fixed q99
    q99 = {m: F.stored_tau(m) for m in METHODS}
    fx, ff = F.fixed(q99)
    stored = {m: int(((ff[f"_tpr_{m}"].astype(bool) != F.full[f"I_{m}"].astype(bool)) & F.T1).sum()
                     + ((ff[f"_fpr_{m}"].astype(bool) != F.full[f"I_{m}_honest"].astype(bool)) & F.ben).sum()) for m in METHODS}
    checks["q99_alerts_vs_stored_mismatches"] = stored
    ad = F.adaptive.macro("simultaneous_evasion", METHODS)
    for m in METHODS:
        rows += [dict(panel="c", method=m, operating_point="VAL_THR q99", metric="integrity_FPR", value=fx[m]["FPR"]["point"],
                      ci_lo=fx[m]["FPR"]["ci_lo"], ci_hi=fx[m]["FPR"]["ci_hi"]),
                 dict(panel="c", method=m, operating_point="VAL_THR q99", metric="integrity_TPR_complete_concealment",
                      value=fx[m]["TPR"]["point"], ci_lo=fx[m]["TPR"]["ci_lo"], ci_hi=fx[m]["TPR"]["ci_hi"]),
                 dict(panel="text", method=m, operating_point="VAL_THR q99", metric="adaptive_simultaneous_evasion_main",
                      value=CM.value(ad, m))]
    gq, _ = F.grid_evasion(q99)
    for m in METHODS:
        rows.append(dict(panel="a", method=m, operating_point="VAL_THR q99", metric="adaptive_simultaneous_evasion_alpha_grid",
                         value=gq[m]["point"], ci_lo=gq[m]["ci_lo"], ci_hi=gq[m]["ci_hi"]))
    main_ev = F.adaptive.f.sort_values(KEYS).reset_index(drop=True)
    viol = {}
    for m in METHODS:                                                   # grid attacker never stronger than the main attack
        thr = F.per_seed(q99[m]); g_ev = F.full["pre_rf"].to_numpy() == 0
        g_ev &= ~(score(F.full, np.zeros(len(F.full), bool), m) > thr)
        for g in F.grid.values():
            g_ev |= (g["post_rf"].to_numpy() == 0) & ~(score(g, np.ones(len(g), bool), m) > thr)
        viol[m] = int((g_ev & ~main_ev[f"evaded_{m}"].to_numpy(bool) & F.full["is_targeted"].to_numpy(bool)).sum())
    checks["alpha_grid_evaded_not_in_main_attack"] = viol

    # ---- TEST-retuned thresholds: (a) alpha-grid evasion, (b) achieved FPR, (c) TPR
    for tg in TARGETS:
        thr = {m: {} for m in METHODS}
        for sd, g in F.full.groupby("seed"):
            bn = g["node_label"].to_numpy() == 0
            for m in METHODS:
                t, a = matched_threshold(score(g, np.zeros(len(g), bool), m)[bn], tg)
                thr[m][sd] = t
                thr_rows.append(dict(method=m, seed=sd, target_FPR=tg, threshold=t, achieved_FPR=a, n_benign=int(bn.sum())))
        fx, _ = F.fixed(thr); ge, _ = F.grid_evasion(thr)
        for m in METHODS:
            op = f"TEST target {int(tg * 100)}%"
            rows += [dict(panel="b", method=m, operating_point=op, metric="integrity_FPR_achieved", value=fx[m]["FPR"]["point"]),
                     dict(panel="c", method=m, operating_point=op, metric="integrity_TPR_complete_concealment",
                          value=fx[m]["TPR"]["point"], ci_lo=fx[m]["TPR"]["ci_lo"], ci_hi=fx[m]["TPR"]["ci_hi"]),
                     dict(panel="a", method=m, operating_point=op, metric="adaptive_simultaneous_evasion_alpha_grid",
                          value=ge[m]["point"], ci_lo=ge[m]["ci_lo"], ci_hi=ge[m]["ci_hi"])]
    Tb = pd.DataFrame(thr_rows)
    checks["matched_FPR_le_target"] = bool((Tb["achieved_FPR"] <= Tb["target_FPR"] + 1e-12).all())

    # ---- (c) ROC-AUC, PR-AUC
    from sklearn.metrics import average_precision_score, roc_auc_score
    for m in METHODS:
        au = []
        for sd, g in F.full.groupby("seed"):
            T = F.T1[g.index]
            s = score(g, T, m); s = np.where(np.isfinite(s), s, -1e18)
            au.append((roc_auc_score(T, s), average_precision_score(T, s)))
        rows += [dict(panel="c", method=m, operating_point="threshold-free", metric="ROC_AUC", value=float(np.mean([a for a, _ in au]))),
                 dict(panel="c", method=m, operating_point="threshold-free", metric="PR_AUC", value=float(np.mean([b for _, b in au])))]

    P = pd.DataFrame(rows)
    P.insert(2, "display", P["method"].map(CM.DISPLAY))
    outs = [CM.write_csv(P, out, "table_s8_matched_fpr"), CM.write_csv(Tb, out, "table_s8_matched_thresholds_by_run")]
    CM.write_provenance(run, out, "supplementary/table_s8_matched_fpr.py",
                        ["shards complete_concealment CW alpha=1.0", "shards concealment_strength CW (alpha grid)",
                         "shards adaptive CW step=0.01"], outs, checks)
    show = P.assign(v=lambda x: (100 * x["value"]).where(~x["metric"].str.contains("AUC"), x["value"]))
    print(show.pivot_table(index="display", columns=["panel", "operating_point", "metric"], values="v", sort=False).round(2).T.to_string())
    print(checks)


if __name__ == "__main__":
    main(sys.argv)
