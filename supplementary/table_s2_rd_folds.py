"""Supplementary Table S2 (d): fold-level results on held-out reporters (RD, s = 0.3, 10 runs; Section 6.5).

Per held-out reporter: Operational TPR under complete and adaptive (step 0.01) concealment (type macro over the types
present in the fold) and benign integrity FPR, mean over runs.
Usage: python supplementary/table_s2_rd_folds.py results/<run_id>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
import common as CM  # noqa: E402


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, "supplementary")
    fx = CM.load_setting(run, "complete_concealment", "RD", CM.CONCEALMENT, "alpha=1.0")
    ad = CM.load_setting(run, "adaptive", "RD", CM.ADAPTIVE, "step=0.01", adaptive=True)
    P = CM.rd_per_fold(fx, ad)
    W = P.pivot_table(index=["fold"], columns=["quantity", "display"], values="value", sort=False)
    W.columns = [f"{q} | {d}" for q, d in W.columns]
    W = W.reset_index()
    outs = [CM.write_csv(P, out, "table_s2d_rd_per_fold_long"), CM.write_csv(W, out, "table_s2d_rd_per_fold")]
    CM.write_provenance(run, out, "supplementary/table_s2_rd_folds.py",
                        ["shards complete_concealment RD alpha=1.0", "shards adaptive RD step=0.01"], outs)
    print(W.round(2).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
