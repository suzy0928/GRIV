"""Section 5: Table 2 (evaluation corpus and attack-type composition) and Table 4 / Supplementary Table S3 (model and operating-point
settings).

Table 2 is computed from the derived data files (configs/local_paths.yaml); the settings table is read from the run's
resolved config.
Usage: python analysis/sec5_dataset_and_settings.py results/<run_id>
"""
from __future__ import annotations

import sys

import common as CM  # noqa: F401  (sets sys.path)
import pandas as pd

from griv import config as C
from griv import data as D
from griv.config import TYPES

SECTION = "sec5_dataset_and_settings"


def table2(cfg) -> pd.DataFrame:
    d = D.load(cfg)
    counts = D.corpus_counts(d.o_all, d.vn, d.dom, D.n_raw_flows(d))
    eps = D.episodes(d.dom, d.gl, int(cfg["split_common"]["episode_gap_min"]))
    rows = [("Flow records involving monitored reporters", counts["raw_flows"], None),
            ("Monitored reporters", counts["reporters"], None),
            ("60-s windows with at least one report", int(d.o["window"].nunique()), None),
            ("Node-window reports", counts["reports"], None),
            ("Benign reports", counts["benign"], None),
            ("Attack-bearing reports", counts["attack"], None)]
    rows += [(t, counts["by_type"].get(t, 0), len(eps[t])) for t in TYPES]
    return pd.DataFrame(rows, columns=["item", "value", "temporal_episodes"])


def settings(cfg) -> pd.DataFrame:
    rf, mlp = cfg["classifier"]["rf"], cfg["mlp"]
    rows = [("Primary classifier", f"Random Forest, {rf['n_estimators']} trees, {rf['class_weight']} class weighting"),
            ("Alternative classifier", f"StandardScaler followed by an MLP with hidden layers {mlp['hidden']} and "
                                       f"{mlp['activation']} activation"),
            ("MLP training", f"class-balanced sample weights, early stopping with patience {mlp['n_iter_no_change']}, "
                             f"maximum {mlp['max_iter']} epochs"),
            ("Probability calibration", "separate Platt models for node and gateway views (VAL_CAL)"),
            ("Primary gateway sampling rate", f"s = {cfg['gateway']['s_main']}"),
            ("Sampling-rate analysis", f"s = {cfg['s_grid']}"),
            ("Threshold rule", f"validation q{int(cfg['threshold']['quantile'] * 100)} with strict threshold exceedance"),
            ("Runs", f"CW {len(cfg['seeds']['cw'])}, RD {len(cfg['seeds']['rd'])}")]
    return pd.DataFrame(rows, columns=["setting", "configuration"])


def main(argv):
    run = CM.run_arg(argv); out = CM.out_dir(run, SECTION)
    cfg = C.load(CM.run_config(run)["_path"])
    outs = [CM.write_csv(table2(cfg), out, "table2_corpus"), CM.write_csv(settings(CM.run_config(run)), out, "table4_model_settings")]
    CM.write_provenance(run, out, "analysis/sec5_dataset_and_settings.py",
                        ["derived data files (manifests/DATA_MANIFEST.json)", "config_resolved.json"], outs,
                        {"temporal_episodes": "attack windows of a type separated by more than 20 windows start a new episode"})
    print(pd.read_csv(outs[0]).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv)
