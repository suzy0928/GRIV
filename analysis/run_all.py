"""Build every table and the data of every figure (CSV) of the paper and the supplementary material from one run directory.

Usage: python analysis/run_all.py results/<run_id> [--skip-s9]
  --skip-s9 skips the Supplementary Table S9 validation check, which retrains the node classifier once per run and rebuilds the
  gateway observations, gateway calibration and thresholds for every sampling rate.
  table_s4_model_quality.py rebuilds the CW gateway observations of every run (no model training).
Outputs: results/<run_id>/paper/<section>/ and results/<run_id>/paper/supplementary/
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "analysis/sec5_dataset_and_settings.py",       # Section 5: Table 2, Table 4 / Supplementary Table S3
    "analysis/sec6_1_concealment_impact.py",       # Section 6.1: Fig. 3
    "analysis/sec6_2_observation_budget.py",       # Section 6.2: Fig. 4
    "analysis/sec6_3_baselines.py",                # Section 6.3: Fig. 5, benign integrity FPR
    "analysis/sec6_4_adaptive.py",                 # Section 6.4: Fig. 6
    "analysis/sec6_5_reporter_ablation.py",        # Section 6.5: Fig. 7, Table 6
    "supplementary/table_s2_partitions.py",        # Supplementary Table S2 (a)-(c), RD fold design (Section 5.2)
    "supplementary/table_s2_rd_folds.py",          # Supplementary Table S2 (d): fold-level RD results (Section 6.5)
    "supplementary/table_s4_model_quality.py",     # Supplementary Table S4, Fig. S1: classifier quality per view (Section 5.4)
    "supplementary/table_s5_alert_workload.py",    # Supplementary Table S5: benign alerts, confusion counts, device-day rates
    "supplementary/fig_s2_attack_types.py",        # Supplementary Fig. S2: Operational TPR and coverage by attack type (Section 6.3)
    "supplementary/table_s6_attack_types.py",      # Supplementary Table S6: results by attack type and RD training presence
    "supplementary/table_s7_decision_rule.py",     # Supplementary Table S7, Fig. S3: decision-rule ablation
    "supplementary/table_s8_matched_fpr.py",       # Supplementary Table S8: matched nominal FPR targets
    "supplementary/table_s9_sampling_rate.py",     # Supplementary Table S9: sampling-rate check and cost details (Section 6.2)
]


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    run = str(Path(argv[1]).resolve())
    skip_s9 = "--skip-s9" in argv
    for step in STEPS:
        if skip_s9 and step.endswith("table_s9_sampling_rate.py"):
            continue
        print(f"=== {step}", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / step), run], cwd=ROOT)
        if r.returncode != 0:
            raise SystemExit(f"{step} failed with exit code {r.returncode}")
    print(f"done -> {run}/paper")


if __name__ == "__main__":
    main(sys.argv)
