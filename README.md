# GRIV: Gateway-Based Report Integrity Verification

Code for the experiments of the paper *GRIV: Gateway-Based Report Integrity Verification against Report Concealment Attacks in Distributed IoT IDS* (Computers, Materials & Continua).

GRIV flags a node report as suspected when the node-report IDS returns a benign decision while the paired, independently sampled gateway observation indicates an attack:

```
I(report) = [node-report IDS decision = benign] AND [gateway observation available] AND [p_G > tau_G]
```

`p_G` is the calibrated attack probability of the gateway observation and `tau_G` is the 99th percentile of `p_G` on covered benign validation reports. The node-report IDS prediction itself is not changed.

## Quick start

All commands are run from the repository root. Steps 3 and 4 reproduce every reported number. Step 5 is a short check of the experiment engine (about two minutes); the analysis scripts of step 4 need the full grid of `configs/paper.yaml`.

**Linux / macOS (bash)**

```bash
# 1. Environment (Python 3.10)
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2. Data: put Network_dataset_1.csv ... Network_dataset_23.csv of TON_IoT in raw/, then build the derived files in data/
python data_prep/build_derived_data.py --raw raw --out data
cp configs/local_paths.example.yaml configs/local_paths.yaml

# 3. Experiments -> per-report shards in results/paper/
python -m griv.runner --config configs/paper.yaml --run-id paper \
    --experiments complete_concealment,concealment_strength,adaptive,sampling_rate,no_platt,mlp
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    python -m griv.runner --config configs/paper.yaml --run-id paper --experiments cost

# 4. Tables and figure data of the paper and the supplementary material -> results/paper/paper/
python analysis/run_all.py results/paper

# 5. (optional) quick end-to-end check: one run, two RD folds -> results/_quick/quick/
python -m griv.runner --config configs/quick.yaml --run-id quick \
    --experiments complete_concealment,concealment_strength,adaptive,sampling_rate,no_platt,mlp
python -m griv.runner --config configs/quick.yaml --run-id quick --experiments cost
```

**Windows (PowerShell)**

```powershell
# 1. Environment (Python 3.10)
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2. Data: put Network_dataset_1.csv ... Network_dataset_23.csv of TON_IoT in raw\, then build the derived files in data\
python data_prep/build_derived_data.py --raw raw --out data
Copy-Item configs/local_paths.example.yaml configs/local_paths.yaml

# 3. Experiments -> per-report shards in results\paper\
python -m griv.runner --config configs/paper.yaml --run-id paper --experiments complete_concealment,concealment_strength,adaptive,sampling_rate,no_platt,mlp
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"; $env:OPENBLAS_NUM_THREADS = "1"
python -m griv.runner --config configs/paper.yaml --run-id paper --experiments cost
Remove-Item Env:OMP_NUM_THREADS, Env:MKL_NUM_THREADS, Env:OPENBLAS_NUM_THREADS

# 4. Tables and figure data of the paper and the supplementary material -> results\paper\paper\
python analysis/run_all.py results/paper

# 5. (optional) quick end-to-end check: one run, two RD folds -> results\_quick\quick\
python -m griv.runner --config configs/quick.yaml --run-id quick --experiments complete_concealment,concealment_strength,adaptive,sampling_rate,no_platt,mlp
python -m griv.runner --config configs/quick.yaml --run-id quick --experiments cost
```

The cost experiment must run on its own (single-threaded timing). Each analysis script can also be run alone, e.g. `python analysis/sec6_4_adaptive.py results/paper`. `python analysis/run_all.py results/paper --skip-s9` skips Supplementary Table S9, the only analysis step that retrains classifiers.

## Requirements

Python 3.10 and the packages in `requirements.txt`, the versions used to produce and verify the results (Python 3.10.20). The MLP experiment needs scikit-learn >= 1.7 (sample weights in `MLPClassifier.fit`).

## Data

The experiments use the TON_IoT network dataset (`Network_dataset_1.csv` ... `Network_dataset_23.csv`), available at https://research.unsw.edu.au/projects/toniot-datasets. The raw data are not redistributed here; follow the dataset's license and citation terms.

`data_prep/build_derived_data.py` builds the five files read by the engine:

| File | Content |
|---|---|
| `o_it_R2.parquet` | Node-window reports of all devices: 18 features per report (per direction: mean duration, seven byte and packet sums, flow count), node label and global label |
| `flow_store_reporters.parquet` | Flow records with a monitored reporter as source or destination; `raw_flow_id` is the running index over all cohort flow records in file order |
| `window_dominant.parquet` | Dominant attack type per 60-s window: the type with the most attack-labeled flow records in the window over the whole dataset (ties by type name; `none` without attack flows) |
| `meta.json` | Monitored reporters, feature order and build summary |
| `CACHE_MANIFEST.json` | File list |

- Cohort: normal traffic and the seven attack types (backdoor, DDoS, DoS, injection, password, scanning, XSS); MITM and ransomware flow records are removed before windowing.
- Monitored reporters: 192.168.x.x devices with at least 10 reports that appear as a flow source, excluding the attacker hosts (192.168.1.28, 192.168.1.30–39), the gateway (192.168.1.1) and the broadcast address (192.168.1.255). This gives 19 reporters.
- `manifests/DATA_MANIFEST.json` lists the SHA-256 of every file. `meta.json` also records the build time, so it is checked by the SHA-256 of its content without `elapsed_seconds`. The builder prints the comparison, and every experiment run checks the files and the Table 2 counts before starting.

`configs/local_paths.yaml` tells the engine where the files are. The example points to `data/`; relative paths are resolved against the repository root.

## Repository layout

| Path | Content | Paper |
|---|---|---|
| `data_prep/build_derived_data.py` | Derived data files from the TON_IoT CSV files | Section 5.1 |
| `griv/features.py` | The 18 report features | Section 4.1, Table S1 |
| `griv/sampling.py` | Gateway flow sampling hash | Section 5.4, Algorithm S1 |
| `griv/splits.py` | CW and RD partitions | Section 5.2 |
| `griv/models.py` | Node-report IDS (RF / MLP), Platt calibration, sampled gateway observations, benign reference | Sections 5.3, 5.4 |
| `griv/methods.py`, `griv/residual.py` | GRIV and the comparison methods | Sections 4.3, 5.5 |
| `griv/compromise.py` | Compromised reporters and targeted reports | Sections 3.2, 5.3 |
| `griv/attacks/` | Prototype-directed concealment (Eq. (1)), adaptive concealment, integer and relation projection | Sections 3.3, 5.3 |
| `griv/thresholds.py`, `griv/metrics.py`, `griv/aggregate.py`, `griv/bootstrap.py` | Thresholds, metrics, aggregation, cluster bootstrap | Section 5.6 |
| `griv/cost.py` | Processing latency and peak memory | Section 6.2 |
| `griv/runner.py` | Experiment runner | – |
| `configs/paper.yaml` | Settings of the reported experiments | Sections 5.2–5.6 |
| `configs/quick.yaml` | Same definitions on one run and two RD folds | – |
| `analysis/` | One script per paper section | Sections 5–6 |
| `supplementary/` | Supplementary tables and figure data | Supplementary material |
| `manifests/DATA_MANIFEST.json` | SHA-256 of the derived data files | – |

### Methods (Section 5.5)

| Code ID | Paper name | Role |
|---|---|---|
| `GRIV` | GRIV | Proposed method |
| `FeatureResidual` | Feature Residual | Cross-view baseline |
| `ProbabilityDifference` | Probability Difference | Cross-view baseline |
| `JSDivergence` | JS Divergence | Cross-view baseline |
| `LogOddsDifference` | Log-odds Difference | Cross-view baseline |
| `MahalanobisDistance` | Mahalanobis Distance | Cross-view baseline |
| `NodeReportIDS` | Node-report IDS only | Single-view reference |
| `GatewayIDS` | Gateway IDS only | Single-view reference |
| `MultiplicativeScore` | Multiplicative score `p_G (1 - p_N)` | Decision-rule ablation (Section 6.5) |

### Experiments (`configs/paper.yaml`)

| Experiment | Setting | Used in |
|---|---|---|
| `complete_concealment` | Concealment with alpha = 1; CW (20 runs) and RD (17 folds, 10 runs) | Sections 6.1, 6.3, 6.5; Supplementary |
| `concealment_strength` | Concealment with alpha = 0.1–0.9, 0.95, 0.99, 1.0; CW | Sections 6.1, 6.3 |
| `adaptive` | Adaptive concealment, search step 0.01 (CW, RD) and 0.02, 0.05 (CW) | Sections 6.4, 6.5; Supplementary |
| `sampling_rate` | Complete concealment for s = 0.1–1.0; CW | Section 6.2 |
| `cost` | Processing latency and peak memory of the GRIV path for s = 0.1–1.0; one run | Section 6.2; Supplementary Table S9 |
| `no_platt` | Platt calibration removed from both views; complete and adaptive concealment; CW | Section 6.5 (Table 6) |
| `mlp` | MLP replaces the RF in both views; complete and adaptive concealment; CW | Section 6.5 (Table 6) |

### Analysis scripts

Every script writes CSV files (tables and the data plotted in each figure) and a provenance record to `results/<run_id>/paper/<section>/` or `results/<run_id>/paper/supplementary/`.

| Script | Output | Paper |
|---|---|---|
| `analysis/sec5_dataset_and_settings.py` | Table 2 (`table2_corpus`); Table 4 and Supplementary Table S3 (`table4_model_settings`) | Sections 5.1, 5.4 |
| `analysis/sec6_1_concealment_impact.py` | Fig. 3 data (alpha and beta curves of Node-report IDS, Gateway IDS, Node-report IDS + GRIV); benign Operational FPR | Section 6.1 |
| `analysis/sec6_2_observation_budget.py` | Fig. 4 data (Operational TPR and availability vs s; latency; peak memory) | Section 6.2 |
| `analysis/sec6_3_baselines.py` | Fig. 5 data (`fig5_*`); complete-concealment TPR and benign integrity FPR | Section 6.3 |
| `analysis/sec6_4_adaptive.py` | Fig. 6 data (adaptive Operational TPR with 95 % cluster-bootstrap CIs, search-step sensitivity; `fig6_*`) | Section 6.4 |
| `analysis/sec6_5_reporter_ablation.py` | Fig. 7 data (`fig7_*`); Table 6 (`table6_ablation_generalization`) | Section 6.5 |
| `supplementary/table_s2_partitions.py` | Table S2 (a)–(c): CW partitions, exclusion ledger; RD fold design | Section 5.2 |
| `supplementary/table_s2_rd_folds.py` | Table S2 (d): fold-level RD results | Section 6.5 |
| `supplementary/table_s4_model_quality.py` | Table S4 (AUROC, Brier score, ECE, feature-wise KS distances) and Fig. S1 data (reliability curves) | Section 5.4 |
| `supplementary/table_s5_alert_workload.py` | Table S5: benign alert burden with 95 % CIs, confusion counts, precision under assumed prevalence, workload, alerts per device-day | Sections 6.3, 6.5 |
| `supplementary/fig_s2_attack_types.py` | Fig. S2 data: Operational TPR, GRIV Covered TPR and coverage per attack type | Section 6.3 |
| `supplementary/table_s6_attack_types.py` | Table S6: results by attack type; training reports of each type per RD fold | Section 6.5 |
| `supplementary/table_s7_decision_rule.py` | Table S7 and Fig. S3 data: decision-rule ablation (GRIV, multiplicative score, Probability Difference) | Sections 6.4–6.5 |
| `supplementary/table_s8_matched_fpr.py` | Table S8: post hoc TEST-retuned thresholds at nominal integrity-FPR targets | Sections 6–7 |
| `supplementary/table_s9_sampling_rate.py` | Table S9: validation check of the reference sampling rate; detailed cost measurements | Section 6.2 |

### Outputs and checks

- Shards: `results/<run_id>/shards/<experiment>/<split>/<fold>/seed=<seed>/s=<s>/<variant>/<classifier>__<attack>__<param>.parquet`, one row per TEST report with honest and submitted decisions of every method.
- Completed shards are never recomputed; a run directory created with different code, configuration or data is refused.
- A run stops and writes `INVALID.json` if a validity check fails, for example:
  - `O_GRIV` differs from `NodeReportIDS OR GatewayIDS`;
  - a submitted report violates the integer and relation constraints;
  - gateway availability changes under node-side manipulation;
  - targeted-report counts differ from the pre-attack counts.

## Notes

- Seeds: CW runs 420–439, RD runs 420–429. Gateway sampling is a deterministic hash of the flow id, the seed and the stream name, so sampled flow sets are nested across s and shared by all methods.
- The adaptive attacker uses an evaluation oracle (gateway availability, realized gateway probability and, for Feature Residual, the gateway feature vector), as described in Section 3.2.
- Cost measurements are offline batch-replay measurements on one run; they exclude traffic capture, flow export, transport and storage.
