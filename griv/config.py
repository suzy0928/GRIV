"""Configuration loading and validation.

Two profiles are accepted:
- ``paper``: the exact settings of the reported experiments (seeds, grids, folds, bootstrap and cost settings).
- ``quick``: the same definitions on a subset of seeds, grids and RD folds, for a fast end-to-end check.

Every value that defines a method, a threshold, a classifier or an attack must equal the paper value in both
profiles; the loader refuses a config that silently overrides one of them.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]

CW_SEEDS = list(range(420, 440))          # 20 runs (CW)
RD_SEEDS = list(range(420, 430))          # 10 runs (RD) = first 10 CW seeds
TYPES = ["backdoor", "ddos", "dos", "injection", "password", "scanning", "xss"]
METHODS = ["GRIV", "FeatureResidual", "ProbabilityDifference", "JSDivergence", "LogOddsDifference", "MahalanobisDistance",
           "MultiplicativeScore"]
REFERENCE_ROWS = ["NodeReportIDS", "GatewayIDS", "NodeOrGatewayIDS"]
BETA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
S_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CORPUS = {"raw_flows": 19307096, "reporters": 19, "reports": 42390, "benign": 24968, "attack": 17422,
          "by_type": {"backdoor": 1560, "ddos": 4252, "dos": 3108, "injection": 705, "password": 4062,
                      "scanning": 2225, "xss": 1510}}
_CONC1 = {"kind": "concealment", "alpha": [1.0]}
PAPER_EXPERIMENTS = {
    "complete_concealment": {"splits": ["CW", "RD"], "attacks": [_CONC1]},
    "concealment_strength": {"splits": ["CW"], "attacks": [{"kind": "concealment", "alpha": "grid"}]},
    "adaptive": {"splits": ["CW", "RD"], "attacks": [{"kind": "adaptive", "step": [0.01]},
                                                     {"kind": "adaptive", "step": [0.02, 0.05], "splits": ["CW"]}]},
    "sampling_rate": {"splits": ["CW"], "s": "grid", "attacks": [_CONC1]},
    "cost": {"splits": ["CW"], "cost": True, "s": "grid"},
    "no_platt": {"splits": ["CW"], "variant": "no_platt", "attacks": [_CONC1, {"kind": "adaptive", "step": [0.01]}]},
    "mlp": {"splits": ["CW"], "classifier": "mlp", "attacks": [_CONC1, {"kind": "adaptive", "step": [0.01]}]},
}


class ConfigError(ValueError):
    pass


def _req(c: dict, path: str):
    cur: Any = c
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise ConfigError(f"missing required config key: {path}")
        cur = cur[k]
    return cur


def _eq(c: dict, path: str, expected):
    v = _req(c, path)
    if v != expected:
        raise ConfigError(f"config {path} = {v!r} != paper value {expected!r}")


def _subset(c: dict, path: str, full: list):
    v = _req(c, path)
    if not isinstance(v, list) or not v or any(x not in full for x in v):
        raise ConfigError(f"config {path} = {v!r} must be a non-empty subset of {full!r}")


def validate(c: dict) -> dict:
    prof = _req(c, "profile")
    if prof not in ("paper", "quick"):
        raise ConfigError(f"unknown profile {prof!r} (paper | quick)")
    _eq(c, "data.corpus_assert", CORPUS)
    # ---- partitions ----
    _eq(c, "split_common", {"K": 20, "K_val": 20, "val_cal_frac": 0.5, "episode_gap_min": 20, "val_purge_side": "alternate"})
    _eq(c, "splits.CW", {"test_frac": 0.30, "val_frac": 0.175, "episode_align": True, "align_tolerance_pp": 10,
                         "xpurge_check": {"max_test_attack_window_loss": 0.10, "all_types_all_partitions": True}})
    _eq(c, "splits.RD", {"base": "CW"})
    # ---- models, threshold, gateway ----
    _eq(c, "threshold", {"quantile": 0.99, "method": "linear", "population": "VAL_THR_benign_covered", "alert": "strict_gt"})
    _eq(c, "classifier", {"node_ids": "predict", "rf": {"n_estimators": 100, "class_weight": "balanced"},
                          "calibration": "platt_on_VAL_CAL"})
    _eq(c, "mlp", {"hidden": [64, 32], "activation": "relu", "alpha": 1.0e-4, "max_iter": 300, "early_stopping": True,
                   "n_iter_no_change": 15, "sample_weight": "balanced", "sklearn_min_version": "1.7"})
    _eq(c, "gateway.s_main", 0.3); _eq(c, "gateway.stream", "gw")
    # ---- compromise, methods, attacks ----
    _eq(c, "methods", METHODS)
    _eq(c, "method_params", {"prob_clip_eps": 1.0e-6, "mahalanobis": {"cov_reg": 1.0e-6}, "std_floor": 1.0e-8,
                             "min_reporter_reports": 10})
    _eq(c, "attacks.mu_ben_population", "calibration_windows")
    _eq(c, "attacks.modification_tolerance", 1.0e-9)
    _eq(c, "attacks.concealment.alpha_main", 1.0)
    _eq(c, "attacks.adaptive.alpha_step", 0.01)
    _eq(c, "bootstrap.rng_seed", 20261002)
    _eq(c, "cost.threads", 1)
    for e, spec in _req(c, "experiments").items():
        if e not in PAPER_EXPERIMENTS:
            raise ConfigError(f"unknown experiment {e!r}; known: {sorted(PAPER_EXPERIMENTS)}")
    if prof == "paper":
        _eq(c, "seeds.cw", CW_SEEDS); _eq(c, "seeds.rd", RD_SEEDS)
        _eq(c, "compromise.beta_grid", BETA_GRID)
        _eq(c, "attacks.concealment.alpha_grid", ALPHA_GRID)
        _eq(c, "attacks.adaptive.step_sensitivity", [0.02, 0.05])
        _eq(c, "s_grid", S_GRID)
        _eq(c, "rd_folds", "all")
        _eq(c, "bootstrap.B", 10000)
        _eq(c, "cost.warmup", 3); _eq(c, "cost.repeats", 20)
        _eq(c, "experiments", PAPER_EXPERIMENTS)
    else:
        _subset(c, "seeds.cw", CW_SEEDS); _subset(c, "seeds.rd", RD_SEEDS)
        _subset(c, "compromise.beta_grid", BETA_GRID)
        _subset(c, "attacks.concealment.alpha_grid", ALPHA_GRID)
        _subset(c, "s_grid", S_GRID)
        if 1.0 not in c["compromise"]["beta_grid"]:
            raise ConfigError("beta grid must contain 1.0")
        rf = _req(c, "rd_folds")
        if not (rf == "all" or (isinstance(rf, int) and rf > 0)):
            raise ConfigError("rd_folds must be 'all' or a positive integer")
    return c


def cw_seeds(c: dict) -> list[int]:
    return list(c["seeds"]["cw"])


def rd_seeds(c: dict) -> list[int]:
    return list(c["seeds"]["rd"])


def config_hash(c: dict) -> str:
    return hashlib.sha256(json.dumps(c, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    with open(p, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c = validate(copy.deepcopy(c))
    c["_path"] = str(p.relative_to(REPO)).replace("\\", "/")
    c["_hash"] = config_hash({k: v for k, v in c.items() if not k.startswith("_")})
    return c


def output_root(c: dict) -> Path:
    return REPO / c.get("output_root", "results")


def local_paths(c: dict) -> dict:
    """Paths of the derived data files; relative paths are resolved against the repository root."""
    with open(REPO / c["data"]["local_paths"], encoding="utf-8") as f:
        lp = yaml.safe_load(f)
    lp["files"] = {k: str(Path(v) if Path(v).is_absolute() else REPO / v) for k, v in lp["files"].items()}
    return lp
