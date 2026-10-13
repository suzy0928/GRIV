"""Shared helpers for the paper analyses: shard loading, aggregation, display names and provenance.

Every number in a table or in the data of a figure is computed here from the per-report shards of one run directory
(results/<run_id>/); nothing is copied by hand. Outputs go to results/<run_id>/paper/<section>/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from griv import aggregate as AG  # noqa: E402
from griv import bootstrap as BS  # noqa: E402
from griv.errors import ValidityError  # noqa: E402

# ------------------------------------------------------------------------------------------------ names
INTEGRITY_ROWS = ["GRIV", "FeatureResidual", "ProbabilityDifference", "JSDivergence", "LogOddsDifference",
                  "MahalanobisDistance"]
ROWS = INTEGRITY_ROWS + ["A:NodeReportIDS", "A:GatewayIDS"]
DISPLAY = {"GRIV": "GRIV", "FeatureResidual": "Feature Residual", "ProbabilityDifference": "Probability Difference",
           "JSDivergence": "JS Divergence", "LogOddsDifference": "Log-odds Difference",
           "MahalanobisDistance": "Mahalanobis Distance", "A:NodeReportIDS": "Node-report IDS only",
           "A:GatewayIDS": "Gateway IDS only", "MultiplicativeScore": "Multiplicative score"}
CONCEALMENT = "concealment"
ADAPTIVE = "adaptive"


# ------------------------------------------------------------------------------------------------ run access
def run_config(run: Path) -> dict:
    return json.loads((Path(run) / "config_resolved.json").read_text(encoding="utf-8"))


def shards(run: Path, **kw) -> pd.DataFrame:
    return AG.load_shards(run, **kw)


def table_files(run: Path, name: str) -> pd.DataFrame:
    fs = sorted((Path(run) / "tables").rglob(name))
    return pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True) if fs else pd.DataFrame()


def cost_table(run: Path) -> pd.DataFrame:
    fs = sorted((Path(run) / "shards").rglob("cost.parquet"))
    return pd.concat([pd.read_parquet(p) for p in fs], ignore_index=True) if fs else pd.DataFrame()


def expected_seeds(cfg: dict, split: str) -> list[int]:
    return list(cfg["seeds"]["rd"] if split == "RD" else cfg["seeds"]["cw"])


class Result:
    """Shards of one setting with their per-seed cells (A: attack cells, U: benign metrics)."""

    def __init__(self, f, A, U, betas, split):
        self.f, self.A, self.U, self.betas, self.split = f, A, U, betas, split

    def macro(self, metric: str, rows=None, stratum="all") -> pd.DataFrame:
        m = AG.macro(self.A, metric, self.betas)
        m = m[m["stratum"] == stratum]
        return m[m["row"].isin(rows)] if rows is not None else m

    def per_type(self, metric: str, rows=None, stratum="all") -> pd.DataFrame:
        m = AG.per_type(self.A, metric, self.betas)
        m = m[m["stratum"] == stratum]
        return m[m["row"].isin(rows)] if rows is not None else m

    def benign(self) -> pd.DataFrame:
        return AG.untampered_summary(self.U, 1.0)


def load_setting(run: Path, experiment: str, split: str, attack: str, param: str, *, variant="base", classifier="rf",
                 s=None, adaptive=False) -> Result:
    """Load the shards of one setting, check runs and targeted-report counts, and compute the per-seed cells."""
    run = Path(run); cfg = run_config(run)
    s = cfg["gateway"]["s_main"] if s is None else s
    f = shards(run, experiment=experiment, split=split, attack=attack, param=param, variant=variant, classifier=classifier, s=s)
    if f.empty:
        raise FileNotFoundError(f"no shards for {experiment}/{split}/{attack}/{param}/{variant}/{classifier}/s={s} in {run}")
    BS.check_seeds(f, expected_seeds(cfg, split), f"{experiment}/{split}/{param}/{variant}/{classifier}")
    betas = [1.0] if split == "RD" else list(cfg["compromise"]["beta_grid"])
    A, U = AG.cells(f, betas, adaptive, pooled_folds=(split == "RD"))
    sna = table_files(run, "sna.parquet")
    if len(sna):
        sna = sna.drop_duplicates(subset=["split", "fold", "seed", "beta", "attack_type"])
        sna = sna[(sna["split"] == split) & sna["seed"].isin(A["seed"].unique()) & sna["beta"].isin(betas)]
        if split == "RD":
            sna = sna[sna["fold"].isin(set(f["fold"]))]
        bad = AG.check_against_sna(A, sna)
        if len(bad):
            raise ValidityError(f"targeted-report counts differ from sna.parquet ({len(bad)} cells) for {experiment}/{split}/{param}")
    return Result(f, A, U, betas, split)


def value(df: pd.DataFrame, row: str, col: str = "value") -> float:
    x = df[df["row"] == row]
    return float(x[col].iloc[0]) if len(x) else float("nan")


def adaptive_bootstrap(res: Result, rows, B: int, rng_seed: int) -> pd.DataFrame:
    """95 % cluster-bootstrap CIs of adaptive Operational TPR (= 1 - simultaneous evasion), paired across rows."""
    f = res.f.copy()
    cols = {}
    for r in rows:
        if r == "A:GatewayIDS":
            f["_det_" + r] = f["A_GatewayIDS"].astype(float)
        else:
            key = r[2:] if r.startswith("A:") else r
            f["_det_" + r] = (~f[f"evaded_{key}"].astype(bool)).astype(float)
        cols[r] = "_det_" + r
    prep = BS.prepare(f, res.betas, cols, "reporter_block60", adaptive=True)
    ci = BS.level_cis(prep, list(rows), B=B, rng_seed=rng_seed)
    return pd.DataFrame([dict(row=r, **v) for r, v in ci.items()])


def rd_per_fold(fx: Result, ad: Result) -> pd.DataFrame:
    """Fold-level Operational TPR (fixed, adaptive) and benign integrity FPR, mean over runs (Supplementary table)."""
    rows = []
    for name, res, adaptive, metric in (("fixed", fx, False, "operational_TPR"), ("adaptive", ad, True, "simultaneous_evasion")):
        for fold, g in res.f.groupby("fold"):
            A, U = AG.cells(g, [1.0], adaptive, pooled_folds=True)
            m = AG.macro(A, metric, [1.0])
            if m.empty or "stratum" not in m.columns:          # fold without attack-bearing TEST reports
                m = pd.DataFrame(columns=["row", "stratum", "value", "level"])
            m = m[m["stratum"] == "all"]
            for r in ROWS:
                v = value(m, r)
                rows.append(dict(fold=fold, row=r, quantity=f"{name}_operational_TPR_pct",
                                 value=100 * (1 - v if metric == "simultaneous_evasion" else v), level=m.loc[m["row"] == r, "level"].iloc[0]
                                 if (m["row"] == r).any() else ""))
            if name == "fixed":
                u = AG.untampered_summary(U, 1.0)
                for r in INTEGRITY_ROWS:
                    rows.append(dict(fold=fold, row=r, quantity="benign_integrity_FPR_pct", value=100 * value(u, r, "benign_integrity_FPR"),
                                     level=f"n_benign={int(value(u, r, 'n_benign'))}"))
    P = pd.DataFrame(rows)
    P["display"] = P["row"].map(DISPLAY)
    return P


# ------------------------------------------------------------------------------------------------ outputs
def out_dir(run: Path, section: str) -> Path:
    p = Path(run) / "paper" / section
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(df: pd.DataFrame, out: Path, name: str) -> Path:
    p = out / f"{name}.csv"
    df.to_csv(p, index=False)
    return p


def write_provenance(run: Path, out: Path, script: str, sources: list, outputs: list, notes: dict | None = None):
    """PROVENANCE.json in a section folder; in the shared supplementary folder, PROVENANCE_<script name>.json."""
    run = Path(run)
    prov = json.loads((run / "provenance.json").read_text(encoding="utf-8")) if (run / "provenance.json").exists() else {}
    sp = ROOT / script
    rec = dict(run=str(run), run_id=prov.get("run_id"), code_hash=prov.get("code_hash"), config_hash=prov.get("config_hash"),
               data_hash=prov.get("data_hash"), git_commit=prov.get("git_commit"), script=script,
               script_sha256=hashlib.sha256(sp.read_bytes()).hexdigest() if sp.exists() else None,
               sources=sources, outputs=[Path(o).name for o in outputs], notes=notes or {})
    name = f"PROVENANCE_{Path(script).stem}.json" if Path(out).name == "supplementary" else "PROVENANCE.json"
    (out / name).write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")


def pct(x):
    return 100.0 * np.asarray(x, float)


def run_arg(argv) -> Path:
    if len(argv) < 2:
        raise SystemExit(f"usage: python {Path(argv[0]).name} results/<run_id>")
    return Path(argv[1]).resolve()
