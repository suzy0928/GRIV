"""Models (Section 5.4): node-report IDS classifier (RF or MLP) trained on TRAIN, view-specific Platt calibration on VAL_CAL,
sampled gateway observations, Feature Residual statistics and the reporter-specific benign reference (Section 5.3).

  - RF: 100 trees, balanced class weights, random_state = run seed
  - Platt scaling: logistic regression (C = 1e6) on the raw classifier probability
  - honest node report = the device-window report built from all flow records of the device
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .data import Data
from .features import FEATS
from .features import FLOW_COLUMNS, SUM_COLUMNS
from .residual import feature_residual_stats
from .sampling import keep_uniform
from .errors import ValidityError
from .splits import PARTS, Split

KEYS = ["device_id", "window"]
VARIANTS = ("base", "no_platt")


# ------------------------------------------------------------------------------------------------ calibrators
class Platt:
    """Platt scaling: logistic regression on the raw classifier probability."""
    def __init__(self):
        self.lr = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, raw, y):
        x = np.asarray(raw, float).reshape(-1, 1); y = np.asarray(y, int)
        if len(np.unique(y)) < 2:
            self.const = float(y.mean()); self.lr = None
        else:
            self.lr.fit(x, y); self.const = None
        return self

    def transform(self, raw):
        x = np.asarray(raw, float).reshape(-1, 1)
        if self.lr is None:
            return np.full(len(x), self.const)
        return self.lr.predict_proba(x)[:, 1]


class IdentityCal:
    """Calibration ablation: raw classifier probabilities."""
    def fit(self, raw, y):
        return self

    def transform(self, raw):
        return np.asarray(raw, float)


def require_two_per_class(y, what: str, where: str) -> dict:
    """Every fitted model needs at least two samples per class."""
    y = np.asarray(y).astype(int)
    cnt = {0: int((y == 0).sum()), 1: int((y == 1).sum())}
    if min(cnt.values()) < 2:
        raise ValidityError(f"{what} at {where}: class counts {cnt} (need >= 2 per class)")
    return cnt


def calibration_grid_monotone(cal) -> tuple[bool, float, float]:
    """Calibration must be non-decreasing on {0, 0.01, ..., 1} with calibrate(0.9) > calibrate(0.1)."""
    g = np.round(np.arange(0, 101) / 100.0, 2)
    v = cal.transform(g)
    return bool(np.all(np.diff(v) >= 0)), float(cal.transform(np.array([0.9]))[0]), float(cal.transform(np.array([0.1]))[0])


def require_monotone(res, where: str):
    mono, hi, lo = res
    if not mono or not hi > lo:
        raise ValidityError(f"{where}: calibration mapping not monotone increasing (mono={mono}, c(.9)>c(.1)={hi > lo})")
    return res


def _pos(model):
    c = list(getattr(model, "classes_", [0, 1]))
    return c.index(1) if 1 in c else len(c) - 1


# ------------------------------------------------------------------------------------------------ bundles
@dataclass
class NodeBundle:
    seed: int
    split: Split
    classifier: str
    model: object
    cal_N: object
    H: dict                      # partition -> honest node reports (KEYS + FEATS + node_label + global_label)
    mu_table: dict               # reporter -> benign reference vector
    mu_global: np.ndarray        # pooled fallback
    mu_meta: dict
    fit_log: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def raw_p(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self.model.predict_proba(np.asarray(X, np.float64))[:, _pos(self.model)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Node-report IDS decision = the classifier's default class prediction (no tuned threshold). 1 = attack."""
        if len(X) == 0:
            return np.zeros(0, dtype=int)
        return np.asarray(self.model.predict(np.asarray(X, np.float64))).astype(int)

    def p_N(self, X: np.ndarray) -> np.ndarray:
        return self.cal_N.transform(self.raw_p(X)) if len(X) else np.zeros(0)


@dataclass
class GatewayBundle:
    s: float
    G: dict                     # partition -> gateway observations (KEYS + FEATS + gateway_available + node_label)
    f_G: object                 # classifier applied to gateway observations (the node classifier)
    cal_G: object
    gstats: object              # Feature Residual statistics
    variant: str
    meta: dict = field(default_factory=dict)

    def raw_p(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self.f_G.predict_proba(np.asarray(X, np.float64))[:, _pos(self.f_G)]

    def p_G(self, X: np.ndarray) -> np.ndarray:
        return self.cal_G.transform(self.raw_p(X)) if len(X) else np.zeros(0)


# ------------------------------------------------------------------------------------------------ benign reference
def mu_ben_table(d: Data, windows, reporters, population: str, min_win: int):
    """Benign reference = feature-wise median of a reporter's TRAIN reports from network-wide attack-free windows;
    reporters with fewer than ``min_win`` such reports use the pooled median of the same population."""
    if population != "calibration_windows":
        raise ValueError(population)
    o = d.o
    r = o[o["window"].isin(set(map(int, windows))) & o["device_id"].isin(set(reporters))]
    cal = set(int(w) for w in windows if int(d.gl.get(int(w), 0)) == 0)
    r = r[r["window"].isin(cal)]
    X = r[FEATS].to_numpy(np.float64)
    glob = np.median(X, axis=0)
    tab, fb = {}, []
    for dev, g in r.groupby("device_id"):
        if len(g) >= min_win:
            tab[str(dev)] = np.median(g[FEATS].to_numpy(np.float64), axis=0)
        else:
            fb.append(str(dev))
    return tab, glob, {"population": population, "n_reports": len(r), "fallback_devices": sorted(fb),
                       "reporters": sorted(map(str, reporters))}


# ------------------------------------------------------------------------------------------------ classifiers
def _fit_classifier(cfg, classifier, seed, X, y):
    if classifier == "rf":
        rc = cfg["classifier"]["rf"]
        m = RandomForestClassifier(n_estimators=int(rc["n_estimators"]), class_weight=rc["class_weight"],
                                   random_state=int(seed), n_jobs=-1)
        m.fit(np.asarray(X, np.float64), np.asarray(y).astype(int))
        # Training is parallel (the trees do not depend on n_jobs); inference runs single-threaded so that the per-tree
        # probability sum is accumulated in tree order (bit-for-bit reproducible).
        m.n_jobs = 1
        return m
    if classifier == "mlp":
        import sklearn
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.utils.class_weight import compute_sample_weight
        mc = cfg["mlp"]
        if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) < tuple(int(x) for x in mc["sklearn_min_version"].split(".")):
            raise RuntimeError("MLPClassifier.fit(sample_weight=) requires scikit-learn >= 1.7")
        m = Pipeline([("scaler", StandardScaler()),
                      ("mlpclassifier", MLPClassifier(hidden_layer_sizes=tuple(mc["hidden"]), activation=mc["activation"],
                                                      alpha=mc["alpha"], max_iter=int(mc["max_iter"]),
                                                      early_stopping=bool(mc["early_stopping"]),
                                                      n_iter_no_change=int(mc["n_iter_no_change"]), random_state=seed))])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(np.asarray(X, np.float64), y, mlpclassifier__sample_weight=compute_sample_weight("balanced", y))
        return m
    raise ValueError(classifier)


# ------------------------------------------------------------------------------------------------ node bundle
def honest_reports(d: Data, windows, reporters) -> pd.DataFrame:
    """Honest node reports of the given reporters in the given windows, sorted by (device_id, window)."""
    sel = d.o_all["window"].isin(set(map(int, windows))) & d.o_all["device_id"].astype(str).isin(set(map(str, reporters)))
    if len(windows) == 0 or len(reporters) == 0 or not bool(sel.any()):
        return pd.DataFrame(columns=KEYS + FEATS + ["node_label", "global_label"])
    r = d.o_all[sel].copy()
    r["device_id"] = r["device_id"].astype(str)
    r["window"] = r["window"].astype("int64")
    return r[KEYS + FEATS + ["node_label", "global_label"]].sort_values(KEYS).reset_index(drop=True)


def build_node(d: Data, cfg: dict, sp: Split, seed: int, *, classifier: str = "rf", cw: Split | None = None,
               variant: str = "base") -> NodeBundle:
    if variant not in VARIANTS:
        raise ValueError(variant)
    t0 = time.perf_counter()
    H = {p: honest_reports(d, sp.windows[p], sp.reporters[p]) for p in PARTS}
    htr = H["TRAIN"]
    X = htr[FEATS].to_numpy(np.float64); y = htr["node_label"].astype(int).to_numpy()
    model = _fit_classifier(cfg, classifier, seed, X, y)
    t1 = time.perf_counter()
    hv = H["VAL_CAL"]
    raw = model.predict_proba(hv[FEATS].to_numpy(np.float64))[:, _pos(model)]
    where = f"{sp.name}/{sp.fold}/seed={seed}"
    class_counts = {"cal_N": require_two_per_class(hv["node_label"], "cal_N (VAL_CAL node_label)", where)}
    cal_N = (IdentityCal() if variant == "no_platt" else Platt()).fit(raw, hv["node_label"].astype(int).to_numpy())
    min_win = int(cfg["method_params"]["min_reporter_reports"])
    pop = cfg["attacks"]["mu_ben_population"]
    if sp.name == "RD":
        # attacker side: the held-out reporter may use its own benign CW TRAIN history (not used for defender fitting)
        mt, mg, mm = mu_ben_table(d, cw.windows["TRAIN"], cw.reporters["TRAIN"], pop, min_win)
        mm["source"] = "CW_TRAIN_all_cw_reporters (attacker-side)"
    else:
        mt, mg, mm = mu_ben_table(d, sp.windows["TRAIN"], sp.reporters["TRAIN"], pop, min_win)
        mm["source"] = f"{sp.name}_TRAIN"
    nb = NodeBundle(seed=seed, split=sp, classifier=classifier, model=model, cal_N=cal_N, H=H, mu_table=mt,
                    mu_global=mg, mu_meta=mm)
    nb.class_counts = class_counts
    nb.fit_log = {"classifier_fit": sorted(set(htr["window"].astype(int))),
                  "classifier_fit_reporters": sorted(set(htr["device_id"])),
                  "platt_N_fit": sorted(set(hv["window"].astype(int))),
                  "platt_N_fit_reporters": sorted(set(hv["device_id"]))}
    t = pd.Series(d.window_type(htr["window"].to_numpy()), index=htr.index)
    nb.audit["train_composition"] = {"n_rows": int(len(htr)), "benign": int((y == 0).sum()), "attack": int((y == 1).sum()),
                                     "by_type": {k: int(v) for k, v in t[y == 1].value_counts().sort_index().items()}}
    nb.audit["cal_N_monotone"] = require_monotone(calibration_grid_monotone(cal_N), f"cal_N {sp.name}/{sp.fold}/seed={seed}")
    nb.timings = {"fit_classifier_s": t1 - t0, "build_node_s": time.perf_counter() - t0}
    return nb


# ------------------------------------------------------------------------------------------------ gateway
def gateway_obs(fs: pd.DataFrame, nodes, windows, s: float, seed: int, *, stream: str = "gw") -> pd.DataFrame:
    """Sampled gateway observations (Section 5.4): each flow is kept iff hash(raw_flow_id, seed, stream) < s, so sampled
    sets are nested across s and shared by all methods. Retained flows are aggregated per (device, window) into the same
    18 features; counts and the seven sums per direction are multiplied by 1/s, mean duration uses retained flows.
    An observation exists when at least one associated flow is retained."""
    vn = set(map(str, nodes)); win = {int(w) for w in windows}
    if not vn or not win:
        return pd.DataFrame(columns=KEYS + FEATS + ["gateway_available"])
    sub = fs[fs["window"].isin(win)]
    if s < 1.0:
        u = keep_uniform(sub["raw_flow_id"].to_numpy(), seed, stream)
        kept = sub[u < float(s)]
    else:
        kept = sub
    base8 = FLOW_COLUMNS; vol7 = SUM_COLUMNS
    frames = []
    inv = 1.0 / float(s) if s < 1.0 else 1.0
    for dname, ipcol in (("out", "src_ip"), ("in", "dst_ip")):
        v = kept[kept[ipcol].isin(vn)]
        if v.empty:
            frames.append(pd.DataFrame(columns=KEYS)); continue
        g = v.groupby([v[ipcol].astype(str).rename("device_id"), v["window"].astype("int64")], observed=True)
        summ = g[base8].sum(); cnt = g.size().rename("k")
        fr = summ.join(cnt).reset_index()
        out = {"device_id": fr["device_id"].astype(str), "window": fr["window"].astype("int64"),
               f"{dname}_duration": fr["duration"] / fr["k"]}
        for c in vol7:
            out[f"{dname}_{c}_sum"] = fr[c] * inv
        out[f"{dname}_count"] = fr["k"] * inv
        frames.append(pd.DataFrame(out))
    o = frames[0].merge(frames[1], on=KEYS, how="outer")
    for c in FEATS:
        if c not in o.columns:
            o[c] = 0.0
    o[FEATS] = o[FEATS].astype(np.float64).fillna(0.0)
    o["device_id"] = o["device_id"].astype(str); o["window"] = o["window"].astype("int64")
    o["gateway_available"] = 1
    o = o[o["window"].isin(win)]
    return o[KEYS + FEATS + ["gateway_available"]].sort_values(KEYS).reset_index(drop=True)


def rd_flow_filter(fs: pd.DataFrame, r: str, others) -> tuple[pd.DataFrame, int]:
    """RD: flows connecting the held-out reporter r to other monitored reporters are excluded from gateway observations."""
    oth = set(map(str, others))
    src = fs["src_ip"]; dst = fs["dst_ip"]
    bad = ((src == r) & dst.isin(oth)) | ((dst == r) & src.isin(oth))
    return fs[~bad.to_numpy()], int(bad.sum())


def attach_gateway(d: Data, cfg: dict, nb: NodeBundle, s: float, *, variant: str = "base") -> GatewayBundle:
    """Gateway observations at sampling rate s, gateway Platt calibration (VAL_CAL, covered reports) and Feature
    Residual statistics (TRAIN pairs)."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    sp = nb.split; seed = nb.seed
    fs = d.flow_store
    meta = {}
    if sp.name == "RD":
        fs, nx = rd_flow_filter(fs, sp.fold, sp.reporters["TRAIN"])
        meta["rd_excluded_flows"] = nx
    G = {}
    for p in PARTS:
        g = gateway_obs(fs, sp.reporters[p], sp.windows[p], s, seed, stream=cfg["gateway"]["stream"])
        g = g.merge(d.o_all[KEYS + ["node_label"]], on=KEYS, how="left")
        n_unmatched = int(g["node_label"].isna().sum())
        if n_unmatched:
            raise ValidityError(f"{sp.name}/{sp.fold}/seed={seed}/s={s}/{variant}/{p}: {n_unmatched} gateway rows without a node report")
        g["node_label"] = g["node_label"].astype(int)
        G[p] = g
    f_G = nb.model                                         # the same classifier is applied to both views
    gv = G["VAL_CAL"]
    raw = f_G.predict_proba(gv[FEATS].to_numpy(np.float64))[:, _pos(f_G)] if len(gv) else np.zeros(0)
    meta["class_counts_cal_G"] = require_two_per_class(gv["node_label"], "cal_G (VAL_CAL gateway node_label)",
                                                       f"{sp.name}/{sp.fold}/seed={seed}/s={s}/{variant}")
    cal_G = (IdentityCal() if variant == "no_platt" else Platt()).fit(raw, gv["node_label"].astype(int).to_numpy())
    mp = cfg["method_params"]
    gstats = feature_residual_stats(nb.H["TRAIN"], G["TRAIN"], FEATS, float(mp["std_floor"]), int(mp["min_reporter_reports"]))
    meta.update({"feature_residual_fallback_devices": list(gstats.fallback_devices),
                 "cal_G_monotone": require_monotone(calibration_grid_monotone(cal_G),
                                                    f"cal_G {sp.name}/{sp.fold}/seed={seed}/s={s}/{variant}"),
                 "n_gateway_rows": {p: int(len(G[p])) for p in PARTS}})
    return GatewayBundle(s=s, G=G, f_G=f_G, cal_G=cal_G, gstats=gstats, variant=variant, meta=meta)
