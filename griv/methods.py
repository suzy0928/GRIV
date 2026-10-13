"""Integrity-verification methods (Sections 4.3 and 5.5) and the detector that fits and applies them.

Integrity alert of method m: I_m = covered & S_m > tau_m, except
    GRIV: I = NodeReportIDS(submitted) == benign & covered & p_G > tau_G.
Operational alert: O_m = NodeReportIDS(submitted) == attack | I_m.
Reference rows (no integrity status): NodeReportIDS = node-report IDS attack decision; GatewayIDS = covered & p_G > tau_G;
NodeOrGatewayIDS = NodeReportIDS | GatewayIDS (equals O_GRIV for every report; checked on every shard).

Scores (p~ = p clipped to [1e-6, 1 - 1e-6]):
    GRIV                   p_G (with the node decision gate above)
    FeatureResidual        sum_k ((r_k - o_k - mu_k) / max(sigma_k, eps))^2, reporter-specific mu, sigma on covered TRAIN pairs
    ProbabilityDifference  p_G - p_N
    JSDivergence           Jensen-Shannon divergence between Bern(p~_N) and Bern(p~_G), natural log
    LogOddsDifference      logit(p~_G) - logit(p~_N)
    MahalanobisDistance    min distance of [logit p~_N, logit p~_G] to the benign / attack Gaussians fitted on covered honest
                           VAL_CAL reports (covariance + 1e-6 I)
    MultiplicativeScore    p_G (1 - p_N) (decision-rule ablation)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import thresholds as TH
from .features import FEATS
from .errors import ValidityError
from .models import require_two_per_class

METHODS = ["GRIV", "FeatureResidual", "ProbabilityDifference", "JSDivergence", "LogOddsDifference", "MahalanobisDistance",
           "MultiplicativeScore"]
ARCH = ["NodeReportIDS", "GatewayIDS", "NodeOrGatewayIDS"]
KEYS = ["device_id", "window"]


def clip(p, eps):
    return np.clip(np.asarray(p, np.float64), eps, 1.0 - eps)


def logit(p):
    return np.log(p) - np.log1p(-p)


def js_bern(pn, pg):
    """Jensen-Shannon divergence between Bern(pn) and Bern(pg), natural log."""
    m = 0.5 * (pn + pg)

    def kl(a, b):
        return a * (np.log(a) - np.log(b)) + (1 - a) * (np.log1p(-a) - np.log1p(-b))
    return 0.5 * kl(pn, m) + 0.5 * kl(pg, m)


class Mahalanobis:
    """Two-cluster Mahalanobis distance in (logit p~_N, logit p~_G) space."""
    def __init__(self, eps, reg):
        self.eps, self.reg = eps, reg

    def fit(self, pN, pG, y):
        z = np.c_[logit(clip(pN, self.eps)), logit(clip(pG, self.eps))]
        self.mu, self.P = [], []
        for c in (0, 1):
            zc = z[np.asarray(y).astype(int) == c]
            cov = np.cov(zc, rowvar=False) + self.reg * np.eye(2)
            self.mu.append(zc.mean(axis=0)); self.P.append(np.linalg.inv(cov))
        self.n = [int((np.asarray(y) == c).sum()) for c in (0, 1)]
        return self

    def score(self, pN, pG):
        z = np.c_[logit(clip(pN, self.eps)), logit(clip(pG, self.eps))]
        ds = []
        for mu, P in zip(self.mu, self.P):
            dz = z - mu
            ds.append(np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", dz, P, dz), 0.0)))
        return np.minimum(ds[0], ds[1])


def feature_residual_arr(X, Gx, devices, gstats):
    """Feature Residual score with e = r - o.
    Rows without a gateway observation (NaN) give NaN."""
    devs = np.asarray(devices).astype(str)
    uniq = {d: (gstats.get_mu(d), np.maximum(gstats.get_sigma(d), gstats.eps)) for d in set(devs.tolist())}
    mu = np.vstack([uniq[d][0] for d in devs]) if len(devs) else np.zeros((0, X.shape[1]))
    sg = np.vstack([uniq[d][1] for d in devs]) if len(devs) else np.zeros((0, X.shape[1]))
    with np.errstate(invalid="ignore"):
        z = (np.asarray(X, np.float64) - np.asarray(Gx, np.float64) - mu) / sg
        g = (z ** 2).sum(axis=1)
    g[~np.isfinite(g)] = np.nan
    return g


def join_gateway(reports: pd.DataFrame, G: pd.DataFrame):
    """Attach gateway observations in report order -> (covered, Gx with NaN rows when unavailable)."""
    m = reports[KEYS].merge(G[KEYS + FEATS], on=KEYS, how="left", validate="many_to_one")
    Gx = m[FEATS].to_numpy(np.float64)
    covered = ~np.isnan(Gx).any(axis=1)
    return covered, Gx


class Detector:
    """For one (node bundle, gateway bundle): fit MahalanobisDistance on VAL_CAL, select every tau on VAL_THR, evaluate reports."""

    def __init__(self, cfg, nb, gb):
        self.cfg, self.nb, self.gb = cfg, nb, gb
        mp = cfg["method_params"]
        self.eps = float(mp["prob_clip_eps"])
        self.fit_log = {}
        vc = nb.H["VAL_CAL"]
        cov_c, Gx_c = join_gateway(vc, gb.G["VAL_CAL"])
        Xc = vc[FEATS].to_numpy(np.float64)
        pN_c = nb.p_N(Xc); pG_c = np.full(len(vc), np.nan)
        if cov_c.any():
            pG_c[cov_c] = gb.p_G(Gx_c[cov_c])
        yc = vc["node_label"].astype(int).to_numpy()
        where = f"{nb.split.name}/{nb.split.fold}/seed={nb.seed}/s={gb.s}/{gb.variant}"
        self.class_counts = {"MahalanobisDistance": require_two_per_class(yc[cov_c], "MahalanobisDistance (VAL_CAL covered node_label)", where)}
        self.mahalanobis = Mahalanobis(self.eps, float(mp["mahalanobis"]["cov_reg"])).fit(pN_c[cov_c], pG_c[cov_c], yc[cov_c])
        vt = nb.H["VAL_THR"]
        ev = self.evaluate(vt, taus=None)
        nl = vt["node_label"].astype(int).to_numpy(); cov_t = ev["covered"]
        self.where = where
        self.check_finite(ev, (nl == 0) & cov_t, "VAL_THR benign-covered")
        q = cfg["threshold"]["quantile"]
        self.tau = {}; self.tau_rows = []
        for m in METHODS:
            r = TH.tau(ev[f"score_{m}"], nl, cov_t, q)
            self.tau[(m, q)] = r["tau"]
            self.tau_rows.append(dict(method=m, quantile=q, **r))
        self.q_main = q

    def scores(self, X, devices, covered, Gx, pN=None, pG=None):
        nb, gb = self.nb, self.gb
        p_N = nb.p_N(X) if pN is None else pN
        if pG is None:
            p_G = np.full(len(X), np.nan)
            if covered.any():
                p_G[covered] = gb.p_G(Gx[covered])
        else:
            p_G = pG
        e = self.eps
        pn, pg = clip(p_N, e), clip(np.nan_to_num(p_G, nan=0.5), e)
        S = {}
        S["GRIV"] = p_G.copy()
        S["FeatureResidual"] = np.where(covered, feature_residual_arr(X, Gx, devices, gb.gstats), np.nan)
        S["ProbabilityDifference"] = p_G - p_N
        S["JSDivergence"] = np.where(covered, js_bern(pn, pg), np.nan)
        S["LogOddsDifference"] = np.where(covered, logit(pg) - logit(pn), np.nan)
        S["MahalanobisDistance"] = np.where(covered, self.mahalanobis.score(p_N, np.nan_to_num(p_G, nan=0.5)), np.nan)
        S["MultiplicativeScore"] = p_G * (1.0 - p_N)
        for m in S:
            S[m] = np.where(covered, S[m], np.nan)
        return p_N, p_G, S

    def check_finite(self, ev, mask, what):
        """Every method score must be finite on covered reports."""
        for m in METHODS:
            s = np.asarray(ev[f"score_{m}"], float)[np.asarray(mask, bool)]
            bad = int((~np.isfinite(s)).sum())
            if bad:
                raise ValidityError(f"{self.where} {what}: {bad} non-finite {m} scores on covered reports")

    def gateway_alert(self, p_G, covered):
        """Gateway IDS decision: covered & p_G > tau_G (no node term)."""
        return TH.alert(p_G, self.tau[("GRIV", self.q_main)], covered)

    def alerts(self, S, covered, pred):
        I = {m: TH.alert(S[m], self.tau[(m, self.q_main)], covered) for m in METHODS}
        I["GRIV"] = I["GRIV"] & (np.asarray(pred) == 0)
        return I

    def evaluate(self, reports: pd.DataFrame, taus="fit", X=None):
        """Evaluate submitted reports. With X, features are X (after manipulation); keys and labels come from reports."""
        nb = self.nb
        Xs = reports[FEATS].to_numpy(np.float64) if X is None else np.asarray(X, np.float64)
        covered, Gx = join_gateway(reports, self.gb.G[_part_of(reports, nb)])
        raw_pN = nb.raw_p(Xs); pred = nb.predict(Xs)
        p_N, p_G, S = self.scores(Xs, reports["device_id"].to_numpy(), covered, Gx,
                                  pN=nb.cal_N.transform(raw_pN) if len(Xs) else np.zeros(0))
        out = {"covered": covered, "raw_pN": raw_pN, "pred": pred, "p_N": p_N, "p_G": p_G, "Gx": Gx}
        raw_pG = np.full(len(Xs), np.nan)
        if covered.any():
            raw_pG[covered] = self.gb.raw_p(Gx[covered])
        out["raw_pG"] = raw_pG
        for m in METHODS:
            out[f"score_{m}"] = S[m]
        if taus is None:
            return out
        I = self.alerts(S, covered, pred)
        for m in METHODS:
            out[f"I_{m}"] = I[m]; out[f"O_{m}"] = (pred == 1) | I[m]
        AG = self.gateway_alert(p_G, covered)
        out["A_NodeReportIDS"] = pred == 1; out["A_GatewayIDS"] = AG; out["A_NodeOrGatewayIDS"] = (pred == 1) | AG
        return out


def _part_of(reports, nb):
    """Partition of a report frame (all of its windows must lie in one partition)."""
    ws = set(reports["window"].astype(int).unique())
    for p, wl in nb.split.windows.items():
        if ws and ws <= set(map(int, wl)):
            return p
    raise ValueError("report frame spans multiple partitions")
