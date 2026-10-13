"""Integer and relation constraints of submitted reports (projection Pi in Eq. (1)) and the report validators.

project applies Pi to a feature matrix (counts and sums rounded half up). validate_reports (set level) and
consistency_violations (row level) check that every submitted report satisfies the constraints.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUT_SUMS = ["out_src_bytes_sum","out_dst_bytes_sum","out_missed_bytes_sum","out_src_pkts_sum",
            "out_src_ip_bytes_sum","out_dst_pkts_sum","out_dst_ip_bytes_sum"]
IN_SUMS = [c.replace("out_","in_") for c in OUT_SUMS]


def validate_reports(df):
    """Return list of violations. Physically-valid node-window report invariants:
       counts non-negative integers; count==0 => that direction's sums and duration ==0;
       all volume sums >=0; packet/byte sums integer; non-negativity everywhere."""
    v=[]
    for d,cnt,sums,dur in [("out","out_count",OUT_SUMS,"out_duration"),("in","in_count",IN_SUMS,"in_duration")]:
        c=df[cnt].to_numpy(float)
        if np.any(c<0): v.append(f"{cnt}: negative")
        if np.any(np.abs(c-np.round(c))>1e-9): v.append(f"{cnt}: non-integer ({int((np.abs(c-np.round(c))>1e-9).sum())} rows)")
        S=df[sums].to_numpy(float)
        if np.any(S<-1e-9): v.append(f"{d} sums: negative")
        zero=c==0
        if zero.any():
            if np.any(np.abs(S[zero])>1e-9): v.append(f"{d}: count==0 but sums!=0")
            if np.any(np.abs(df[dur].to_numpy(float)[zero])>1e-9): v.append(f"{d}: count==0 but duration!=0")
        # packet sums integer
        for pk in [s for s in sums if "pkts" in s or "bytes" in s]:
            x=df[pk].to_numpy(float)
            if np.any(np.abs(x-np.round(x))>1e-6):
                v.append(f"{pk}: non-integer ({int((np.abs(x-np.round(x))>1e-6).sum())} rows)"); break
    return v


VOL7 = ["src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"]


def consistency_violations(df, feats):
    """returns boolean mask of VALID reports (True=passes)."""
    ok = np.ones(len(df), bool)
    for d, cnt, dur in [("out", "out_count", "out_duration"), ("in", "in_count", "in_duration")]:
        c = df[cnt].to_numpy(float)
        ok &= (c >= 0) & (np.abs(c - np.round(c)) <= 1e-9)                     # non-neg integer count
        sums = [f"{d}_{v}_sum" for v in VOL7]
        Sm = df[sums].to_numpy(float)
        ok &= (Sm >= -1e-9).all(axis=1)                                       # non-neg volume sums
        zero = c == 0
        ok &= ~(zero & ((np.abs(Sm) > 1e-9).any(axis=1) | (np.abs(df[dur].to_numpy(float)) > 1e-9)))
        for pk in [f"{d}_{v}_sum" for v in VOL7 if ("pkts" in v or "bytes" in v)]:
            x = df[pk].to_numpy(float); ok &= (np.abs(x - np.round(x)) <= 1e-6)  # integer pkt/byte sums
    return ok


HALF_UP_TOL = 1e-4


def round_half_up(x: np.ndarray) -> np.ndarray:
    """Round half up for non-negative arrays (np.round rounds half to even).

    floor(x + 0.5 + 1e-4) guards against float error turning an exact .5 tie into 0.4999... On the evaluated grids the
    exact values of (1 - alpha) x + alpha mu have at most three decimals (x integer, mu integer or .5), so non-tie values
    are at least 0.005 from .5; the 1e-4 tolerance lies inside that margin and far above float64 error."""
    x = np.asarray(x, dtype=np.float64)
    return np.floor(x + 0.5 + HALF_UP_TOL)


def project(X: np.ndarray, feats: list) -> np.ndarray:
    """Apply the projection to an n x 18 feature matrix: counts and the seven sums per direction become non-negative
    integers (round half up), mean durations become non-negative, and a zero count zeroes that direction's sums and duration."""
    X = np.array(X, dtype=np.float64, copy=True)
    idx = {f: i for i, f in enumerate(feats)}
    for pre in ("out_", "in_"):
        ci = idx[pre + "count"]
        X[:, ci] = round_half_up(np.maximum(X[:, ci], 0.0))
        si = [idx[f"{pre}{v}_sum"] for v in VOL7]
        X[:, si] = round_half_up(np.maximum(X[:, si], 0.0))
        di = idx[pre + "duration"]
        X[:, di] = np.maximum(X[:, di], 0.0)
        z = X[:, ci] == 0
        X[np.ix_(z, si + [di])] = 0.0
    return X


def validator_ok(X: np.ndarray, feats: list) -> np.ndarray:
    """Row-wise validator (same rule as consistency_violations). True = valid."""
    df = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=feats)
    return consistency_violations(df, feats)


def validate_matrix(X: np.ndarray, feats: list) -> list:
    """Set-level validator (validate_reports). Empty list = valid."""
    return validate_reports(pd.DataFrame(np.asarray(X, dtype=np.float64), columns=feats))
