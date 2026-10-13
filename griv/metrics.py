"""Evaluation metrics (Section 5.6).

Attack metrics per (beta, attack type, row, stratum) with stratum = all / pre_rf=1 / pre_rf=0 (node-report IDS decision on the
honest report):
  Operational TPR, Covered TPR, simultaneous evasion (= 1 - Operational TPR), tampered-report coverage (availability),
  tampered-report attribution TPR (integrity alerts on modified reports).
Benign metrics (all unmodified benign TEST reports): Operational FPR, benign integrity FPR, alerts per 1,000
node-windows and per device-day equivalent.

Denominators:
- fixed concealment (mode "fixed"): targeted reports whose features were actually modified.
- adaptive concealment (mode "chosen"): all targeted reports, including those for which the attacker selects alpha = 0.
  Attribution uses the reports actually modified against that row.
Reports of reporters that are not compromised at beta use their honest columns (*_honest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .compromise import n_compromised
from .config import TYPES
from .methods import ARCH, METHODS

ALL_ROWS = METHODS + [f"A:{a}" for a in ARCH]
STRATA = ("all", "pre_rf=1", "pre_rf=0")


def k_of(beta: float, N: int) -> int:
    return n_compromised(beta, N)


def n_eligible(f: pd.DataFrame) -> int:
    """Length N of the compromise ranking (constant within one shard and seed)."""
    if "n_eligible" in f.columns and len(f):
        v = f["n_eligible"].unique()
        if len(v) != 1:
            raise ValueError("n_eligible differs within frame -- compute per shard/seed/fold")
        return int(v[0])
    return int(f.loc[f["compromise_rank"] >= 0, "device_id"].nunique())


def targeted_at(f: pd.DataFrame, beta: float, N: int | None = None) -> np.ndarray:
    N = n_eligible(f) if N is None else N
    k = k_of(beta, N)
    rk = f["compromise_rank"].to_numpy()
    return (rk >= 0) & (rk < k) & (f["node_label"].to_numpy() == 1)


def _mode(adaptive) -> str:
    return "chosen" if adaptive in (True, "chosen") else "fixed"


def _cols(row: str, adaptive):
    """(operational column or None, integrity column or None, node-decision column, was_modified column or None)."""
    ch = _mode(adaptive) == "chosen"
    if row.startswith("A:"):
        a = row[2:]
        if ch and a != "GatewayIDS":
            return f"A_{a}", None, f"post_rf_{a}", f"was_modified_{a}"
        return f"A_{a}", None, "post_rf", (None if ch else "was_modified")
    if ch:
        return f"O_{row}", f"I_{row}", f"post_rf_{row}", f"was_modified_{row}"
    return f"O_{row}", f"I_{row}", "post_rf", "was_modified"


def honest_name(col: str) -> str:
    """Submitted-value column -> honest-value column (post_rf, post_rf_<m> -> pre_rf; others -> <col>_honest)."""
    return "pre_rf" if col == "post_rf" or col.startswith("post_rf_") else col + "_honest"


def _applicable(f):
    return f["attack_applicable"].to_numpy().astype(bool) if "attack_applicable" in f.columns else np.ones(len(f), bool)


def tampered_mask(f: pd.DataFrame, tb: np.ndarray, adaptive, row: str | None = None) -> np.ndarray:
    """Modified reports. fixed: targeted & applicable & was_modified; chosen: targeted & applicable & was_modified_<row>."""
    app = _applicable(f)
    if _mode(adaptive) == "fixed":
        return tb & app & f["was_modified"].to_numpy().astype(bool)
    wm = _cols(row, adaptive)[3] if row else None
    if wm and wm in f.columns:
        return tb & app & f[wm].to_numpy().astype(bool)
    return tb & app


def operational_mask(f, tb, adaptive, row=None):
    """Operational denominator: fixed = modified targeted reports; chosen = all targeted reports."""
    if _mode(adaptive) == "fixed":
        return tampered_mask(f, tb, "fixed")
    return tb & _applicable(f)


def cell_state(n_targeted, n_applicable, n_modified, mode) -> str:
    """OK, or SNA (no targeted report) / ONA (none applicable) / ZERO_EFFECT (none modified): metric NA, excluded from macros."""
    if n_targeted == 0:
        return "SNA"
    if n_applicable == 0:
        return "ONA"
    if mode == "fixed" and n_modified == 0:
        return "ZERO_EFFECT"
    return "OK"


def attack_metrics(f: pd.DataFrame, beta: float, adaptive, rows=ALL_ROWS, by_type=True) -> pd.DataFrame:
    """Attack metrics per (beta, attack type, row, stratum) with their denominators (n = operational, n_attr = attribution)."""
    mode = _mode(adaptive)
    N = n_eligible(f)
    tb = targeted_at(f, beta, N)
    app = _applicable(f)
    pre = f["pre_rf"].to_numpy()
    cov = f["covered"].to_numpy().astype(bool)
    at = f["attack_type"].to_numpy()
    out = []
    types = TYPES if by_type else ["ALL"]
    for r in rows:
        ocol, icol, pcol, wcol = _cols(r, adaptive)
        if (ocol and ocol not in f.columns) or (icol and icol not in f.columns):
            continue
        O = f[ocol].to_numpy().astype(bool) if ocol else None
        I = f[icol].to_numpy().astype(bool) if icol else None
        WM = f[wcol].to_numpy().astype(bool) if (wcol and wcol in f.columns) else None
        WMc = WM if WM is not None else (f["was_modified_GRIV"].to_numpy().astype(bool)
                                         if "was_modified_GRIV" in f.columns else None)
        Mop = operational_mask(f, tb, adaptive, r)
        Matt = tampered_mask(f, tb, adaptive, r) if (mode == "fixed" or WM is not None) else None
        for t in types:
            ty = (at == t) if by_type else np.ones(len(f), bool)
            for st in STRATA:
                sm = ty if st == "all" else (ty & (pre == (1 if st == "pre_rf=1" else 0)))
                n_t = int((tb & sm).sum()); n_a = int((tb & app & sm).sum())
                n_m = int((tb & app & sm & WMc).sum()) if WMc is not None else -1
                state = cell_state(n_t, n_a, n_m, mode)
                m = Mop & sm
                n = int(m.sum()); nc = int((m & cov).sum())
                rec = dict(beta=beta, attack_type=t, row=r, stratum=st, n=n, n_covered=nc, n_targeted=n_t,
                           n_applicable=n_a, n_modified=n_m, state=state, mode=mode)
                ok = state == "OK" and n > 0
                if ok and O is not None:
                    rec["operational_TPR"] = float(O[m].mean())
                    rec["simultaneous_evasion"] = 1.0 - rec["operational_TPR"]
                    rec["covered_TPR"] = float(O[m & cov].mean()) if nc else np.nan
                    rec["n_alert"] = int(O[m].sum())
                if ok:
                    rec["tampered_report_coverage"] = float(cov[m].mean())
                if I is not None and Matt is not None:
                    ma = Matt & sm
                    rec["n_attr"] = int(ma.sum())
                    if state == "OK" and ma.any():
                        rec["tampered_report_attribution_TPR"] = float(I[ma].mean())
                        rec["n_integrity_alert"] = int(I[ma].sum())
                out.append(rec)
    return pd.DataFrame(out)


def untampered_metrics(f: pd.DataFrame, beta: float, adaptive, rows=ALL_ROWS, n_test_windows: int | None = None
                       ) -> pd.DataFrame:
    """Benign metrics on unmodified reports (honest values). Integrity rows: benign integrity FPR =
    benign_report_misattribution_rate (integrity alerts over all unmodified benign reports)."""
    N = n_eligible(f)
    tb = targeted_at(f, beta, N)
    nl = f["node_label"].to_numpy()
    cov = f["covered_honest"].to_numpy().astype(bool) if "covered_honest" in f.columns else f["covered"].to_numpy()
    if "n_test_reporters" in f.columns and f["fold"].nunique() <= 1:
        n_rep = int(f["n_test_reporters"].iloc[0])
    else:
        n_rep = f["device_id"].nunique()
    if n_test_windows is not None:
        nw = n_test_windows
    elif "n_test_windows" in f.columns:
        nw = int(f["n_test_windows"].iloc[0])
    else:
        nw = int(f["window"].nunique())
    days = nw * 60.0 / 86400.0
    out = []
    for r in rows:
        ocol, icol, _, _ = _cols(r, adaptive)
        if (ocol and ocol not in f.columns) or (icol and icol not in f.columns):
            continue
        T = tampered_mask(f, tb, adaptive, r)
        unt = ~T
        ben = unt & (nl == 0)
        O = f[honest_name(ocol)].to_numpy().astype(bool) if ocol else None
        I = f[honest_name(icol)].to_numpy().astype(bool) if icol else None
        rec = dict(beta=beta, row=r, n_benign=int(ben.sum()), n_benign_covered=int((ben & cov).sum()),
                   n_untampered=int(unt.sum()), n_reporters=int(n_rep), test_days_equivalent=days)
        if O is not None:
            rec["benign_operational_FPR"] = float(O[ben].mean()) if ben.any() else np.nan
            rec["alerts_per_1000_node_windows"] = 1000.0 * float(O[unt].mean()) if unt.any() else np.nan
            rec["alerts_per_device_day_equivalent"] = float(O[unt].sum()) / (n_rep * days) if n_rep and days else np.nan
        if I is not None:
            rec["benign_integrity_FPR"] = float(I[ben].mean()) if ben.any() else np.nan
            rec["benign_integrity_FPR_covered"] = float(I[ben & cov].mean()) if (ben & cov).any() else np.nan
            rec["integrity_alerts_per_1000_node_windows"] = 1000.0 * float(I[unt].mean()) if unt.any() else np.nan
            rec["integrity_alerts_per_device_day_equivalent"] = float(I[unt].sum()) / (n_rep * days) if n_rep and days else np.nan
        out.append(rec)
    return pd.DataFrame(out)
