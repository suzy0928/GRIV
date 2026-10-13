"""Data partitions (Section 5.2): strengthened chronological-window (CW) and reporter-disjoint (RD).

Partitions: TRAIN / VAL_CAL / VAL_THR / TEST. Reports of the same absolute-time window are assigned together.
Purges use window IDs (one ID = one 60-s window): adjacent partitions are separated by more than K windows.
Purged and unused windows belong to no partition.
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import TYPES
from .data import Data, episodes
from .errors import ValidityError
from .provenance import json_hash

PARTS = ["TRAIN", "VAL_CAL", "VAL_THR", "TEST"]
FIT_PARTS = ["TRAIN", "VAL_CAL", "VAL_THR"]


class SplitCheckError(RuntimeError):
    """The CW cross-label purge removed too many TEST attack windows or emptied an attack type in a partition."""


@dataclass
class Split:
    name: str                       # CW | RD
    K: int
    windows: dict                   # partition -> sorted list of window ids
    reporters: dict                 # partition -> sorted list of reporter ids
    purged: dict = field(default_factory=dict)      # window -> reason
    fold: str = "-"                 # RD: held-out reporter
    meta: dict = field(default_factory=dict)

    @property
    def split_id(self) -> str:
        return json_hash({"name": self.name, "K": self.K, "fold": self.fold,
                          "windows": {p: list(map(int, self.windows[p])) for p in PARTS},
                          "reporters": {p: list(self.reporters[p]) for p in PARTS}})[:16]

    def part_of(self) -> dict:
        return {int(w): p for p in PARTS for w in self.windows[p]}

    def reports(self, d: Data, part: str) -> pd.DataFrame:
        """Reports of the monitored reporters in the partition's windows."""
        w = set(map(int, self.windows[part])); r = set(self.reporters[part])
        o = d.o
        return o[o["window"].isin(w) & o["device_id"].isin(r)].reset_index(drop=True)

    def partition_frame(self) -> pd.DataFrame:
        rows = [(self.name, self.fold, int(w), p) for p in PARTS for w in self.windows[p]]
        rows += [(self.name, self.fold, int(w), f"PURGED:{why}") for w, why in sorted(self.purged.items())]
        return pd.DataFrame(rows, columns=["split", "fold", "window", "partition"])


# ----------------------------------------------------------------------------------------------- helpers
def _purge_later(earlier_last: int, later: list, K: int):
    """Drop windows at the start of ``later`` with id <= earlier_last + K."""
    keep = [w for w in later if w > earlier_last + K]
    drop = [w for w in later if w <= earlier_last + K]
    return keep, drop


def split_val(val: list, K: int, side: str = "alternate", frac: float = 0.5):
    """Validation block -> (VAL_CAL, VAL_THR, purged). The first floor(n * frac) windows form VAL_CAL, the rest VAL_THR.
    Windows near the boundary are removed until the ID gap exceeds K (alternately from VAL_THR's start and VAL_CAL's end)."""
    v = sorted(int(w) for w in val)
    h = int(math.floor(len(v) * frac))
    cal, thr = v[:h], v[h:]
    purged = []
    turn = "THR"
    while cal and thr and thr[0] - cal[-1] <= K:
        if side == "later" or turn == "THR":
            purged.append(thr.pop(0)); turn = "CAL"
        else:
            purged.append(cal.pop()); turn = "THR"
    return cal, thr, sorted(purged)


def _attack_windows(d: Data, windows, reporters=None) -> set:
    """Windows with at least one attack-bearing report (optionally restricted to ``reporters``)."""
    o = d.o
    m = o["window"].isin(set(map(int, windows))) & (o["node_label"].astype(int) == 1)
    if reporters is not None:
        m &= o["device_id"].isin(set(reporters))
    return set(o.loc[m, "window"].astype(int))


def _types_present(d: Data, windows) -> set:
    return set(t for t in d.window_type(list(windows)) if t != "none")


def _min_dist(a, b) -> float:
    """Minimum window-ID distance between two window sets (inf if one is empty)."""
    A = np.array(sorted(map(int, a)), dtype=np.int64); Bv = np.array(sorted(map(int, b)), dtype=np.int64)
    if not len(A) or not len(Bv):
        return np.inf
    idx = np.searchsorted(Bv, A)
    d = np.inf
    for k in (idx - 1, idx):
        ok = (k >= 0) & (k < len(Bv))
        if ok.any():
            d = min(d, float(np.abs(A[ok] - Bv[k[ok]]).min()))
    return d


def cross_purge(windows: dict, K: int) -> tuple[dict, dict]:
    """Cross-label purge: remove TEST windows within K windows of any fitting window (TRAIN, VAL_CAL, VAL_THR) of any
    label group. Fitting windows are never removed."""
    fit = np.array(sorted(int(w) for p in FIT_PARTS for w in windows[p]), dtype=np.int64)
    drop = {}
    for w in windows["TEST"]:
        w = int(w)
        lo = np.searchsorted(fit, w - K, side="left"); hi = np.searchsorted(fit, w + K, side="right")
        if hi > lo:
            drop[w] = "xpurge_test_protective"
    out = {p: list(windows[p]) for p in FIT_PARTS}
    out["TEST"] = [w for w in windows["TEST"] if int(w) not in drop]
    return out, drop


# ----------------------------------------------------------------------------------------------- CW
def _strata(d: Data) -> dict:
    """Label groups: benign windows ('none') and one group per dominant attack type."""
    strata: dict = {}
    for w in d.all_windows:
        w = int(w)
        s = d.dom.get(w, "none") if int(d.gl[w]) == 1 else "none"
        strata.setdefault(s, []).append(w)
    return strata


def _stratum_bounds(n: int, K: int, test_frac: float, val_frac: float):
    """Nominal sizes of the chronological blocks within one label group (Python round)."""
    n_test = int(round(test_frac * n)); n_val = int(round(val_frac * n)); n_train = n - n_test - n_val - 2 * K
    if not (n_train > 0 and n_val > 0 and n_test > 0):
        raise SplitCheckError(f"label group infeasible at K={K} (n={n})")
    return n_train, n_val, n_test


def _align_stratum(ws, ep_of, K, n_train, n_val, n_test, tol, test_frac, val_frac):
    """Move a TRAIN|VAL or VAL|TEST boundary that falls inside a temporal episode to the nearest episode gap, if the
    partition fractions stay within ``tol`` of the nominal fractions. Returns (train, val, test, purged, moved)."""
    n = len(ws)
    tr = ws[:n_train]; va = ws[n_train + K:n_train + K + n_val]; te = ws[n - n_test:]
    purged = ws[n_train:n_train + K] + ws[n_train + K + n_val:n - n_test]
    moved = {"b1": "kept", "b2": "kept"}
    if ep_of is None:
        return tr, va, te, purged, moved
    gaps = [i for i in range(1, n) if ep_of(ws[i - 1]) != ep_of(ws[i])]

    def ok(tr_, va_, te_):
        tgt = {"TRAIN": 1 - test_frac - val_frac, "VAL": val_frac, "TEST": test_frac}
        got = {"TRAIN": len(tr_) / n, "VAL": len(va_) / n, "TEST": len(te_) / n}
        return all(len(x) > 0 for x in (tr_, va_, te_)) and all(abs(got[k] - tgt[k]) <= tol for k in tgt)

    if tr and va and ep_of(tr[-1]) == ep_of(va[0]):                      # boundary 1: TRAIN | VAL
        a = tr[-1]
        cand = sorted(gaps, key=lambda i: (abs(ws[i - 1] - a), ws[i - 1]))
        if cand:
            i = cand[0]
            tr2 = ws[:i]
            rest = [w for w in ws[i:] if w <= va[-1]] if va else []
            va2, _ = _purge_later(tr2[-1], rest, K)
            pur2 = [w for w in ws if w not in set(tr2) | set(va2) | set(te)]
            if ok(tr2, va2, te):
                moved["b1"] = f"moved:{a}->{tr2[-1]}"
                tr, va, purged = tr2, va2, pur2
            else:
                moved["b1"] = "inside_not_moved(criteria)"
        else:
            moved["b1"] = "inside_no_gap"
    if va and te and ep_of(va[-1]) == ep_of(te[0]):                      # boundary 2: VAL | TEST
        c = va[-1]
        cand = sorted(gaps, key=lambda i: (abs(ws[i - 1] - c), ws[i - 1]))
        if cand:
            i = cand[0]
            va2 = [w for w in ws[:i] if w >= va[0]]
            te2, _ = _purge_later(ws[i - 1], ws[i:], K)
            pur2 = [w for w in ws if w not in set(tr) | set(va2) | set(te2)]
            if ok(tr, va2, te2):
                moved["b2"] = f"moved:{c}->{va2[-1] if va2 else None}"
                va, te, purged = va2, te2, pur2
            else:
                moved["b2"] = "inside_not_moved(criteria)"
        else:
            moved["b2"] = "inside_no_gap"
    return tr, va, te, sorted(purged), moved


def _episode_index(bounds):
    """Map a window id to the index of the episode (a, b, _) in ``bounds`` that contains it, or -1."""
    def ep_of(w):
        for i, (a, b, _) in enumerate(bounds):
            if a <= w <= b:
                return i
        return -1
    return ep_of


def make_cw(d: Data, cfg: dict) -> Split:
    sc = cfg["splits"]["CW"]; com = cfg["split_common"]
    K = int(com["K"])
    eps = episodes(d.dom, d.gl, int(com["episode_gap_min"]))
    windows = {p: [] for p in PARTS}
    purged, stratum_of = {}, {}
    meta = {"strata": {}, "alignment": {}}
    for s, ws in _strata(d).items():
        ws = sorted(ws); n = len(ws)
        n_train, n_val, n_test = _stratum_bounds(n, K, sc["test_frac"], sc["val_frac"])
        ep_of = None
        if sc["episode_align"] and s != "none" and len(eps.get(s, [])) >= 2:
            ep_of = _episode_index(eps[s])
        tr, va, te, pur, moved = _align_stratum(ws, ep_of, K, n_train, n_val, n_test, sc["align_tolerance_pp"] / 100.0,
                                                sc["test_frac"], sc["val_frac"])
        cal, thr, vp = split_val(va, int(com["K_val"]), com["val_purge_side"], com["val_cal_frac"])
        stratum_of.update({int(w): s for w in tr + va + te})
        windows["TRAIN"] += tr; windows["VAL_CAL"] += cal; windows["VAL_THR"] += thr; windows["TEST"] += te
        for w in pur:
            purged[int(w)] = "stratum_purge"
        for w in vp:
            purged[int(w)] = "valcal_valthr_purge"
        meta["strata"][s] = dict(n=n, n_train=len(tr), n_val=len(va), n_val_cal=len(cal), n_val_thr=len(thr),
                                 n_test=len(te), n_purged=len(pur) + len(vp))
        meta["alignment"][s] = moved
    windows = {p: sorted(map(int, v)) for p, v in windows.items()}
    pre = {p: list(v) for p, v in windows.items()}
    windows, drop = cross_purge(windows, K)
    for w in drop:
        purged[w] = "xpurge"
    meta["xpurge_excluded_by_partition"] = {p: len(set(pre[p]) - set(windows[p])) for p in PARTS}
    _need(all(sorted(pre[p]) == sorted(windows[p]) for p in FIT_PARTS), "cross-label purge changed a fitting partition")
    xa = _attack_windows(d, sorted(drop))
    meta["xpurge_test_windows_excluded"] = {"benign": len(drop) - len(xa), "attack": len(xa)}
    before = _attack_windows(d, pre["TEST"]); after = _attack_windows(d, windows["TEST"])
    meta["test_attack_windows_before_xpurge"] = len(before)
    meta["test_attack_windows_after_xpurge"] = len(after)
    meta["test_attack_window_loss"] = (len(before) - len(after)) / max(1, len(before))
    meta["types_by_partition"] = {p: sorted(_types_present(d, windows[p])) for p in PARTS}
    # reporters = monitored reporters present in both TRAIN and TEST
    o = d.o
    n_tr = set(o.loc[o["window"].isin(set(windows["TRAIN"])), "device_id"])
    n_te = set(o.loc[o["window"].isin(set(windows["TEST"])), "device_id"])
    svn = sorted(set(d.vn) & n_tr & n_te)
    sp = Split(name="CW", K=K, windows=windows, reporters={p: svn for p in PARTS}, purged=purged, meta=meta)
    sp.stratum_of = stratum_of
    sp.K_val = int(com["K_val"])
    check_xpurge(sp, cfg)
    return sp


def check_xpurge(sp: Split, cfg: dict):
    """TEST attack-window loss from the cross-label purge must stay <= 10 % and every attack type must remain in every
    partition; otherwise the run stops."""
    chk = cfg["splits"]["CW"]["xpurge_check"]
    loss = sp.meta["test_attack_window_loss"]
    missing = {p: [t for t in TYPES if t not in sp.meta["types_by_partition"][p]] for p in PARTS}
    failed = loss > chk["max_test_attack_window_loss"] or (chk["all_types_all_partitions"] and any(missing.values()))
    sp.meta["xpurge_check"] = {"loss": loss, "max": chk["max_test_attack_window_loss"], "missing_types": missing,
                               "failed": bool(failed)}
    if failed:
        raise SplitCheckError(f"{sp.name}: cross-label purge criterion violated (loss={loss:.4f}, missing={missing})")


# ----------------------------------------------------------------------------------------------- RD
def rd_folds(d: Data, cw: Split) -> tuple[list, list]:
    """(all folds = CW reporters, attack folds = reporters with attack-bearing CW TEST reports)."""
    allr = list(cw.reporters["TEST"])
    o = d.o
    te = o[o["window"].isin(set(cw.windows["TEST"])) & o["device_id"].isin(set(allr))]
    atk = sorted(set(te.loc[te["node_label"].astype(int) == 1, "device_id"]))
    return allr, atk


def make_rd(d: Data, cw: Split, r: str) -> Split:
    """Hold out reporter r: CW window assignment, r excluded from TRAIN / VAL_CAL / VAL_THR, TEST = r's CW TEST reports."""
    others = sorted(x for x in cw.reporters["TRAIN"] if x != r)
    reps = {"TRAIN": others, "VAL_CAL": others, "VAL_THR": others, "TEST": [r]}
    sp = Split(name="RD", K=cw.K, windows={p: list(cw.windows[p]) for p in PARTS}, reporters=reps,
               purged=dict(cw.purged), fold=r, meta={"base_split_id": cw.split_id, "held_out": r})
    sp.stratum_of = getattr(cw, "stratum_of", {}); sp.K_val = getattr(cw, "K_val", cw.K)
    return sp


def select_rd_folds(d: Data, cw: Split, spec) -> list:
    """spec = 'all' (17 folds) or an integer n (the first n attack-bearing folds; quick profile)."""
    allr, atk = rd_folds(d, cw)
    if spec == "all":
        return allr
    return atk[:int(spec)]


# ----------------------------------------------------------------------------------------------- invariants / tables
def _need(cond, msg):
    if not cond:
        raise ValidityError(f"[split invariant] {msg}")


def check_invariants(sp: Split, d: Data, *, flows: bool = False):
    """No window in two partitions, purged windows outside partitions, no (reporter, window) in two partitions,
    TEST more than K windows from every fitting window, and within each label group TRAIN -> VAL -> TEST gaps > K and
    VAL_CAL / VAL_THR gap > K_val. With flows=True, no raw flow is shared by two partitions."""
    sets = {p: set(map(int, sp.windows[p])) for p in PARTS}
    for i, a in enumerate(PARTS):
        for b in PARTS[i + 1:]:
            _need(not (sets[a] & sets[b]), f"{sp.name}: window overlap {a}/{b}")
    allp = set().union(*sets.values())
    _need(not (allp & set(sp.purged)), f"{sp.name}: purged window inside a partition")
    Kv = int(getattr(sp, "K_val", sp.K))
    fit = set().union(*(sets[p] for p in FIT_PARTS))
    dmin = _min_dist(sets["TEST"], fit)
    _need(dmin > sp.K, f"{sp.name}: TEST within K of a fitting window ({dmin})")
    so = getattr(sp, "stratum_of", {})
    by = {}
    for p in PARTS:
        for w in sets[p]:
            by.setdefault(so.get(w, "?"), {}).setdefault(p, set()).add(w)
    for s_, g in by.items():
        tr, cal, thr, te = (g.get(p, set()) for p in PARTS)
        va = cal | thr
        if tr and va:
            _need(min(va) - max(tr) > sp.K, f"{sp.name}/{s_}: TRAIN|VAL gap <= K")
        if va and te:
            _need(min(te) - max(va) > sp.K, f"{sp.name}/{s_}: VAL|TEST gap <= K")
        if tr and te and not va:
            _need(min(te) - max(tr) > sp.K, f"{sp.name}/{s_}: TRAIN|TEST gap <= K")
        _need(_min_dist(cal, thr) > Kv, f"{sp.name}/{s_}: VAL_CAL/VAL_THR closer than K_val")
    k = pd.concat([sp.reports(d, p)[["device_id", "window"]] for p in PARTS])
    _need(not k.duplicated().any(), f"{sp.name}: (reporter, window) duplicated across partitions")
    if flows:
        fs = d.flow_store
        sub = fs[["raw_flow_id", "window"]]
        g = sub.assign(p=sub["window"].map(sp.part_of())).dropna(subset=["p"]).groupby("raw_flow_id")["p"].nunique()
        _need(int((g > 1).sum()) == 0, f"{sp.name}: raw_flow_id in two partitions")
    return True


def _utc(w: int) -> str:
    return datetime.datetime.fromtimestamp(int(w) * 60, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")


def partition_table(sp: Split, d: Data, gap_min: int = 20) -> pd.DataFrame:
    """Per partition: date range (UTC), windows, reporters, temporal episodes, reports (benign / attack by type)."""
    eps = episodes(d.dom, d.gl, gap_min)
    rows = []
    for p in PARTS:
        ws = sp.windows[p]
        r = sp.reports(d, p)
        nl = r["node_label"].astype(int)
        t = pd.Series(d.window_type(r["window"].to_numpy()), index=r.index)
        wset = set(map(int, ws))
        n_eps = sum(1 for lst in eps.values() for (a, b, _) in lst if any(a <= w <= b for w in wset))
        row = dict(split=sp.name, fold=sp.fold, partition=p, n_windows=len(ws),
                   date_start_utc=_utc(min(ws)) if ws else "", date_end_utc=_utc(max(ws)) if ws else "",
                   n_reporters=int(r["device_id"].nunique()), n_temporal_episodes=n_eps,
                   n_attack_windows=len(_attack_windows(d, ws, sp.reporters[p])),
                   n_reports=len(r), n_benign_reports=int((nl == 0).sum()), n_attack_reports=int((nl == 1).sum()))
        for ty in TYPES:
            row[f"attack_reports_{ty}"] = int(((nl == 1) & (t == ty)).sum())
        row["xpurge_excluded_windows"] = sp.meta.get("xpurge_excluded_by_partition", {}).get(p, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def make(d: Data, cfg: dict, name: str) -> Split:
    if name == "CW":
        return make_cw(d, cfg)
    raise ValueError(name)
