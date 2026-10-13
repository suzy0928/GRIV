"""Processing cost of the GRIV path (Section 6.2, Supplementary Table S9).

Stages: A gateway flow selection, B feature aggregation, C node-report IDS decision + gateway probability (classifier +
Platt), D GRIV decision, E the whole online path A -> B -> join with reports -> C -> D run as one data flow.
Warm-up and repeated timing with time.perf_counter_ns, single-threaded inference. Unit: one pass over the whole TEST
partition; per-report latency = time / number of reports.
Memory: peak resident set size of a fresh subprocess that runs stage E once on the TEST partition (batch replay);
the subprocess receives only the objects that the GRIV path needs (node classifier, gateway classifier + calibration,
tau_G, sampling settings, TEST flows and reports).
These offline measurements exclude traffic capture, flow export, transport and storage.
"""
from __future__ import annotations

import ctypes
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from . import thresholds as TH
from .features import FEATS
from .features import FLOW_COLUMNS, SUM_COLUMNS
from .models import _pos
from .sampling import keep_uniform

KEYS = ["device_id", "window"]


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        # explicit argtypes/restype keep the pseudo handle (-1) intact on 64-bit Windows
        k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb) or c.PeakWorkingSetSize <= 0:
            raise RuntimeError("GetProcessMemoryInfo failed (peak RSS unavailable)")
        return int(c.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _timeit(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter_ns(); fn(); ts.append(time.perf_counter_ns() - t)
    return np.array(ts, dtype=np.int64)


# ------------------------------------------------------------------------------------------------ stages
def stage_A(sub, s, seed, stream):
    if s >= 1.0:
        return sub
    u = keep_uniform(sub["raw_flow_id"].to_numpy(), seed, stream)
    return sub[u < float(s)]


def stage_B(kept, reps, s):
    """Retained flows -> (device, window) gateway features (same estimator as models.gateway_obs)."""
    base8 = FLOW_COLUMNS; vol7 = SUM_COLUMNS
    inv = 1.0 / float(s) if s < 1.0 else 1.0
    frames = []
    for dname, ipcol in (("out", "src_ip"), ("in", "dst_ip")):
        v = kept[kept[ipcol].isin(reps)]
        g = v.groupby([v[ipcol].astype(str).rename("device_id"), v["window"].astype("int64")], observed=True)
        summ = g[base8].sum(); cnt = g.size()
        o = pd.DataFrame({f"{dname}_duration": summ["duration"] / cnt})
        for c in vol7:
            o[f"{dname}_{c}_sum"] = summ[c] * inv
        o[f"{dname}_count"] = cnt * inv
        frames.append(o)
    G = frames[0].join(frames[1], how="outer").fillna(0.0)
    for c in FEATS:
        if c not in G.columns:
            G[c] = 0.0
    G = G[FEATS].astype(np.float64).reset_index()
    G.columns = KEYS + FEATS
    return G


def stage_join(te, G):
    m = te[KEYS].merge(G, on=KEYS, how="left", validate="many_to_one")
    Gx = m[FEATS].to_numpy(np.float64)
    return ~np.isnan(Gx).any(axis=1), Gx


class GrivOnline:
    """Minimal objects of the GRIV online path: node classifier (predict), gateway classifier + calibration, tau_G."""

    def __init__(self, node_model, f_G, cal_G, tau_G: float):
        self.node_model, self.f_G, self.cal_G, self.tau_G = node_model, f_G, cal_G, float(tau_G)

    def node_predict(self, X):
        """= NodeBundle.predict (default class prediction). 1 = attack."""
        if len(X) == 0:
            return np.zeros(0, dtype=int)
        return np.asarray(self.node_model.predict(np.asarray(X, np.float64))).astype(int)

    def p_G(self, Gx):
        """= GatewayBundle.p_G (classifier probability -> calibration)."""
        if len(Gx) == 0:
            return np.zeros(0)
        return self.cal_G.transform(self.f_G.predict_proba(np.asarray(Gx, np.float64))[:, _pos(self.f_G)])

    def gateway_alert(self, pG, covered):
        """= Detector.gateway_alert: covered & p_G > tau_G."""
        return TH.alert(pG, self.tau_G, covered)


def griv_online(det) -> GrivOnline:
    if hasattr(det, "node_predict") and hasattr(det, "tau_G"):     # already a GrivOnline (isinstance fails in the subprocess)
        return det
    return GrivOnline(det.nb.model, det.gb.f_G, det.gb.cal_G, det.tau[("GRIV", det.q_main)])


def stage_C_griv(g, X, covered, Gx):
    """Stage C: node-report IDS decision + p_G of covered reports."""
    g = griv_online(g)
    pred = g.node_predict(X)
    pG = np.full(len(X), np.nan)
    if covered.any():
        pG[covered] = g.p_G(Gx[covered])
    return pred, pG


def stage_D_griv(g, covered, pred, pG):
    """Stage D: I_GRIV = NodeReportIDS benign & covered & p_G > tau_G."""
    return (np.asarray(pred) == 0) & griv_online(g).gateway_alert(pG, covered)


def online_path_griv(g, sub, te, reps, s, seed, stream):
    """Stage E: A -> B -> join -> C -> D. Returns I_GRIV."""
    g = griv_online(g)
    kept = stage_A(sub, s, seed, stream)
    G = stage_B(kept, reps, s)
    covered, Gx = stage_join(te, G)
    X = te[FEATS].to_numpy(np.float64)
    pred, pG = stage_C_griv(g, X, covered, Gx)
    return stage_D_griv(g, covered, pred, pG)


def stageE_payload(det, sub, te, reps, s, seed, stream) -> dict:
    """Subprocess input: GRIV objects, sampling settings, TEST flows and reports (keys + features)."""
    return dict(griv=griv_online(det), sub=sub, te=te[KEYS + FEATS].reset_index(drop=True), reps=sorted(reps),
                s=s, seed=seed, stream=stream)


def _subprocess_peak(det, sub, te, reps, s, seed, stream) -> int:
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "stageE.pkl")
        with open(p, "wb") as f:
            pickle.dump(stageE_payload(det, sub, te, reps, s, seed, stream), f)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run([sys.executable, "-m", "griv.cost", p], cwd=root, env=env, capture_output=True, text=True,
                           timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"stage-E subprocess failed: {r.stderr[-2000:]}")
        return int(json.loads(r.stdout.strip().splitlines()[-1])["peak_rss_bytes"])


def measure(d, cfg, nb, gb, det, seed, s) -> pd.DataFrame:
    cc = cfg["cost"]; W, R = int(cc["warmup"]), int(cc["repeats"])
    sp = nb.split; te = nb.H["TEST"].reset_index(drop=True); nrep = len(te)
    stream = cfg["gateway"]["stream"]
    fs = d.flow_store
    tw = set(map(int, sp.windows["TEST"])); reps = set(sp.reporters["TEST"])
    sub = fs[fs["window"].isin(tw)]
    sub = sub[sub["src_ip"].isin(reps) | sub["dst_ip"].isin(reps)]
    models = [nb.model, gb.f_G]
    old = [getattr(m, "n_jobs", None) for m in models]
    for m in models:
        if hasattr(m, "n_jobs"):
            m.n_jobs = 1
    try:
        kept = stage_A(sub, s, seed, stream)
        G = stage_B(kept, reps, s)
        covered, Gx = stage_join(te, G)
        X = te[FEATS].to_numpy(np.float64)
        g = griv_online(det)
        predg, pGg = stage_C_griv(g, X, covered, Gx)
        stages = (("A_gateway_flow_selection", lambda: stage_A(sub, s, seed, stream)),
                  ("B_feature_aggregation", lambda: stage_B(kept, reps, s)),
                  ("C_node_ids_and_gateway_prob", lambda: stage_C_griv(g, X, covered, Gx)),
                  ("D_griv_decision", lambda: stage_D_griv(g, covered, predg, pGg)),
                  ("E_griv_online_path", lambda: online_path_griv(g, sub, te, reps, s, seed, stream)))
        rows = []
        for stage, fn in stages:
            ts = _timeit(fn, W, R)
            rows.append(dict(path="GRIV", stage=stage, s=s, seed=seed, warmup=W, repeats=R, n_reports=nrep,
                             n_flows_test=len(sub), n_flows_kept=len(kept), median_ns=float(np.median(ts)),
                             p95_ns=float(np.percentile(ts, 95)),
                             per_report_median_us=float(np.median(ts)) / 1000.0 / max(1, nrep),
                             throughput_reports_per_s=max(1, nrep) / (float(np.median(ts)) / 1e9)))
        peak = _subprocess_peak(det, sub, te, reps, s, seed, stream)
    finally:
        for m, o in zip(models, old):
            if o is not None:
                m.n_jobs = o
    t = pd.DataFrame(rows)
    t["stageE_subprocess_peak_rss_bytes"] = np.where(t["stage"] == "E_griv_online_path", peak, np.nan)
    env = dict(python=sys.version.split()[0], platform=platform.platform(), processor=platform.processor(),
               cpu_count=os.cpu_count(), numpy=np.__version__, pandas=pd.__version__,
               sklearn=__import__("sklearn").__version__,
               omp_num_threads=os.environ.get("OMP_NUM_THREADS", ""), mkl_num_threads=os.environ.get("MKL_NUM_THREADS", ""))
    for k, v in env.items():
        t[k] = v
    return t


if __name__ == "__main__":       # stage-E subprocess: load input -> run stage E once -> report peak RSS
    sys.dont_write_bytecode = True
    with open(sys.argv[1], "rb") as f:
        a = pickle.load(f)
    g = a["griv"]
    for m in (g.node_model, g.f_G):
        if hasattr(m, "n_jobs"):
            m.n_jobs = 1
    online_path_griv(g, a["sub"], a["te"], set(a["reps"]), a["s"], a["seed"], a["stream"])
    print(json.dumps({"peak_rss_bytes": peak_rss_bytes()}))
