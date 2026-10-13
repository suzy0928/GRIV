"""Experiment runner: experiment x split x fold x seed x sampling rate x attack -> per-report shards.

All experiment differences are expressed in the config (configs/paper.yaml); there are no per-experiment scripts.
Shard path: results/<run_id>/shards/<experiment>/<split>/<fold>/seed=<seed>/s=<s>/<variant>/<classifier>__<attack>__<param>.parquet
Each shard is written to a temporary file, renamed, and followed by <name>.COMPLETE.json (file SHA-256, columns,
provenance). Completed shards are never recomputed; reusing a run directory with different code, config or data is
refused. Validity checks (a failure marks the whole run INVALID): O_GRIV equals NodeReportIDS | GatewayIDS for every report,
I_GRIV implies a benign node decision and an available gateway observation, submitted reports satisfy the integer and
relation constraints, gateway availability is unchanged by node-side manipulation, no NaN in required columns.

Usage: python -m griv.runner --config configs/paper.yaml --run-id paper [--experiments complete_concealment,adaptive]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
warnings.simplefilter("ignore", pd.errors.PerformanceWarning)

from . import config as C  # noqa: E402
from . import data as D  # noqa: E402
from . import methods as ME  # noqa: E402
from . import models as M  # noqa: E402
from . import provenance as P  # noqa: E402
from . import splits as S  # noqa: E402
from .attacks import adaptive as ADA  # noqa: E402
from .attacks import concealment as CON  # noqa: E402
from .attacks import constraints as CS  # noqa: E402
from .compromise import Compromise  # noqa: E402
from .features import FEATS  # noqa: E402
from .errors import ValidityError  # noqa: E402

SCHEMA_VERSION = "paper-shard-1"


class CompleteExistsError(RuntimeError):
    pass


# ============================================================================================ planning
def _grid(spec, full):
    return list(full) if spec == "grid" else list(spec)


def plan(cfg: dict, only=None) -> list[dict]:
    """config -> jobs. job = dict(E, split, seeds, s, classifier, variant, attack, param)."""
    jobs = []
    s_main = cfg["gateway"]["s_main"]
    for e, spec in cfg["experiments"].items():
        if only and e not in only:
            continue
        s_list = _grid(spec.get("s", [s_main]), cfg["s_grid"])
        var = spec.get("variant", "base")
        clf = spec.get("classifier", "rf")
        if spec.get("cost"):
            for s in s_list:
                jobs.append(dict(E=e, split=spec["splits"][0], s=s, classifier=clf, variant=var, attack="cost", param="-"))
            continue
        for a in spec["attacks"]:
            if a["kind"] == "concealment":
                params = [("concealment", f"alpha={x}") for x in _grid(a["alpha"], cfg["attacks"]["concealment"]["alpha_grid"])]
            elif a["kind"] == "adaptive":
                params = [("adaptive", f"step={x}") for x in a["step"]]
            else:
                raise ValueError(a["kind"])
            for sp in a.get("splits", spec["splits"]):
                for s in s_list:
                    for atk, prm in params:
                        jobs.append(dict(E=e, split=sp, s=s, classifier=clf, variant=var, attack=atk, param=prm))
    for j in jobs:
        j["seeds"] = C.rd_seeds(cfg) if j["split"] == "RD" else C.cw_seeds(cfg)
        if j["attack"] == "cost":
            j["seeds"] = C.cw_seeds(cfg)[:1]                 # cost: one run
    return jobs


# ============================================================================================ shard I/O
def shard_path(root: Path, j: dict, fold: str, seed: int) -> Path:
    return (root / "shards" / j["E"] / j["split"] / fold.replace(".", "_") / f"seed={seed}" / f"s={j['s']}" / j["variant"] /
            f"{j['classifier']}__{j['attack']}__{j['param']}.parquet")


def write_shard(df: pd.DataFrame, path: Path, prov: dict, extra: dict, overwrite: bool):
    comp = path.with_suffix(".COMPLETE.json")
    if comp.exists() and not overwrite:
        raise CompleteExistsError(f"refuse to overwrite completed shard {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    rec = dict(prov, file=path.name, sha256=P.sha256_file(path), n_rows=int(len(df)), columns=list(df.columns),
               schema_version=SCHEMA_VERSION, **extra)
    tmpc = comp.with_suffix(".tmp")
    tmpc.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    os.replace(tmpc, comp)


def is_complete(path: Path, prov: dict | None = None) -> bool:
    """A completed shard has a matching SHA-256; with prov, it must also come from the same code, config and data."""
    comp = path.with_suffix(".COMPLETE.json")
    if not comp.exists() or not path.exists():
        return False
    rec = json.loads(comp.read_text(encoding="utf-8"))
    if rec.get("sha256") != P.sha256_file(path):
        return False
    if prov is not None:
        for k in ("config_hash", "data_hash", "code_hash"):
            if rec.get(k) != prov.get(k):
                raise CompleteExistsError(f"existing shard {path.name} has a different {k} -- use a new --run-id")
    return True


# ============================================================================================ shard content
def _base_frame(d, nb, comp, te, seed, j, fold, prov):
    at = d.window_type(te["window"].to_numpy())
    rk = comp.rank_of(te["device_id"].to_numpy())
    return pd.DataFrame({
        "run_id": prov["run_id"], "git_commit": prov["git_commit"], "code_hash": prov["code_hash"],
        "config_hash": prov["config_hash"], "data_hash": prov["data_hash"],
        "experiment": j["E"], "split": nb.split.name, "split_id": nb.split.split_id, "fold": fold, "seed": seed,
        "s": j["s"], "classifier": j["classifier"], "variant": j["variant"], "attack": j["attack"], "attack_param": j["param"],
        "device_id": te["device_id"].astype(str).to_numpy(), "window": te["window"].astype(np.int64).to_numpy(),
        "attack_type": at, "node_label": te["node_label"].astype(int).to_numpy(),
        "global_label": te["global_label"].astype(int).to_numpy(), "partition": "TEST",
        "is_compromised_at_beta1": rk >= 0, "compromise_rank": rk, "n_eligible": comp.N,
        "n_test_windows": len(nb.split.windows["TEST"]), "n_test_reporters": len(nb.split.reporters["TEST"]),
        "is_targeted": comp.targeted(te, 1.0), "block_60min": te["window"].astype(np.int64).to_numpy() // 60,
    })


def _put_eval(f, ev, suffix=""):
    f[f"covered{suffix}"] = ev["covered"]
    f[f"post_rf{suffix}" if not suffix else "pre_rf"] = ev["pred"]
    f[f"rf_raw_prob_post{suffix}" if not suffix else "rf_raw_prob_pre"] = ev["raw_pN"]
    f[f"p_N{suffix}"] = ev["p_N"]
    for m in ME.METHODS:
        f[f"score_{m}{suffix}"] = ev[f"score_{m}"]
        f[f"I_{m}{suffix}"] = ev[f"I_{m}"]; f[f"O_{m}{suffix}"] = ev[f"O_{m}"]
    for a in ME.ARCH:
        f[f"A_{a}{suffix}"] = ev[f"A_{a}"]


def _check_validity(f, adaptive=False):
    if not adaptive:
        if not np.array_equal(f["O_GRIV"].to_numpy(), f["A_NodeOrGatewayIDS"].to_numpy()):
            raise ValidityError("O_GRIV != NodeReportIDS | GatewayIDS")
        if (f["I_GRIV"] & ~((f["post_rf"] == 0) & f["covered"])).any():
            raise ValidityError("I_GRIV without a benign node decision and an available gateway observation")
        if (f["A_GatewayIDS"] & ~f["covered"]).any():
            raise ValidityError("GatewayIDS alert on an uncovered report")
    if not np.array_equal(f["O_GRIV_honest"].to_numpy(), f["A_NodeOrGatewayIDS_honest"].to_numpy()):
        raise ValidityError("O_GRIV != NodeReportIDS | GatewayIDS on honest reports")
    for c in ("device_id", "window", "node_label", "covered", "pre_rf", "p_N_honest"):
        if f[c].isna().any():
            raise ValidityError(f"NaN in required column {c}")


def attack_shard(d, cfg, nb, gb, det, comp, j, fold, seed, prov):
    te = nb.H["TEST"].reset_index(drop=True)
    f = _base_frame(d, nb, comp, te, seed, j, fold, prov)
    X = te[FEATS].to_numpy(np.float64)
    ev_h = det.evaluate(te)
    det.check_finite(ev_h, ev_h["covered"], f"TEST honest ({j['attack']} {j['param']})")
    _put_eval(f, ev_h, suffix="_honest")
    f["p_G"] = ev_h["p_G"]; f["rf_raw_prob_G"] = ev_h["raw_pG"]
    for m in ME.METHODS:
        f[f"tau_{m}"] = det.tau[(m, det.q_main)]
    tg = f["is_targeted"].to_numpy()
    devs = te["device_id"].astype(str).to_numpy()
    if j["attack"] == "adaptive":
        return adaptive_shard(cfg, nb, det, j, fold, seed, f, te, X, ev_h)
    if j["attack"] != "concealment":
        raise ValueError(j["attack"])
    a = float(j["param"].split("=")[1])
    Q = X.copy()
    Q[tg] = CON.conceal(X[tg], CON.mu_matrix(devs[tg], nb.mu_table, nb.mu_global), a, FEATS)
    tol = float(cfg["attacks"]["modification_tolerance"])
    diff = np.abs(Q - X)
    f["was_modified"] = (diff > tol).any(axis=1) & tg
    f["n_changed_fields"] = np.where(tg, (diff > np.maximum(tol, 1e-6 * np.abs(X))).sum(axis=1), 0)
    f["attack_applicable"] = True
    ev = det.evaluate(te, X=Q)
    det.check_finite(ev, ev["covered"], f"TEST submitted ({j['attack']} {j['param']})")
    _put_eval(f, ev)
    f["validator_ok"] = CS.validator_ok(Q, FEATS)
    if tg.any() and CS.validate_matrix(Q[tg], FEATS):
        raise ValidityError(f"report validator failed ({j['param']})")
    if not f.loc[tg, "validator_ok"].all():
        raise ValidityError(f"validator failed on {int((~f.loc[tg, 'validator_ok']).sum())} submitted reports ({j['param']})")
    if not (ev["covered"] == ev_h["covered"]).all():
        raise ValidityError("gateway availability changed under node-side manipulation")
    _check_validity(f)
    return f.copy()


def adaptive_shard(cfg, nb, det, j, fold, seed, f, te, X, ev_h):
    """Adaptive concealment: per method m, the submitted report chosen against m (columns *_<m>)."""
    step = float(j["param"].split("=")[1])
    tg = f["is_targeted"].to_numpy()
    devs = te["device_id"].astype(str).to_numpy()
    MU = CON.mu_matrix(devs[tg], nb.mu_table, nb.mu_global)
    res, grid = ADA.search(det, X[tg], MU, devs[tg], ev_h["covered"][tg], ev_h["Gx"][tg], ev_h["p_G"][tg], FEATS, step)
    tol = float(cfg["attacks"]["modification_tolerance"])
    f["was_modified"] = False; f["n_changed_fields"] = 0; f["attack_applicable"] = True
    _put_eval(f, ev_h)                        # generic columns = honest reports; method columns below
    f["validator_ok"] = CS.validator_ok(X, FEATS)
    idx = np.where(tg)[0]
    f["adaptive_grid_n"] = len(grid); f["adaptive_step"] = step
    for m in ME.METHODS:
        r = res[m]
        a = np.zeros(len(te)); a[idx] = r["alpha"]
        wm = np.zeros(len(te), bool); wm[idx] = (np.abs(r["X"] - X[tg]) > tol).any(axis=1)
        prf = ev_h["pred"].copy(); prf[idx] = r["pred"]
        pn = ev_h["p_N"].copy(); pn[idx] = r["p_N"]
        sc = ev_h[f"score_{m}"].copy(); sc[idx] = r["score"]
        Im = ev_h[f"I_{m}"].copy(); Im[idx] = r["I"]
        f[f"adaptive_alpha_{m}"] = a; f[f"was_modified_{m}"] = wm; f[f"post_rf_{m}"] = prf; f[f"p_N_{m}"] = pn
        f[f"score_{m}"] = sc; f[f"I_{m}"] = Im; f[f"O_{m}"] = (prf == 1) | Im
        f[f"evaded_{m}"] = tg & (prf == 0) & ~Im
        if len(idx) and (~np.isfinite(np.asarray(r["score"], float)[ev_h["covered"][idx]])).any():
            raise ValidityError(f"{nb.split.name}/{fold}/seed={seed} adaptive {m}: non-finite covered scores")
        case = np.full(len(te), "-", dtype=object); case[idx] = r["case"]; f[f"adaptive_case_{m}"] = case
        if len(idx) and not CS.validator_ok(r["X"], FEATS).all():
            raise ValidityError(f"adaptive {m}: validator failed")
    # reference rows, each searched as its own target; the gateway observation is unchanged, so GatewayIDS keeps its honest value
    for row, ocol, icol in (("NodeReportIDS", "A_NodeReportIDS", None), ("NodeOrGatewayIDS", "A_NodeOrGatewayIDS", "A_GatewayIDS")):
        r = res[row]
        prf = ev_h["pred"].copy(); prf[idx] = r["pred"]
        Ir = np.zeros(len(te), bool) if icol is None else ev_h[icol].copy()
        if icol is not None:
            Ir[idx] = r["I"]
        a = np.zeros(len(te)); a[idx] = r["alpha"]
        f[f"adaptive_alpha_{row}"] = a; f[f"post_rf_{row}"] = prf
        wm = np.zeros(len(te), bool); wm[idx] = (np.abs(r["X"] - X[tg]) > tol).any(axis=1)
        f[f"was_modified_{row}"] = wm
        f[ocol] = (prf == 1) | Ir
        f[f"evaded_{row}"] = tg & ~f[ocol].to_numpy()
        if len(idx) and not CS.validator_ok(r["X"], FEATS).all():
            raise ValidityError(f"adaptive {row}: validator failed")
    if not np.array_equal(f["O_GRIV"].to_numpy(), f["A_NodeOrGatewayIDS"].to_numpy()):
        raise ValidityError("adaptive: O_GRIV != NodeReportIDS | GatewayIDS")
    if not np.array_equal(f["O_GRIV"].to_numpy(), (f["post_rf_GRIV"].to_numpy() == 1) | f["A_GatewayIDS"].to_numpy()):
        raise ValidityError("adaptive: O_GRIV != NodeReportIDS(submitted) | GatewayIDS")
    if (f["I_GRIV"] & ~((f["post_rf_GRIV"] == 0) & f["covered"])).any():
        raise ValidityError("adaptive: I_GRIV without a benign node decision and an available observation")
    if (f["A_GatewayIDS"] & ~f["covered"]).any():
        raise ValidityError("adaptive: GatewayIDS alert on an uncovered report")
    for m in list(ME.METHODS) + list(ADA.EXTRA_ROWS):
        if len(idx) and CS.validate_matrix(res[m]["X"], FEATS):
            raise ValidityError(f"adaptive {m}: report validator failed")
    # targets with covered & p_G > tau_G can never evade GRIV
    if f.loc[f["A_GatewayIDS"].to_numpy() & tg, "evaded_GRIV"].any():
        raise ValidityError("adaptive: a target with a gateway alert evaded GRIV")
    _check_validity(f, adaptive=True)
    return f.copy()


# ============================================================================================ driver
class Suite:
    def __init__(self, cfg_path, run_id=None, overwrite=False, only=None, log=print):
        self.cfg = C.load(cfg_path)
        self.only = only
        self.log = log
        self.overwrite = overwrite
        if overwrite and self.cfg["profile"] != "quick":
            raise RuntimeError("--overwrite is allowed only for the quick profile")
        self.d = D.load(self.cfg)                         # data SHA-256 preflight + Table 2 counts
        self.run_id = run_id or P.make_run_id(self.cfg)
        self.root = C.output_root(self.cfg) / self.run_id
        if (self.root / "INVALID.json").exists() and not overwrite:
            raise ValidityError(f"run directory {self.root} is marked INVALID -- use a new --run-id")
        self.root.mkdir(parents=True, exist_ok=True)
        self.prov = P.stamp(self.cfg, self.d.data_hash, self.run_id)
        old = self.root / "provenance.json"
        if old.exists() and not overwrite:
            o = json.loads(old.read_text(encoding="utf-8"))
            for k in ("config_hash", "data_hash", "code_hash"):
                if o.get(k) != self.prov.get(k):
                    raise CompleteExistsError(f"run {self.run_id}: existing {k} differs from the current one -- use a new --run-id")
        planned = sorted(set(j["E"] for j in plan(self.cfg, only)))
        if "cost" in planned and planned != ["cost"]:
            raise ValidityError(f"the cost experiment must run alone (planned {planned})")
        if planned == ["cost"] and self.cfg["profile"] == "paper":
            bad = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                   if os.environ.get(v) != "1"}
            if bad:
                raise ValidityError(f"cost measurement requires single-thread environment variables = '1' (got {bad})")
        self.splits, self._split_meta_paths, self.timings = {}, {}, []

    def split(self, name):
        if name not in self.splits:
            sp = S.make(self.d, self.cfg, name)
            S.check_invariants(sp, self.d, flows=True)
            self.splits[name] = sp
        return self.splits[name]

    def write_meta(self):
        (self.root / "provenance.json").write_text(json.dumps(self.prov, indent=1, default=str), encoding="utf-8")
        (self.root / "config_resolved.json").write_text(json.dumps(self.cfg, indent=1, default=str), encoding="utf-8")

    def run(self):
        try:
            return self._run()
        except ValidityError as e:
            (self.root / "INVALID.json").write_text(json.dumps(dict(error=str(e), provenance=self.prov), indent=1,
                                                               default=str), encoding="utf-8")
            raise

    def _run(self):
        cfg, d = self.cfg, self.d
        self.write_meta()
        jobs = plan(cfg, self.only)
        self.log(f"[plan] {len(jobs)} jobs ({sorted(set(j['E'] for j in jobs))})")
        cw = self.split("CW")
        groups = {}
        for j in jobs:
            groups.setdefault((j["split"], j["classifier"], j["variant"]), []).append(j)
        written = set()
        for (spname, clf, var), gj in groups.items():
            if spname == "RD":
                folds = S.select_rd_folds(d, cw, cfg["rd_folds"])
                sps = [(r, S.make_rd(d, cw, r)) for r in folds]
                for r, x in sps:
                    S.check_invariants(x, d)
                    if not (set(x.reporters["TEST"]) == {r} and r not in x.reporters["TRAIN"]):
                        raise ValidityError(f"RD fold {r}: reporter partition violated")
            else:
                sps = [("-", cw)]
            for fold, sp in sps:
                if (sp.name, fold) not in written:
                    out = self.root / "partitions" / f"{sp.name}__{fold.replace('.', '_')}"
                    out.mkdir(parents=True, exist_ok=True)
                    sp.partition_frame().to_parquet(out / "partitions.parquet", index=False)
                    S.partition_table(sp, d).to_csv(out / "partition_table.csv", index=False)
                    self._split_meta_paths[(sp.name, fold)] = (out / "split_meta.json", sp)
                    written.add((sp.name, fold))
                for seed in sorted(set(s for j in gj for s in j["seeds"])):
                    sj = [j for j in gj if seed in j["seeds"]]
                    todo = [j for j in sj if self.overwrite or not self._done(j, fold, seed)]
                    if not todo:
                        self.log(f"[skip] {spname} {fold} seed={seed}: all shards complete"); continue
                    t0 = time.perf_counter()
                    nb = M.build_node(d, cfg, sp, seed, classifier=clf, cw=cw, variant=var)
                    self.timings.append(dict(stage="build_node", split=spname, fold=fold, seed=seed, classifier=clf,
                                             variant=var, seconds=time.perf_counter() - t0))
                    comp = Compromise(d, sp, cfg["compromise"]["beta_grid"])
                    for s in sorted(set(j["s"] for j in todo)):
                        t1 = time.perf_counter()
                        gb = M.attach_gateway(d, cfg, nb, s, variant=var)
                        det = ME.Detector(cfg, nb, gb)
                        self.timings.append(dict(stage="attach_gateway+detector", split=spname, fold=fold, seed=seed,
                                                 s=s, variant=var, seconds=time.perf_counter() - t1))
                        self._write_group_tables(nb, gb, det, comp, spname, fold, seed, s, var, clf)
                        for j in [x for x in todo if x["s"] == s]:
                            self._run_job(j, nb, gb, det, comp, fold, seed)
        for pth, sp_ in self._split_meta_paths.values():
            pth.write_text(json.dumps(sp_.meta, default=str, indent=1), encoding="utf-8")
        tdir = self.root / "timings"; tdir.mkdir(exist_ok=True)
        tag = "-".join(self.only) if self.only else "all"
        pd.DataFrame(self.timings).to_csv(tdir / f"timings_{tag}_{os.getpid()}_{int(time.time())}.csv", index=False)
        self.log(f"[done] run_id={self.run_id} root={self.root}")
        return self.root

    def _cost_path(self, j, fold, seed):
        return shard_path(self.root, j, fold, seed).parent / "cost.parquet"

    def _done(self, j, fold, seed):
        if j["attack"] == "cost":
            return self._cost_path(j, fold, seed).exists()
        return is_complete(shard_path(self.root, j, fold, seed), self.prov)

    def _write_group_tables(self, nb, gb, det, comp, spname, fold, seed, s, var, clf):
        out = self.root / "tables" / spname / fold.replace(".", "_") / f"seed={seed}" / f"s={s}" / var / clf
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(det.tau_rows).assign(split=spname, fold=fold, seed=seed, s=s, variant=var, classifier=clf).to_parquet(
            out / "thresholds.parquet", index=False)
        te = nb.H["TEST"]
        comp.sna_table(te, self.d.window_type(te["window"].to_numpy()), seed).assign(split=spname, fold=fold).to_parquet(
            out / "sna.parquet", index=False)
        audit = dict(class_counts=dict(node=getattr(nb, "class_counts", {}), cal_G=gb.meta.get("class_counts_cal_G"),
                                       detector=det.class_counts),
                     node_audit=nb.audit, mu_meta=nb.mu_meta,
                     gateway_meta={k: v for k, v in gb.meta.items()},
                     compromise=dict(eligible=comp.eligible, ranking=comp.ranking, k=comp.k),
                     fallback=dict(feature_residual=gb.meta["feature_residual_fallback_devices"],
                                   benign_reference=nb.mu_meta["fallback_devices"]),
                     split_id=nb.split.split_id, provenance=self.prov)
        (out / "audit.json").write_text(json.dumps(audit, default=str, indent=1), encoding="utf-8")

    def _run_job(self, j, nb, gb, det, comp, fold, seed):
        t0 = time.perf_counter()
        if j["attack"] == "cost":
            from . import cost as CO
            tab = CO.measure(self.d, self.cfg, nb, gb, det, seed, j["s"])
            p = self._cost_path(j, fold, seed); p.parent.mkdir(parents=True, exist_ok=True)
            tab.to_parquet(p, index=False)
        else:
            path = shard_path(self.root, j, fold, seed)
            if is_complete(path, self.prov) and not self.overwrite:
                return
            f = attack_shard(self.d, self.cfg, nb, gb, det, comp, j, fold, seed, self.prov)
            write_shard(f, path, self.prov, dict(job=j, fold=fold, seed=seed, split_id=nb.split.split_id), self.overwrite)
        dt = time.perf_counter() - t0
        self.timings.append(dict(stage=f"job:{j['attack']}", E=j["E"], split=j["split"], fold=fold, seed=seed, s=j["s"],
                                 variant=j["variant"], param=j["param"], seconds=dt))
        self.log(f"[job] {j['E']} {j['split']} {fold} seed={seed} s={j['s']} {j['variant']} {j['classifier']} "
                 f"{j['attack']} {j['param']} {dt:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiments", default=None, help="comma-separated experiment names (default: all in the config)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--overwrite", action="store_true", help="quick profile only")
    a = ap.parse_args(argv)
    only = a.experiments.split(",") if a.experiments else None
    try:
        Suite(a.config, run_id=a.run_id, overwrite=a.overwrite, only=only, log=lambda m: print(m, flush=True)).run()
    except S.SplitCheckError as e:
        print(f"[STOP] {e}", flush=True); raise SystemExit(3)
    except ValidityError as e:
        print(f"[INVALID] {e}", flush=True); raise SystemExit(4)


if __name__ == "__main__":
    main()
