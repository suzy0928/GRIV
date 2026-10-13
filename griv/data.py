"""Data loading, SHA-256 preflight, corpus-count assertion (Table 2) and temporal episodes.

The derived data files are read in place from the paths in configs/local_paths.yaml; nothing is copied.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO, TYPES, local_paths
from .provenance import json_hash, sha256_file

KEYS = ["device_id", "window"]


class DataPreflightError(RuntimeError):
    pass


@dataclass
class Data:
    o_all: pd.DataFrame            # node-window reports of all devices
    o: pd.DataFrame                # reports of the 19 monitored reporters
    vn: list                       # monitored reporters
    dom: dict                      # window -> dominant attack type (attack windows)
    gl: pd.Series                  # window -> global label (max over all devices)
    all_windows: np.ndarray        # all windows with at least one report (sorted)
    data_hash: str
    sha: dict
    paths: dict
    _fs: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def flow_store(self) -> pd.DataFrame:
        if self._fs is None:
            self._fs = load_flow_store(self.paths["flow_store_reporters.parquet"])
        return self._fs

    def window_type(self, w) -> np.ndarray:
        """Attack-type grouping label of a window: dominant type for attack windows, 'none' otherwise."""
        w = np.asarray(w, dtype=np.int64)
        glv = self.gl.reindex(w).fillna(0).to_numpy().astype(int)
        t = np.array([self.dom.get(int(x), "none") for x in w], dtype=object)
        return np.where(glv == 1, t, "none")


def load_flow_store(path) -> pd.DataFrame:
    fs = pd.read_parquet(path)
    fs["src_ip"] = fs["src_ip"].astype("category")
    fs["dst_ip"] = fs["dst_ip"].astype("category")
    return fs


def meta_content_hash(p) -> str:
    """SHA-256 of meta.json without the build time (elapsed_seconds), as canonical JSON."""
    m = json.load(open(p, encoding="utf-8"))
    return json_hash({k: v for k, v in m.items() if k != "elapsed_seconds"})


def verify_sha(cfg: dict) -> tuple[dict, dict]:
    """Stop if any input file differs from manifests/DATA_MANIFEST.json. meta.json passes if its file hash matches or,
    for a file rebuilt with data_prep/build_derived_data.py, if its content without elapsed_seconds matches.
    Returns the verified manifest hashes (so data_hash is the same for released and rebuilt files) and the paths."""
    mf = json.load(open(REPO / cfg["data"]["manifest"], encoding="utf-8"))
    man = mf["sha256"]
    lp = local_paths(cfg)["files"]
    for name, expected in man.items():
        p = Path(lp[name])
        if not p.exists():
            raise DataPreflightError(f"data file missing: {p}")
        a = sha256_file(p)
        if a == expected:
            continue
        if name == "meta.json" and meta_content_hash(p) == mf["meta_json_content_sha256"]:
            continue
        raise DataPreflightError(f"SHA-256 mismatch for {name}: {a} != {expected}")
    return dict(man), lp


def corpus_counts(o_all: pd.DataFrame, vn, dom, n_raw_flows: int) -> dict:
    """Table 2 counts: flow records, reporters, node-window reports (benign / attack-bearing / by attack type)."""
    o = o_all[o_all["device_id"].astype(str).isin(set(vn))]
    nl = o["node_label"].astype(int)
    atk = o[nl == 1]
    t = atk["window"].astype(int).map(lambda w: dom.get(int(w), "none"))
    by = {k: int(v) for k, v in t.value_counts().sort_index().items()}
    return {"raw_flows": int(n_raw_flows), "reporters": int(len(set(vn))), "reports": int(len(o)),
            "benign": int((nl == 0).sum()), "attack": int((nl == 1).sum()), "by_type": by}


def assert_corpus(counts: dict, expected: dict):
    if counts != expected:
        raise DataPreflightError(f"corpus counts (Table 2) differ: {counts} != {expected}")


def episodes(dom: dict, gl: pd.Series, gap_min: int) -> dict:
    """Temporal episodes per attack type: attack windows of a type in time order; a gap > gap_min windows starts a new
    episode. Returns {type: [(start_window, end_window, n_windows), ...]}."""
    out = {t: [] for t in TYPES}
    atk_w = sorted(int(w) for w in gl.index if int(gl[w]) == 1)
    by: dict = {}
    for w in atk_w:
        by.setdefault(dom.get(w, "none"), []).append(w)
    for t, ws in by.items():
        if t not in out:
            out[t] = []
        cur = [ws[0]]
        for w in ws[1:]:
            if w - cur[-1] > gap_min:
                out[t].append((cur[0], cur[-1], len(cur))); cur = [w]
            else:
                cur.append(w)
        out[t].append((cur[0], cur[-1], len(cur)))
    return out


def n_raw_flows(d: "Data") -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(d.paths["flow_store_reporters.parquet"]).metadata.num_rows


def load(cfg: dict, *, verify: bool = True, check_corpus: bool = True) -> Data:
    if verify:
        sha, lp = verify_sha(cfg)
    else:
        lp = local_paths(cfg)["files"]
        sha = json.load(open(REPO / cfg["data"]["manifest"], encoding="utf-8"))["sha256"]
    o_all = pd.read_parquet(lp["o_it_R2.parquet"])
    o_all["device_id"] = o_all["device_id"].astype(str)
    o_all["window"] = o_all["window"].astype(np.int64)
    meta = json.load(open(lp["meta.json"], encoding="utf-8"))
    vn = sorted(str(x) for x in meta["valid_nodes"])
    domdf = pd.read_parquet(lp["window_dominant.parquet"])
    dom = {int(w): str(t) for w, t in zip(domdf.window, domdf.dominant_attack_type)}
    gl = o_all.groupby("window")["global_label"].max().astype(int)
    o = o_all[o_all["device_id"].isin(set(vn))].sort_values(KEYS).reset_index(drop=True)
    d = Data(o_all=o_all, o=o, vn=vn, dom=dom, gl=gl, all_windows=np.array(sorted(gl.index.astype(np.int64))),
             data_hash=json_hash(sha), sha=sha, paths=lp)
    if check_corpus:
        assert_corpus(corpus_counts(o_all, vn, dom, n_raw_flows(d)), cfg["data"]["corpus_assert"])
    return d
