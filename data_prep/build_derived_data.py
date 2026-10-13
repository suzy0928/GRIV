"""Build the five derived data files read by the engine from the TON_IoT network CSV files (Section 5.1, Table 2).

Outputs (in --out):
  o_it_R2.parquet               node-window reports of all devices: 18 features per report (per direction: mean duration,
                                seven additive sums, flow count), node label (any attack flow of the device in the window)
                                and global label (any attack flow in the window)
  meta.json                     monitored reporters (valid_nodes), feature order, cohort and build summary
  flow_store_reporters.parquet  flow records with a monitored reporter as source or destination; raw_flow_id = running
                                index over the cohort flows in file order (input of the gateway sampling hash)
  window_dominant.parquet       dominant attack type of every 60-s window with a report: the type with the most
                                attack-labeled flow records in the window over the whole dataset (all devices, each flow
                                counted once); ties by type name (alphabetical); "none" for windows without attack flows
  CACHE_MANIFEST.json           file list

Cohort: normal and the seven attack types; MITM and ransomware flow records are removed before windowing.
Monitored reporters: 192.168.x.x devices with at least 10 reports that appear as a flow source, excluding the attacker
hosts (192.168.1.28, 192.168.1.30-39), the gateway (192.168.1.1) and the broadcast address (192.168.1.255).

The SHA-256 of every output is compared with manifests/DATA_MANIFEST.json. meta.json records the build time
(elapsed_seconds), so its file hash differs; its content without elapsed_seconds is compared with the manifest, as in
the preflight of the engine (griv/data.py), which accepts the rebuilt files.
Usage: python data_prep/build_derived_data.py --raw <folder with Network_dataset_1.csv ... Network_dataset_23.csv>
                                              --out <output folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from griv.data import meta_content_hash  # noqa: E402
from griv.features import FEATS, FLOW_COLUMNS  # noqa: E402

FILES = [f"Network_dataset_{i}.csv" for i in range(1, 24)]
COHORT = ["normal", "ddos", "dos", "scanning", "xss", "injection", "password", "backdoor"]
WINDOW_SECONDS = 60
ATTACKER_HOSTS = ["192.168.1.28"] + [f"192.168.1.{i}" for i in range(30, 40)]
INFRASTRUCTURE = ["192.168.1.1", "192.168.1.255"]
MIN_REPORTS = 10
IP_PREFIX = "192.168."
REPRESENTATION = "semantic_volume_aware"     # label of the 18-feature representation stored in meta.json
CACHE_MANIFEST = ('{\n  "o_it_R2.parquet": {"path": "o_it_R2.parquet"},\n'
                  '  "flow_store_reporters.parquet": {"path": "flow_store_reporters.parquet"},\n'
                  '  "window_dominant.parquet": {"path": "window_dominant.parquet"},\n  "meta.json": {"path": "meta.json"}\n}\n')


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_flows(path: Path) -> pd.DataFrame:
    """Flow records of the cohort with a 60-s window index."""
    cols = ["ts", "src_ip", "dst_ip", "label", "type"] + FLOW_COLUMNS
    header = pd.read_csv(path, nrows=0)
    missing = [c for c in cols if c not in header.columns]
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["src_ip"] = df["src_ip"].astype(str).str.strip()
    df["dst_ip"] = df["dst_ip"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.lower().str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype("int8")
    for c in FLOW_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
    df = df[df["type"].isin(set(COHORT))].copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["window"] = (df["ts"] // WINDOW_SECONDS).astype("int64")
    return df


def sum_count(df: pd.DataFrame):
    """Per (device, window) and direction: sums of the flow columns, flow count and max flow label; per window: max label."""
    glob = df.groupby("window")["label"].max().reset_index().rename(columns={"label": "global_label"})
    parts = {}
    for d, ipcol in (("out", "src_ip"), ("in", "dst_ip")):
        v = df[["window", ipcol, "label"] + FLOW_COLUMNS].rename(columns={ipcol: "device_id"})
        s = v.groupby(["device_id", "window"])[FLOW_COLUMNS].sum().reset_index()
        c = v.groupby(["device_id", "window"]).size().reset_index(name=f"{d}_count")
        lb = v.groupby(["device_id", "window"])["label"].max().reset_index().rename(columns={"label": f"{d}_label"})
        parts[d] = s.merge(c, on=["device_id", "window"], how="left").merge(lb, on=["device_id", "window"], how="left")
    return parts["out"], parts["in"], glob


def device_windows(out_parts, in_parts, glob_parts) -> pd.DataFrame:
    """Node-window reports of all devices (18 features + labels)."""
    def combine(parts, cnt_col, lab_col):
        s = pd.concat(parts, ignore_index=True).groupby(["device_id", "window"])[FLOW_COLUMNS + [cnt_col, lab_col]].sum().reset_index()
        s[lab_col] = (s[lab_col] > 0).astype("int8")
        return s

    def features(s, d):
        cols = {"device_id": s["device_id"], "window": s["window"],
                f"{d}_count": s[f"{d}_count"].astype("int64"), f"{d}_label": s[f"{d}_label"]}
        cnt = s[f"{d}_count"].replace(0, pd.NA)
        for f in FLOW_COLUMNS:
            cols[f"{d}_{f}" if f == "duration" else f"{d}_{f}_sum"] = (s[f] / cnt) if f == "duration" else s[f]
        return pd.DataFrame(cols)

    o_it = features(combine(out_parts, "out_count", "out_label"), "out").merge(
        features(combine(in_parts, "in_count", "in_label"), "in"), on=["device_id", "window"], how="outer")
    for c in FEATS:
        o_it[c] = o_it[c].fillna(0).astype("float32")
    o_it["out_count"] = o_it["out_count"].fillna(0).astype("int32")
    o_it["in_count"] = o_it["in_count"].fillna(0).astype("int32")
    o_it["out_label"] = o_it["out_label"].fillna(0).astype("int8")
    o_it["in_label"] = o_it["in_label"].fillna(0).astype("int8")
    o_it["node_label"] = o_it[["out_label", "in_label"]].max(axis=1).astype("int8")
    glob = pd.concat(glob_parts, ignore_index=True).groupby("window")["global_label"].max().reset_index()
    o_it = o_it.merge(glob, on="window", how="left")
    o_it["global_label"] = o_it["global_label"].fillna(0).astype("int8")
    if o_it.duplicated(["device_id", "window"]).any():
        raise ValueError("duplicate (device_id, window) rows")
    if o_it[FEATS].isna().any().any():
        raise ValueError("NaN in feature columns")
    return o_it


def monitored_reporters(o_it: pd.DataFrame) -> list[str]:
    n = o_it.groupby("device_id").agg(n_windows=("window", "count"), total_out_count=("out_count", "sum")).reset_index()
    ok = (n["device_id"].astype(str).str.startswith(IP_PREFIX) & (n["n_windows"] >= MIN_REPORTS)
          & (n["total_out_count"] > 0) & ~n["device_id"].isin(set(ATTACKER_HOSTS) | set(INFRASTRUCTURE)))
    return n[ok]["device_id"].astype(str).sort_values().tolist()


def dominant(dom_parts, o_it) -> tuple[pd.DataFrame, int]:
    dd = pd.concat(dom_parts, ignore_index=True).groupby(["window", "type"], as_index=False)["n"].sum()
    top = dd.groupby("window")["n"].transform("max")
    n_tied = int((dd[dd["n"] == top].groupby("window").size() > 1).sum())
    dd = dd.sort_values(["n", "type"], ascending=[False, True], kind="mergesort").drop_duplicates("window")
    dom = dd[["window", "type"]].rename(columns={"type": "dominant_attack_type"})
    dom = o_it[["window"]].drop_duplicates().merge(dom, on="window", how="left").fillna({"dominant_attack_type": "none"})
    return dom, n_tied


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    raw, out = Path(a.raw).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    missing = [f for f in FILES if not (raw / f).exists()]
    if missing:
        raise SystemExit(f"missing raw files in {raw}: {missing}")

    t0 = time.time()
    out_parts, in_parts, glob_parts, flows = [], [], [], []
    offset = 0
    for fname in FILES:
        df = read_flows(raw / fname)
        if len(df):
            o, i, g = sum_count(df)
            out_parts.append(o); in_parts.append(i); glob_parts.append(g)
        df = df.reset_index(drop=True)
        df["raw_flow_id"] = np.arange(offset, offset + len(df), dtype=np.int64); offset += len(df)
        flows.append(df[["raw_flow_id", "window", "src_ip", "dst_ip", "label", "type"] + FLOW_COLUMNS])
        print(f"{fname}: cumulative cohort flows {offset}", flush=True)
    o_it = device_windows(out_parts, in_parts, glob_parts)
    valid = monitored_reporters(o_it)
    o_it.to_parquet(out / "o_it_R2.parquet", index=False)
    meta = {"valid_nodes": valid, "feature_cols": FEATS, "cohort": COHORT, "representation": REPRESENTATION,
            "num_device_window_rows": int(len(o_it)), "num_windows": int(o_it["window"].nunique()),
            "num_valid_nodes": int(len(valid)), "elapsed_seconds": time.time() - t0}
    with open(out / "meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
    print(f"o_it_R2: {len(o_it)} rows, {o_it['window'].nunique()} windows, {len(valid)} monitored reporters", flush=True)

    vn = set(valid); parts = []; dom_parts = []
    for df in flows:
        keep = df["src_ip"].astype(str).isin(vn) | df["dst_ip"].astype(str).isin(vn)
        parts.append(df.loc[keep, ["raw_flow_id", "window", "src_ip", "dst_ip", "label"] + FLOW_COLUMNS]
                     .rename(columns={"label": "flow_label"}))
        atk = df[df["label"] == 1]
        if len(atk):
            dom_parts.append(atk.groupby(["window", "type"]).size().reset_index(name="n"))
    store = pd.concat(parts, ignore_index=True)
    store.to_parquet(out / "flow_store_reporters.parquet", index=False)
    print(f"flow_store_reporters: {len(store)} flow records", flush=True)

    dom, n_tied = dominant(dom_parts, o_it)
    dom.to_parquet(out / "window_dominant.parquet", index=False)
    print(f"window_dominant: {len(dom)} windows, windows with tied top types: {n_tied}")
    print(dom["dominant_attack_type"].value_counts().to_string())
    (out / "CACHE_MANIFEST.json").write_bytes(CACHE_MANIFEST.encode())

    mf = json.load(open(ROOT / "manifests" / "DATA_MANIFEST.json", encoding="utf-8"))
    print("\nSHA-256 vs manifests/DATA_MANIFEST.json")
    for name, exp in mf["sha256"].items():
        got = sha256(out / name)
        print(f"  {'equal' if got == exp else 'DIFFERENT'}  {name}  {got}")
    got = meta_content_hash(out / "meta.json")
    print(f"  {'equal' if got == mf['meta_json_content_sha256'] else 'DIFFERENT'}  meta.json without elapsed_seconds  {got}")


if __name__ == "__main__":
    main()
