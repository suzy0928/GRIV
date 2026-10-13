"""Report features (Section 4.1, Supplementary Table S1).

A report summarizes the flow records of one device in one 60-s window, separately for outbound flows (device = source)
and inbound flows (device = destination). Per direction: mean duration, the seven additive byte and packet sums, and the
flow count -> 18 features. Node reports and gateway observations use the same features in the same order.
"""
from __future__ import annotations

FLOW_COLUMNS = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts",
                "dst_ip_bytes"]
SUM_COLUMNS = [c for c in FLOW_COLUMNS if c != "duration"]
DIRECTIONS = (("out", "src_ip"), ("in", "dst_ip"))


def feature_names() -> list[str]:
    names = []
    for d, _ in DIRECTIONS:
        names.append(f"{d}_duration")
        names += [f"{d}_{c}_sum" for c in SUM_COLUMNS]
        names.append(f"{d}_count")
    return names


FEATS = feature_names()
