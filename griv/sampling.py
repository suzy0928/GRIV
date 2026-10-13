"""Gateway flow sampling (Section 5.4, Algorithm S1).

Each flow record is kept iff u(raw_flow_id) < s, where u is a deterministic uniform value in [0, 1) computed from the
flow id and a run-specific salt. The same u is used for every s, so the sampled sets are nested across s and shared by
all methods.
"""
from __future__ import annotations

import hashlib

import numpy as np

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)
_M3 = np.uint64(0x94D049BB133111EB)


def salt(seed: int, stream: str) -> np.uint64:
    """blake2b(f"{seed}|{stream}", digest_size=8), read big-endian."""
    h = hashlib.blake2b(f"{int(seed)}|{stream}".encode(), digest_size=8).digest()
    return np.uint64(int.from_bytes(h, "big"))


def keep_uniform(raw_flow_id: np.ndarray, seed: int, stream: str) -> np.ndarray:
    """u = splitmix64 finalizer of (raw_flow_id XOR salt) * 0x9E3779B97F4A7C15; top 53 bits / 2^53."""
    z = (raw_flow_id.astype(np.uint64) ^ salt(seed, stream)) * _M1
    z ^= z >> np.uint64(30); z *= _M2; z ^= z >> np.uint64(27); z *= _M3; z ^= z >> np.uint64(31)
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)
