"""Provenance stamped on every output: code hash, git commit (if any), config and data hashes, run id, environment."""
from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE_DIRS = ("griv", "configs")


def code_files(root: Path = REPO) -> list[str]:
    out = []
    for d in CODE_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                out.append(str(p.relative_to(root)).replace("\\", "/"))
    return sorted(out)


def code_hash(root: Path = REPO) -> str:
    """SHA-256 over (relative path, NUL, content, NUL) of every file in griv/ and configs/, in path order."""
    h = hashlib.sha256()
    for rel in code_files(root):
        h.update(rel.encode()); h.update(b"\x00"); h.update((root / rel).read_bytes()); h.update(b"\x00")
    return h.hexdigest()


def git_state(root: Path = REPO) -> dict:
    """Commit and dirty flag when the code is in a git checkout; 'none' otherwise."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise OSError
        st = subprocess.run(["git", "status", "--porcelain", "--", *CODE_DIRS], cwd=root, capture_output=True, text=True,
                            timeout=30)
        return {"git_commit": r.stdout.strip(), "git_dirty": bool(st.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "none", "git_dirty": None}


def env_info() -> dict:
    import numpy, pandas, scipy, sklearn
    return {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__, "platform": platform.platform(),
            "processor": platform.processor()}


def make_run_id(cfg: dict) -> str:
    return f"{cfg['profile']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{cfg['_hash'][:8]}"


def stamp(cfg: dict, data_hash: str, run_id: str) -> dict:
    g = git_state()
    return {"run_id": run_id, "git_commit": g["git_commit"], "git_dirty": g["git_dirty"], "code_hash": code_hash(),
            "config_hash": cfg["_hash"], "config_path": cfg["_path"], "data_hash": data_hash,
            "time": datetime.datetime.now().isoformat(timespec="seconds"), "env": env_info()}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def json_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
