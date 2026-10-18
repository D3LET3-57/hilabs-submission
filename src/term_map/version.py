"""Engine version stamp + terminology manifest hash."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

__version__ = "0.1.0"

REPO_ROOT = Path(__file__).resolve().parents[2]


def manifest_hash() -> str:
    """sha256 over the terminology manifest if present, else 'nomamifest'."""
    candidates = [
        REPO_ROOT / "data" / "processed" / "manifest.json",
    ]
    for p in candidates:
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            return h
    return "nomanifest"


def engine_version() -> str:
    return f"{__version__}+{manifest_hash()}"


def load_manifest() -> dict:
    p = REPO_ROOT / "data" / "processed" / "manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
