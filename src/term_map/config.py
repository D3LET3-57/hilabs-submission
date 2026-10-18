"""Config loading with repo-root-relative path resolution."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from .version import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    p = Path(path) if path else Path(os.environ.get("TERMMAP_CONFIG") or DEFAULT_CONFIG)
    cfg: dict[str, Any] = {}
    if p.exists():
        cfg = yaml.safe_load(p.read_text()) or {}
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def resolve(cfg_path: str) -> Path:
    """Resolve a config path relative to repo root (absolute paths pass through)."""
    p = Path(cfg_path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p
