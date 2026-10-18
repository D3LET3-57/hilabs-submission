"""Input/output helpers: schema validation, prediction writing, per-row resume cache."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path

import pandas as pd

INPUT_ALIASES = {
    "chunk_id": ["chunk_id", "chunkid", "chunk"],
    "clinical_entity": ["clinical_entity", "entity", "term", "text", "clinical_term"],
    "chunk_text": ["chunk_text", "context", "chunk", "note_text"],
}


def _pick_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    out = {}
    for target, aliases in INPUT_ALIASES.items():
        for a in aliases:
            if a in cols:
                out[target] = cols[a]
                break
    missing = [t for t in ("clinical_entity",) if t not in out]
    if missing:
        raise ValueError(f"input missing required column(s): {missing}; columns={list(df.columns)}")
    res = pd.DataFrame()
    n = len(df)
    res["chunk_id"] = df[out["chunk_id"]].astype(str) if "chunk_id" in out else [str(i) for i in range(n)]
    res["clinical_entity"] = df[out["clinical_entity"]].astype(str)
    res["chunk_text"] = df[out["chunk_text"]].astype(str) if "chunk_text" in out else ""
    return res


def read_input(path: str | os.PathLike) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    return _pick_columns(df)


def write_predictions(df: pd.DataFrame, path: str | os.PathLike) -> None:
    p = Path(path)
    os.makedirs(p.parent, exist_ok=True)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def row_hash(chunk_id: str, entity: str, chunk_text: str) -> str:
    raw = f"{chunk_id}\x1f{entity}\x1f{chunk_text}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class RowCache:
    """SQLite per-row cache for resumable runs (deterministic skip of completed rows)."""

    def __init__(self, path: str | None):
        self.enabled = bool(path)
        self._conn = None
        self._lock = threading.Lock()
        if self.enabled:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS rows (key TEXT PRIMARY KEY, output TEXT, created_at REAL)"
            )
            self._conn.commit()

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        with self._lock:
            cur = self._conn.execute("SELECT output FROM rows WHERE key=?", (key,))
            row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, output: dict) -> None:
        if not self.enabled:
            return
        import time

        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO rows (key, output, created_at) VALUES (?,?,?)",
                (key, json.dumps(output), time.time()),
            )
            self._conn.commit()
