#!/usr/bin/env python
"""M1 step: build local indexes — lexical (TF-IDF word+char), fuzzy, FAISS dense.

Usage:
  python scripts/03_build_indexes.py --lexical-only            # no GPU needed
  python scripts/03_build_indexes.py --model gte_base          # FAISS for one model
  python scripts/03_build_indexes.py --model gte_base --validate dev

Outputs:
  artifacts/indexes/lexical_{system}.joblib
  artifacts/indexes/fuzzy_{system}.joblib
  artifacts/indexes/<model>/{system}/desc.index + codes.npy + meta.json
  eval/reports/recall_m1.json   (with --validate)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from term_map.candidates import lexical as lex_src  # noqa: E402
from term_map.candidates.fuzzy import FuzzyTermIndex  # noqa: E402
from term_map.config import load_config, resolve  # noqa: E402
from term_map.terminology import load_terminology  # noqa: E402


def load_desc(system: str):
    cfg = load_config()
    if system == "snomed":
        df = pd.read_parquet(resolve(cfg["terminology"]["snomed_parquet"]))
        if "snomed_code" in df.columns:
            df = df.rename(columns={"snomed_code": "code"})
    else:
        df = pd.read_parquet(resolve(cfg["terminology"]["icd_parquet"]))
        if "icd_code" in df.columns:
            df = df.rename(columns={"icd_code": "code"})
    return df["description"].tolist(), df["code"].tolist()


def build_lexical(system: str, out_root: Path, with_char: bool = True) -> None:
    descs, codes = load_desc(system)
    print(f"[lexical:{system}] building from {len(descs):,} descriptions ...", flush=True)
    t0 = time.time()
    idx = lex_src.build_lexical_index(descs, codes, system, with_char=with_char)
    out = out_root / f"lexical_{system}.joblib"
    idx.save(str(out))
    print(f"[lexical:{system}] saved {out} ({time.time() - t0:.1f}s)")


def build_fuzzy(system: str, out_root: Path) -> None:
    import joblib

    descs, codes = load_desc(system)
    print(f"[fuzzy:{system}] building ...", flush=True)
    idx = FuzzyTermIndex(descs, codes, system)
    out = out_root / f"fuzzy_{system}.joblib"
    joblib.dump(idx, out, compress=3)
    print(f"[fuzzy:{system}] keys={len(idx.keys):,} -> {out}")


def embeddings_dir(cfg, model: str, system: str) -> Path:
    return resolve(cfg["dense"]["embedding_dir"]) / model / system


def build_faiss(model: str, system: str, out_root: Path) -> bool:
    cfg = load_config()
    import faiss

    src = embeddings_dir(cfg, model, system)
    emb_p = src / "desc_emb.f16.npy"
    rows_p = src / "desc_rows.npy"
    codes_p = src / "desc_codes.npy"
    if not emb_p.exists():
        print(f"[faiss:{model}/{system}] missing {emb_p} — skipped")
        return False
    emb = np.load(emb_p)
    codes = np.load(codes_p, allow_pickle=True) if codes_p.exists() else None
    if codes is None and rows_p.exists():
        codes = np.load(rows_p, allow_pickle=True)
    print(f"[faiss:{model}/{system}] {emb.shape} building FlatIP ...", flush=True)
    xb = np.ascontiguousarray(emb.astype(np.float32))
    faiss.normalize_L2(xb)
    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    dst = out_root / model / system
    dst.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(dst / "desc.index"))
    np.save(dst / "codes.npy", np.asarray(codes))
    meta = {"model": model, "system": system, "n": int(xb.shape[0]), "dim": int(xb.shape[1]),
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[faiss:{model}/{system}] saved {dst} n={xb.shape[0]:,}")
    return True


def validate(cfg, split: str = "dev") -> dict:
    """Recall@k on the dev split for lexical / dense / union (gold = any desc of gold code)."""
    term = load_terminology(
        resolve(cfg["terminology"]["snomed_parquet"]),
        resolve(cfg["terminology"]["icd_parquet"]),
        resolve(cfg["terminology"]["map_parquet"]),
    )
    splits = pd.read_parquet(resolve(cfg["splits"]["path"]))
    dev = splits[splits["split"] == split]

    from term_map.normalize import basic_norm, extract_context, clean_chunk

    reports: dict = {"split": split, "systems": {}}
    for system in ("snomed", "icd"):
        code_col = "snomed_code" if system == "snomed" else "icd10cm_code"
        rows = dev[dev[code_col].notna()]
        idx_root = resolve(cfg["dense"]["index_dir"])
        lex_p = idx_root / f"lexical_{system}.joblib"
        lex = lex_src.LexicalIndex.load(str(lex_p)) if lex_p.exists() else None
        queries = []
        for r in rows.itertuples():
            q = basic_norm(str(r.entity))
            queries.append(q)
        golds = list(getattr(rows, code_col))
        n = len(rows)
        out = {"n": n}

        def recall_from(get_fn, ks=(10, 50, 100)):
            res = {f"recall@{k}": 0 for k in ks}
            hits100: list[bool] = []
            for i, (q, gold) in enumerate(zip(queries, golds)):
                cands = get_fn(i, q)
                for k in ks:
                    if gold in cands[:k]:
                        res[f"recall@{k}"] += 1
                hits100.append(gold in cands[:100])
            for k in ks:
                res[f"recall@{k}"] = round(res[f"recall@{k}"] / max(1, n), 4)
            return res, hits100

        union_hits = [False] * n
        sources = {}
        if lex is not None:
            cached = {}

            def get_lex(i, q):
                if i not in cached:
                    cached[i] = [c for c, _ in lex.query([q], top_k=100)[0]]
                return cached[i]

            r, hits = recall_from(get_lex)
            sources["lexical"] = r
            union_hits = [a or b for a, b in zip(union_hits, hits)]
        # dense per model
        from term_map.candidates.dense import DenseEncoder, DenseModel
        for mcfg in cfg["dense"]["models"]:
            m = DenseModel(name=mcfg["name"], hf_id=mcfg["hf_id"], dim=mcfg.get("dim", 768),
                           query_prefix=mcfg.get("query_prefix", ""))
            if not m.load(str(idx_root / mcfg["name"] / system)):
                continue
            qcache = resolve(cfg["dense"]["embedding_dir"]) / mcfg["name"] / "queries"
            enc = DenseEncoder(m.hf_id, m.query_prefix,
                               cache_dir=str(qcache) if qcache.exists() else None)
            emb = enc.encode(queries)
            hits = m.search(emb, 100)

            def get_dense(i, q, hits=hits):
                return [c for c, _ in hits[i]]

            r, hit_list = recall_from(get_dense)
            sources[f"dense_{mcfg['name']}"] = r
            union_hits = [a or b for a, b in zip(union_hits, hit_list)]
        out["sources"] = sources
        out["union_recall@100"] = round(sum(union_hits) / max(1, n), 4)
        reports["systems"][system] = out
        print(f"[validate:{system}] {json.dumps(out, indent=1)}")
    rep_dir = REPO_ROOT / "eval" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "recall_m1.json").write_text(json.dumps(reports, indent=2))
    print(f"[validate] wrote {rep_dir / 'recall_m1.json'}")
    return reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexical-only", action="store_true")
    ap.add_argument("--no-char", action="store_true", help="word TF-IDF only (floor repro)")
    ap.add_argument("--fuzzy-only", action="store_true")
    ap.add_argument("--model", action="append", default=[], help="embedding model name(s) for FAISS")
    ap.add_argument("--systems", default="snomed,icd")
    ap.add_argument("--validate", default=None, metavar="SPLIT", nargs="?", const="dev")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    out_root = Path(args.out) if args.out else resolve(cfg["dense"]["index_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    if args.lexical_only or args.fuzzy_only:
        for system in systems:
            if args.lexical_only:
                build_lexical(system, out_root, with_char=not args.no_char)
            if args.fuzzy_only:
                build_fuzzy(system, out_root)
        return 0

    if args.model:
        for model in args.model:
            for system in systems:
                build_faiss(model, system, out_root)

    if args.validate:
        validate(cfg, args.validate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
