#!/usr/bin/env python
"""M2: train the pointwise policy rankers on the train split.

Features come from the exact same candidate-assembly path used at inference
(Engine.collect_candidates). Memory candidates are made honest with
leave-one-component-out: any memory hit whose source train row belongs to the
same entity<->chunk component as the query row is dropped, so the trained model
cannot exploit split-internal label leakage (doc 04 section 2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from term_map.adjudicate import confidence as conf  # noqa: E402
from term_map.adjudicate.policy import PolicyModel, extract_features  # noqa: E402
from term_map.config import load_config, resolve  # noqa: E402
from term_map.pipeline import Engine, load_resources  # noqa: E402
from term_map.splits import UnionFind  # noqa: E402

GOLD_COL = {"icd": "icd10cm_code", "snomed": "snomed_code"}


def _build_component_fn(splits: pd.DataFrame):
    uf = UnionFind()
    for e, cid in zip(splits["entity_norm"], splits["chunk_id"]):
        uf.union(("e", e), ("c", str(cid)))
    return uf


def _drop_leaky_memory(cs, row_comp, comp_of_idx):
    changed = False
    for src in ("memory_exact", "memory_fuzzy"):
        cands = cs.provenance.get(src)
        if not cands:
            continue
        kept = []
        for c in cands:
            ti = (c.meta or {}).get("train_idx")
            if ti is not None and comp_of_idx.get(ti) == row_comp:
                changed = True
                continue
            kept.append(c)
        if kept:
            cs.provenance[src] = kept
        else:
            cs.provenance.pop(src, None)
    if changed:
        newc = {}
        for src, cands in cs.provenance.items():
            for c in cands:
                ex = newc.get(c.code)
                if ex is None or c.score > ex.score:
                    newc[c.code] = c
        cs.candidates = newc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/policy")
    ap.add_argument("--splits", default="data/processed/splits.parquet")
    ap.add_argument("--sources", default="exact,memory,lexical,dense,fuzzy,crosswalk")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=None)
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    splits = pd.read_parquet(resolve(args.splits))
    train = splits[splits["split"] == "train"].reset_index(drop=True)
    train["clinical_entity"] = train["entity"]
    uf = _build_component_fn(splits)

    cfg = load_config()
    res = load_resources(cfg, enable_dense=not args.no_dense)
    engine = Engine(res, use_rules=False, use_policy=False, use_llm=False,
                    sources=args.sources.split(","), use_rerank=args.rerank)

    # lookup row idx -> component root (lookup order == train parquet order)
    comp_of_idx = {}
    tl = res.train_lookup
    if tl is not None and tl.chunk_id is not None:
        for i, (e, cid) in enumerate(zip(tl.entity_norm, tl.chunk_id)):
            comp_of_idx[i] = uf.find(("e", e))

    print(f"[policy] collecting candidates for {len(train)} train rows ...")
    data = engine.collect_candidates(train, limit=args.limit)
    train = train.iloc[: len(data)]

    from sklearn.ensemble import HistGradientBoostingClassifier

    stats = {}
    for system in ("icd", "snomed"):
        gold_col = GOLD_COL[system]
        feats: list[np.ndarray] = []
        labels: list[int] = []
        n_rows = n_pos_rows = n_gold_missing = 0
        feat_names: list[str] = []
        for d in data:
            row = d["row"]
            gold = row.get(gold_col)
            if not isinstance(gold, str) or not gold:
                n_gold_missing += 1
                continue
            gold = str(gold).strip()
            if system == "icd":
                from term_map.normalize import fmt_icd

                gold = fmt_icd(gold)
            n_rows += 1
            cs = d["sets"][system]
            row_comp = uf.find(("e", basic(row["clinical_entity"])))
            _drop_leaky_memory(cs, row_comp, comp_of_idx)
            scores = conf.fuse(cs)
            if gold in cs.candidates:
                n_pos_rows += 1
            codes, X, feat_names = extract_features(
                cs, scores, str(row["clinical_entity"]), d["context"], res.term, system)
            if X.shape[0] == 0:
                continue
            y = np.array([1 if c == gold else 0 for c in codes], dtype=np.int8)
            feats.append(X)
            labels.extend(y.tolist())

        X = np.vstack(feats) if feats else np.zeros((0, len(feat_names)), dtype=np.float32)
        y = np.asarray(labels, dtype=np.int8)
        pos_w = float((y == 0).sum()) / max(1, int((y == 1).sum()))
        w = np.where(y == 1, pos_w, 1.0).astype(np.float32)
        print(f"[policy:{system}] rows={n_rows} (no gold: {n_gold_missing}) "
              f"gold-in-candidates={n_pos_rows} ({n_pos_rows / max(1, n_rows):.1%}) "
              f"candidates={X.shape[0]} positives={int(y.sum())} pos_weight={pos_w:.1f}")
        clf = HistGradientBoostingClassifier(
            max_iter=args.max_iter, learning_rate=args.learning_rate,
            max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=False,
        )
        clf.fit(X, y, sample_weight=w)
        pm = PolicyModel(system=system, model=clf, feature_names=feat_names, blend=0.45)
        out_path = resolve(args.out) / f"{system}_policy.pkl"
        pm.save(str(out_path))
        acc = float((clf.predict(X) == y).mean())
        stats[system] = {"rows": n_rows, "gold_in_candidates": n_pos_rows,
                         "candidates": int(X.shape[0]), "positives": int(y.sum()),
                         "train_point_acc": round(acc, 4)}
        print(f"[policy:{system}] saved {out_path} (train point acc={acc:.3f})")

    (resolve(args.out) / "policy_meta.json").write_text(json.dumps(stats, indent=2))
    print("[policy] done")


def basic(s: str) -> str:
    from term_map.normalize import basic_norm

    return basic_norm(s)


if __name__ == "__main__":
    main()
