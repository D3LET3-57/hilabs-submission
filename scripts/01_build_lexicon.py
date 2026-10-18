#!/usr/bin/env python
"""M0 step 2: lexicons, train pools, component-based splits, eval inputs.

Outputs (data/processed/):
  splits.parquet          pair_id, entity, chunk_id, chunk_hash, split, gold codes
  abbrev_graph.json       normalized abbreviation -> [expansions]
  typo_pairs.json         {pairs: {variant: canonical}, vocab: {token: freq}}
  chunk_table.parquet     chunk_id, chunk_hash, chunk_text
  train_lookup.parquet    entity (train split only) -> codes for memory layer
  queries.parquet         qid, text, view for embedding precompute
eval/:
  dev_input.csv / test_input.csv / train_input.csv  (no gold columns)

Run: python scripts/01_build_lexicon.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from term_map.normalize import (  # noqa: E402
    basic_norm, clean_chunk, extract_context, split_abbr_description, strip_trailing_punct,
)
from term_map.splits import build_splits  # noqa: E402

OUT = REPO_ROOT / "data" / "processed"
EVAL = REPO_ROOT / "eval"

STOPWORDS = set(
    "a an and or of the to in with without for on by at from as is are was were be been "
    "no not other specified unspecified type due associated related".split()
)

CURATED_ABBREV: dict[str, list[str]] = {
    "ms": ["multiple sclerosis", "mitral stenosis", "morphine sulfate"],
    "ra": ["rheumatoid arthritis", "right atrium"],
    "pid": ["pelvic inflammatory disease", "prolapsed intervertebral disc"],
    "pt": ["physical therapy", "prothrombin time", "posterior tibial"],
    "af": ["atrial fibrillation", "amniotic fluid", "anterior fontanelle"],
    "as": ["aortic stenosis", "ankylosing spondylitis"],
    "ca": ["cancer", "calcium", "coronary artery"],
    "mi": ["myocardial infarction", "mitral insufficiency"],
    "pa": ["pulmonary artery", "pernicious anemia"],
    "ua": ["urinalysis", "unstable angina"],
    "bs": ["bowel sounds", "blood sugar"],
    "cva": ["cerebrovascular accident", "costovertebral angle tenderness"],
    "cp": ["chest pain", "cerebral palsy"],
    "ha": ["headache"],
    "cad": ["coronary artery disease"],
    "chf": ["congestive heart failure"],
    "ckd": ["chronic kidney disease"],
    "copd": ["chronic obstructive pulmonary disease"],
    "dm": ["diabetes mellitus"],
    "t1dm": ["type 1 diabetes mellitus"],
    "t2dm": ["type 2 diabetes mellitus"],
    "iddm": ["type 1 diabetes mellitus"],
    "niddm": ["type 2 diabetes mellitus"],
    "htn": ["hypertension"],
    "hld": ["hyperlipidemia"],
    "gerd": ["gastroesophageal reflux disease"],
    "uti": ["urinary tract infection"],
    "uri": ["upper respiratory infection"],
    "dvt": ["deep vein thrombosis"],
    "pe": ["pulmonary embolism"],
    "tia": ["transient ischemic attack"],
    "ards": ["acute respiratory distress syndrome"],
    "aki": ["acute kidney injury"],
    "bph": ["benign prostatic hyperplasia"],
    "hcc": ["hepatocellular carcinoma"],
    "ibd": ["inflammatory bowel disease"],
    "ibs": ["irritable bowel syndrome"],
    "sle": ["systemic lupus erythematosus"],
    "doe": ["dyspnea on exertion"],
    "sob": ["shortness of breath"],
    "gbm": ["glioblastoma multiforme"],
    "cabg": ["coronary artery bypass graft"],
    "pvd": ["peripheral vascular disease"],
    "esrd": ["end stage renal disease"],
    "osa": ["obstructive sleep apnea"],
    "bph": ["benign prostatic hyperplasia"],
    "rbbb": ["right bundle branch block"],
    "lbbb": ["left bundle branch block"],
    "wpw": ["wolff parkinson white syndrome"],
}


def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ abbrev graph
def mine_abbrev_graph(snomed: pd.DataFrame, icd: pd.DataFrame, train_pairs: pd.DataFrame,
                      train_split_entities: set[str]) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {}
    for df in (snomed, icd):
        for desc in df["description"].tolist():
            abbr, long = split_abbr_description(desc)
            if abbr and long:
                lst = graph.setdefault(abbr, [])
                ln = basic_norm(long)
                if ln and ln not in lst and len(lst) < 8:
                    lst.append(ln)
    # mined from train: abbreviation-looking entity -> display long form
    tp = train_pairs[train_pairs["entity_norm"].isin(train_split_entities)]
    for entity_norm, disp_sn, disp_ic in zip(tp["entity_norm"], tp["snomed_display"], tp["icd_display"]):
        if not entity_norm or " " in entity_norm or len(entity_norm) > 7:
            continue
        disp = disp_sn if isinstance(disp_sn, str) and disp_sn else disp_ic
        if not isinstance(disp, str) or not disp:
            continue
        abbr, long = split_abbr_description(disp)
        if abbr == entity_norm and long:
            lst = graph.setdefault(entity_norm, [])
            ln = basic_norm(long)
            if ln and ln not in lst and len(lst) < 8:
                lst.append(ln)
    # curated
    for abbr, exps in CURATED_ABBREV.items():
        key = basic_norm(abbr)
        lst = graph.setdefault(key, [])
        for e in exps:
            en = basic_norm(e)
            if en and en not in lst:
                lst.append(en)
    return {k: v for k, v in graph.items() if v}


# ------------------------------------------------------------------ typo pairs
_TOKEN_RE = re.compile(r"[a-z][a-z\-']{1,}")


def mine_typo_pairs(snomed: pd.DataFrame, icd: pd.DataFrame, train_entities: list[str]):
    from rapidfuzz import fuzz, process

    vocab: Counter = Counter()
    for desc in snomed["description"].tolist():
        for tok in _TOKEN_RE.findall(basic_norm(str(desc))):
            if len(tok) >= 3 and tok not in STOPWORDS:
                vocab[tok] += 1
    for desc in icd["description"].tolist():
        for tok in _TOKEN_RE.findall(basic_norm(str(desc))):
            if len(tok) >= 3 and tok not in STOPWORDS:
                vocab[tok] += 1
    vocab = {t: c for t, c in vocab.items() if c >= 3}
    keys = list(vocab.keys())

    pairs: dict[str, str] = {}
    for ent in set(basic_norm(e) for e in train_entities):
        for tok in _TOKEN_RE.findall(ent):
            if len(tok) < 5 or tok in vocab:
                continue
            m = process.extractOne(tok, keys, scorer=fuzz.ratio, score_cutoff=90)
            if m and m[1] >= 92 and vocab.get(m[0], 0) >= 5:
                pairs[tok] = m[0]
    return {"pairs": pairs, "vocab": vocab}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=str(OUT))
    ap.add_argument("--eval-dir", default=str(EVAL))
    args = ap.parse_args()
    out = Path(args.processed_dir)
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pairs = pd.read_parquet(out / "train_pairs.parquet")
    snomed = pd.read_parquet(out / "snomed_desc.parquet")
    icd = pd.read_parquet(out / "icd_desc.parquet")

    # entity norm + chunk ids
    pairs["entity_norm"] = pairs["entity"].map(lambda e: basic_norm(str(e)))
    chunks = pairs["chunk_text"].fillna("").map(clean_chunk)
    uniq_chunks = [c for c in pd.unique(chunks) if c]
    chunk_ids = {ch: f"c{i:05d}" for i, ch in enumerate(uniq_chunks)}
    pairs["chunk_clean"] = chunks
    pairs["pair_id"] = [f"p{i:05d}" for i in range(len(pairs))]
    # chunkless rows get unique pseudo-chunk ids so they never share components
    pairs["chunk_id"] = [
        chunk_ids[c] if c else f"noid_{pid}" for c, pid in zip(pairs["chunk_clean"], pairs["pair_id"])
    ]
    pairs["chunk_hash"] = [
        chunk_hash(t) if isinstance(t, str) and t else f"noid_{pid}"
        for t, pid in zip(pairs["chunk_text"].fillna(""), pairs["pair_id"])
    ]

    # ---- splits
    split_df = build_splits(pairs)
    split_df.to_parquet(out / "splits.parquet", index=False)
    counts = split_df["split"].value_counts().to_dict()
    print(f"[splits] {counts}")

    # leakage audit
    for s1, s2 in (("train", "dev"), ("train", "test"), ("dev", "test")):
        e1 = set(split_df[split_df["split"] == s1]["entity_norm"])
        e2 = set(split_df[split_df["split"] == s2]["entity_norm"])
        c1 = set(split_df[split_df["split"] == s1]["chunk_id"])
        c2 = set(split_df[split_df["split"] == s2]["chunk_id"])
        assert not (e1 & e2), f"entity leakage {s1}/{s2}: {len(e1 & e2)}"
        assert not (c1 & c2), f"chunk leakage {s1}/{s2}: {len(c1 & c2)}"
    print("[splits] leakage audit: 0 entity and 0 chunk overlaps")

    # ---- chunk table
    chunk_table = pd.DataFrame({
        "chunk_id": list(chunk_ids.values()),
        "chunk_hash": [chunk_hash(ch) for ch in uniq_chunks],
        "chunk_text": list(uniq_chunks),
    })
    chunk_table.to_parquet(out / "chunk_table.parquet", index=False)

    # ---- train lookup (memory layer, train split only)
    train_rows = split_df[split_df["split"] == "train"][
        ["entity", "entity_norm", "icd10cm_code", "snomed_code", "chunk_id"]
    ].copy()
    train_rows.to_parquet(out / "train_lookup.parquet", index=False)
    print(f"[memory] train lookup rows={len(train_rows)}")

    # ---- abbrev graph
    train_entities = set(train_rows["entity_norm"])
    graph = mine_abbrev_graph(snomed, icd, pairs, train_entities)
    (out / "abbrev_graph.json").write_text(json.dumps(graph, indent=1))
    print(f"[abbrev] graph keys={len(graph):,}")

    # ---- typo pairs
    typos = mine_typo_pairs(snomed, icd, list(train_rows["entity"]))
    (out / "typo_pairs.json").write_text(json.dumps(typos))
    print(f"[typo] pairs={len(typos['pairs'])} vocab={len(typos['vocab']):,}")

    # ---- eval inputs (no gold columns)
    for split in ("train", "dev", "test"):
        sub = split_df[split_df["split"] == split]
        inp = pd.DataFrame({
            "chunk_id": sub["chunk_id"],
            "clinical_entity": sub["entity"],
            "chunk_text": sub["chunk_text"],
        })
        inp.to_csv(eval_dir / f"{split}_input.csv", index=False)
        print(f"[eval] {split}_input.csv rows={len(inp)}")

    # ---- queries for embedding precompute
    qrows = []
    seen = set()
    for split in ("train", "dev", "test"):
        for r in split_df[split_df["split"] == split].itertuples():
            key = (r.entity_norm, r.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            ctx = extract_context(clean_chunk(r.chunk_text), str(r.entity), 300)
            qrows.append({"qid": r.pair_id, "text": str(r.entity).strip(), "view": "entity"})
            qrows.append({"qid": r.pair_id, "text": f"{str(r.entity).strip()} || {ctx}",
                          "view": "entity_context"})
    qdf = pd.DataFrame(qrows)
    qdf.to_parquet(out / "queries.parquet", index=False)
    print(f"[queries] rows={len(qdf)}")

    # ---- split manifest
    manifest = {
        "counts": {k: int(v) for k, v in counts.items()},
        "both_gold": {
            s: int(split_df[split_df["split"] == s]["both_gold"].sum()) for s in ("train", "dev", "test")
        },
        "seed": 17,
        "abbrev_keys": len(graph),
        "typo_pairs": len(typos["pairs"]),
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
