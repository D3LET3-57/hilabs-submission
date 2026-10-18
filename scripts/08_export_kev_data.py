"""Export candidate sets as Kev/SystemOne JSONL decision data.

For each (entity, chunk) row and each system with gold:
  - score the row's candidate set with the fused confidence model
  - keep top-K codes as options (plus gold for train rows, so the label is valid)
  - question = choice over {code: display}
  - label = gold (train) / none (dev)

Option sets are cached to out/<split>_sets.pkl so the (expensive) candidate
generation runs once.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from term_map.adjudicate import confidence as conf_mod  # noqa: E402
from term_map.config import load_config  # noqa: E402
from term_map.pipeline import Engine, load_resources  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

INSTRUCTIONS = {
    "icd": ("You are a certified medical coder. Choose the ICD-10-CM code that best "
            "matches the clinical concept in the patient context."),
    "snomed": ("You are a certified medical terminologist. Choose the SNOMED CT concept "
               "that best matches the clinical concept in the patient context."),
}


def _train_helpers():
    spec = importlib.util.spec_from_file_location(
        "tp_train", REPO / "scripts" / "06_train_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--top-k", type=int, default=40, help="fused-score top-K (used when --per-source-k 0)")
    ap.add_argument("--per-source-k", type=int, default=20,
                    help="take top-k from EVERY source and union (better recall at similar size); 0 = fused only")
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--force-gold", action="store_true",
                    help="ensure gold is among options (required for training)")
    ap.add_argument("--drop-leaky", dest="drop_leaky", action="store_true", default=None,
                    help="drop same-component memory hits (default: on for train)")
    ap.add_argument("--no-drop-leaky", dest="drop_leaky", action="store_false")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", default=None, help="override candidate sets cache path")
    args = ap.parse_args()
    if args.drop_leaky is None:
        args.drop_leaky = args.split == "train"

    cfg = load_config()
    sp = pd.read_parquet(REPO / "data" / "processed" / "splits.parquet")
    df = sp[sp["split"] == args.split].reset_index(drop=True)
    df["clinical_entity"] = df["entity"]
    if args.limit:
        df = df.head(args.limit)

    cache_path = (Path(args.cache) if args.cache
                  else REPO / "out" / f"{args.split}_sets.pkl")
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if len(data) != len(df):
            raise SystemExit(f"cache {cache_path} has {len(data)} rows, df has {len(df)}")
    else:
        res = load_resources(cfg, enable_dense=True)
        eng = Engine(res, use_rules=False, use_policy=False,
                     sources=["exact", "memory", "lexical", "dense", "fuzzy", "crosswalk"],
                     use_rerank=False)
        data = eng.collect_candidates(df)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        print(f"[kev-export] cached sets -> {cache_path}")

    res = load_resources(cfg, enable_dense=False)

    tp = comp_of_idx = None
    if args.drop_leaky and res.train_lookup is not None:
        tp = _train_helpers()
        full = pd.read_parquet(REPO / "data" / "processed" / "splits.parquet")
        uf = tp._build_component_fn(full)
        tl = res.train_lookup
        comp_of_idx = {i: uf.find(("e", e))
                       for i, (e, _cid) in enumerate(zip(tl.entity_norm, tl.chunk_id))}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.jsonl")
    fmeta = open(meta_path, "w")
    stats = {"rows": 0, "questions": {"icd": 0, "snomed": 0},
             "gold_in_topk": {"icd": 0, "snomed": 0},
             "gold_rows": {"icd": 0, "snomed": 0}, "forced_gold": {"icd": 0, "snomed": 0}}
    with open(out_path, "w") as fout:
        for i, d in enumerate(data):
            row = df.iloc[i]
            state = {"entity": str(row["entity"]),
                     "context": str(d.get("context") or "")[:400]}
            meta = {"i": i, "chunk_id": str(row.get("chunk_id")), "entity": str(row["entity"])}
            questions = {}
            for system, gcol, qid in (("icd", "icd10cm_code", "icd_pick"),
                                      ("snomed", "snomed_code", "snomed_pick")):
                gold = row.get(gcol)
                if not isinstance(gold, str) or not gold.strip():
                    continue
                gold = gold.strip()
                cs = d["sets"][system]
                if args.drop_leaky:
                    row_comp = uf.find(("e", tp.basic(row["clinical_entity"])))
                    tp._drop_leaky_memory(cs, row_comp, comp_of_idx)
                if args.per_source_k:
                    top = []
                    seen = set()
                    for src_cands in cs.provenance.values():
                        for c in sorted(src_cands, key=lambda c: -c.score)[: args.per_source_k]:
                            if c.code not in seen:
                                seen.add(c.code)
                                top.append(c.code)
                else:
                    scores = conf_mod.fuse(cs, weights=None)
                    ranked = [c for c, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
                    top = ranked[: args.top_k]
                forced = False
                gold_in_options = gold in top
                if not gold_in_options:
                    if args.force_gold:
                        top = top + [gold]
                        forced = True
                    elif args.split == "train":
                        continue
                    else:
                        stats.setdefault("gold_missing_options", {}) \
                            .setdefault(system, 0)
                        stats["gold_missing_options"][system] += 1
                random.Random(i * 7919 + (0 if system == "icd" else 1)).shuffle(top)
                disp_map = res.term.icd_display if system == "icd" else res.term.snomed_display
                criteria = {}
                for code in top:
                    disp = disp_map.get(code) or next(
                        (c.display for c in cs.candidates.values() if c.code == code and c.display), None)
                    criteria[code] = (disp or code)[:180]
                q = {"type": "choice", "instructions": INSTRUCTIONS[system],
                     "criteria": criteria}
                if args.force_gold:
                    q["label"] = gold
                questions[qid] = q
                stats["questions"][system] += 1
                stats["gold_rows"][system] += 1
                if gold_in_options:
                    stats["gold_in_topk"][system] += 1
                if forced:
                    stats["forced_gold"][system] += 1
                meta.setdefault("gold", {})[system] = gold
                meta.setdefault("options", {})[system] = list(criteria)
            if questions:
                fout.write(json.dumps({"state": state, "questions": questions}) + "\n")
                fmeta.write(json.dumps(meta) + "\n")
                stats["rows"] += 1
    fmeta.close()

    print(f"[kev-export] {args.split}: rows={stats['rows']} questions={stats['questions']}")
    for system in ("icd", "snomed"):
        n = stats["gold_rows"][system]
        if n:
            print(f"[kev-export] {system}: gold-in-top{args.top_k}="
                  f"{stats['gold_in_topk'][system]}/{n} "
                  f"({stats['gold_in_topk'][system] / n:.1%}) "
                  f"forced={stats['forced_gold'][system]}")
    if stats.get("gold_missing_options"):
        print(f"[kev-export] gold missing from options (kept for eval): "
              f"{stats['gold_missing_options']}")
    print(f"[kev-export] wrote {out_path}")


if __name__ == "__main__":
    main()
