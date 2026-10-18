"""Build pairwise (state, candidate) training data for the UltraBERT judge.

Source: out/kev/train.jsonl + .meta.jsonl (leak-filtered, source-diverse options).
Positive = (entity+context, code | display of gold).
Negatives = sampled from the same option list (hard negatives by construction).
Validation rows = deterministic 10% split by entity hash (no entity leakage).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def row_split(entity: str) -> str:
    return "val" if int(hashlib.sha256(entity.encode()).hexdigest()[:8], 16) % 10 == 0 else "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out/kev/train.jsonl")
    ap.add_argument("--meta", default="out/kev/train.meta.jsonl")
    ap.add_argument("--neg-per-pos", type=int, default=12)
    ap.add_argument("--out-dir", default="out/ultrabert")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.src)]
    metas = [json.loads(x) for x in open(args.meta)]
    assert len(rows) == len(metas)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    files = {"train": open(out / "pairs_train.jsonl", "w"),
             "val": open(out / "pairs_val.jsonl", "w")}
    stats = {"train": 0, "val": 0, "pos": 0, "neg": 0}
    for row, meta in zip(rows, metas):
        state = f"{row['state']['entity']} || {row['state']['context']}"
        split = row_split(row["state"]["entity"])
        f = files[split]
        for system, qid in (("icd", "icd_pick"), ("snomed", "snomed_pick")):
            q = row["questions"].get(qid)
            if not q:
                continue
            gold = meta["gold"][system]
            opts = meta["options"][system]
            criteria = q["criteria"]
            pos_text = f"{gold} | {criteria.get(gold, gold)}"
            f.write(json.dumps({"text": state, "candidate": pos_text, "label": 1,
                                "system": system, "i": meta["i"]}) + "\n")
            stats[split] += 1
            stats["pos"] += 1
            negs = [c for c in opts if c != gold]
            rng.shuffle(negs)
            for code in negs[: args.neg_per_pos]:
                f.write(json.dumps({"text": state,
                                    "candidate": f"{code} | {criteria.get(code, code)}",
                                    "label": 0, "system": system, "i": meta["i"]}) + "\n")
                stats[split] += 1
                stats["neg"] += 1
    for f in files.values():
        f.close()
    print(f"[pairs] train={stats['train']} val={stats['val']} "
          f"(pos={stats['pos']}, neg={stats['neg']}) -> {out}")


if __name__ == "__main__":
    main()
