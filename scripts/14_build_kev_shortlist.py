"""Build Kev/SystemOne JSONL from a reranked shortlist.

Stage 2 of the two-stage design: sources -> fused + encoder-judge blend ranking ->
top-K shortlist -> chooser (Kev/LLM/any). Keeps Kev's context small while keeping
shortlist recall high (~86/83% at K=20 on dev).
"""

from __future__ import annotations

import argparse
import json
import pickle
import random

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from term_map.adjudicate import confidence as conf_mod  # noqa: E402
from term_map.config import load_config  # noqa: E402
from term_map.pipeline import load_resources  # noqa: E402

INSTRUCTIONS = {
    "icd": ("You are a certified medical coder. Choose the ICD-10-CM code that best "
            "matches the clinical concept in the patient context."),
    "snomed": ("You are a certified medical terminologist. Choose the SNOMED CT concept "
               "that best matches the clinical concept in the patient context."),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev"])
    ap.add_argument("--k", type=int, default=None, help="uniform K for both systems")
    ap.add_argument("--k-icd", type=int, default=None)
    ap.add_argument("--k-snomed", type=int, default=None)
    ap.add_argument("--judge-scores", default=None)
    ap.add_argument("--judge-scores-icd", default=None, help="per-system override")
    ap.add_argument("--judge-scores-snomed", default=None, help="per-system override")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jsonl", default=None, help="override out/kev/{split}.jsonl")
    ap.add_argument("--meta-in", default=None, help="override out/kev/{split}.meta.jsonl")
    ap.add_argument("--sets-in", default=None, help="override out/{split}_sets.pkl")
    ap.add_argument("--disp-len", type=int, default=80)
    ap.add_argument("--judge-weight", type=float, default=0.2)
    ap.add_argument("--judge-weight-icd", type=float, default=None)
    ap.add_argument("--judge-weight-snomed", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    res = load_resources(load_config(), enable_dense=False)
    rows = [json.loads(x) for x in open(args.jsonl or f"out/kev/{args.split}.jsonl")]
    metas = [json.loads(x) for x in open(args.meta_in or f"out/kev/{args.split}.meta.jsonl")]
    with open(args.sets_in or f"out/{args.split}_sets.pkl", "rb") as f:
        data = pickle.load(f)
    judges = {}
    for system, per_sys in (("icd", args.judge_scores_icd), ("snomed", args.judge_scores_snomed)):
        path = per_sys or args.judge_scores
        if not path:
            raise SystemExit(f"missing judge scores for {system} (--judge-scores or --judge-scores-{system})")
        jd = {}
        for line in open(path):
            d = json.loads(line)
            if d["system"] == system:
                jd[(d["i"], d["system"], d["code"])] = d["score"]
        judges[system] = jd
    weights = {
        "icd": args.judge_weight_icd if args.judge_weight_icd is not None else args.judge_weight,
        "snomed": args.judge_weight_snomed if args.judge_weight_snomed is not None else args.judge_weight,
    }
    rng = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmeta = open(out_path.with_suffix(".meta.jsonl"), "w")
    stats = {"rows": 0, "q": {"icd": 0, "snomed": 0},
             "gold_in": {"icd": 0, "snomed": 0}, "skipped": 0}
    with open(out_path, "w") as fout:
        for i, (row, meta) in enumerate(zip(rows, metas)):
            state = row["state"]
            questions = {}
            for system, qid in (("icd", "icd_pick"), ("snomed", "snomed_pick")):
                gold = meta["gold"].get(system)
                if not gold:
                    continue
                opts = meta["options"][system]
                cs = data[i]["sets"][system]
                fused = conf_mod.fuse(cs)
                fz = np.array([fused.get(c, 0.0) for c in opts])
                jz = np.array([judges[system].get((i, system, c), -99.0) for c in opts])
                fzn = (fz - fz.mean()) / (fz.std() + 1e-9)
                jzn = (jz - jz.mean()) / (jz.std() + 1e-9)
                w = weights[system]
                order = np.argsort(-(w * jzn + (1 - w) * fzn))
                k = args.k if args.k else (args.k_icd if system == "icd" else args.k_snomed)
                short = [opts[j] for j in order[:k]]
                if gold not in short:
                    stats["skipped"] += 1
                    if args.split == "train":
                        continue
                rng.shuffle(short)
                disp_map = res.term.icd_display if system == "icd" else res.term.snomed_display
                crit = {c: (disp_map.get(c) or c)[: args.disp_len] for c in short}
                q = {"type": "choice", "instructions": INSTRUCTIONS[system],
                     "criteria": crit}
                if args.split == "train":
                    q["label"] = gold
                questions[qid] = q
                stats["q"][system] += 1
                if gold in short:
                    stats["gold_in"][system] += 1
                meta.setdefault("short", {})[system] = short
            if questions:
                fout.write(json.dumps({"state": state, "questions": questions}) + "\n")
                fmeta.write(json.dumps(meta) + "\n")
                stats["rows"] += 1
    fmeta.close()
    print(f"[shortlist] {args.split} k={args.k}: rows={stats['rows']} q={stats['q']} "
          f"skipped(no gold in shortlist)={stats['skipped']}")
    for s in ("icd", "snomed"):
        n = stats["q"][s]
        if n:
            print(f"[shortlist] {s}: gold_in_shortlist={stats['gold_in'][s]}/{n} "
                  f"({stats['gold_in'][s] / n:.1%})")
    print(f"[shortlist] wrote {out_path}")


if __name__ == "__main__":
    main()
