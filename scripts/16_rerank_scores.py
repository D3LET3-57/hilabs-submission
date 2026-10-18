"""Score dev candidate options with a CrossEncoder reranker (Qwen3-Reranker / MedCPT).

Dumps (i, system, code, score) JSONL consumable by 14_build_kev_shortlist.py
via --judge-scores, and reports standalone argmax accuracy over the option set.

Scores are flushed incrementally to <out>.partial (pair order is deterministic),
so a killed run resumes where it left off; <out> is written once complete.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from sentence_transformers import CrossEncoder


def _load_done(partial: str) -> int:
    if not os.path.exists(partial):
        return 0
    good = []
    with open(partial) as fh:
        for line in fh:
            try:
                json.loads(line)
                good.append(line)
            except json.JSONDecodeError:
                break
    if len(good) != sum(1 for _ in open(partial)):
        with open(partial, "w") as fh:
            fh.writelines(good)
    return len(good)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", default="out/kev/dev_ub5.jsonl")
    ap.add_argument("--meta", default="out/kev/dev_ub5.meta.jsonl")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=512, help="pairs per predict chunk; each chunk is flushed to <out>.partial")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N pairs (smoke tests)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.jsonl)]
    metas = [json.loads(x) for x in open(args.meta)]

    pairs, owners = [], []
    for i, (row, meta) in enumerate(zip(rows, metas)):
        state = f"{row['state']['entity']} || {row['state']['context']}"
        for system, qid in (("icd", "icd_pick"), ("snomed", "snomed_pick")):
            q = row["questions"].get(qid)
            if not q:
                continue
            for code in meta["options"][system]:
                disp = q["criteria"].get(code, code)
                pairs.append((state, f"{code} | {disp}"))
                owners.append((i, system, code))
    if args.limit:
        pairs, owners = pairs[: args.limit], owners[: args.limit]
    total = len(pairs)

    partial = args.out + ".partial"
    done = _load_done(partial)
    print(f"[rerank-scores] {done}/{total} pairs already scored", flush=True)

    if done < total:
        pairs, owners = pairs[done:], owners[done:]
        model = CrossEncoder(args.model, device=args.device, max_length=args.max_len,
                             model_kwargs={"torch_dtype": torch.bfloat16})
        with open(partial, "a") as fh:
            for start in range(0, len(pairs), args.chunk):
                cp = pairs[start:start + args.chunk]
                co = owners[start:start + args.chunk]
                sc = model.predict(cp, batch_size=args.batch, show_progress_bar=False)
                for (i, system, code), s in zip(co, sc):
                    fh.write(json.dumps({"i": i, "system": system, "code": code,
                                         "score": float(s)}) + "\n")
                fh.flush()
                print(f"[rerank-scores] {done + start + len(cp)}/{total}", flush=True)
        del model
        torch.cuda.empty_cache()

    if done == total:
        with open(partial) as fh, open(args.out, "w") as fs:
            fs.writelines(fh)

    best: dict[tuple[int, str], tuple[float, str]] = {}
    n = 0
    with open(partial) as fh:
        for line in fh:
            d = json.loads(line)
            n += 1
            key = (d["i"], d["system"])
            if key not in best or d["score"] > best[key][0]:
                best[key] = (d["score"], d["code"])

    stats = {s: {"n": 0, "correct": 0, "gold_in_options": 0} for s in ("icd", "snomed")}
    for (i, system), (sc, pick) in sorted(best.items()):
        gold = metas[i]["gold"][system]
        st = stats[system]
        st["n"] += 1
        st["correct"] += int(pick == gold)
        st["gold_in_options"] += int(gold in metas[i]["options"][system])
    for system in ("icd", "snomed"):
        st = stats[system]
        if not st["n"]:
            continue
        print(f" {system:6s} n={st['n']} acc={st['correct'] / st['n']:.3f} "
              f"({st['correct']}/{st['n']}) option_recall={st['gold_in_options'] / st['n']:.3f} "
              f"acc|present={st['correct'] / max(1, st['gold_in_options']):.3f}")
    print(f"[rerank-scores] {n}/{total} scores; wrote {args.out}" if n == total
          else f"[rerank-scores] {n}/{total} scores; partial only", flush=True)


if __name__ == "__main__":
    main()
