"""Score dev candidate options with the fine-tuned UltraBERT judge; report exact match."""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/ultrabert_judge")
    ap.add_argument("--jsonl", default="out/kev/dev.jsonl")
    ap.add_argument("--meta", default="out/kev/dev.meta.jsonl")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", default="out/kev/dev_ultrabert_preds.jsonl")
    ap.add_argument("--dump-scores", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, trust_remote_code=True, dtype=torch.bfloat16)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    rows = [json.loads(x) for x in open(args.jsonl)]
    metas = [json.loads(x) for x in open(args.meta)]

    texts, cands, owners = [], [], []
    for i, (row, meta) in enumerate(zip(rows, metas)):
        state = f"{row['state']['entity']} || {row['state']['context']}"
        for system, qid in (("icd", "icd_pick"), ("snomed", "snomed_pick")):
            q = row["questions"].get(qid)
            if not q:
                continue
            for code in meta["options"][system]:
                texts.append(state)
                cands.append(f"{code} | {q['criteria'].get(code, code)}")
                owners.append((i, system, code))

    scores = []
    with torch.no_grad():
        for s in range(0, len(texts), args.batch):
            b = tok(texts[s:s + args.batch], cands[s:s + args.batch],
                    truncation=True, max_length=args.max_len,
                    padding=True, return_tensors="pt").to(device)
            logits = model(**b).logits.squeeze(-1).float().cpu()
            scores.extend(logits.tolist())

    best: dict[tuple[int, str], tuple[float, str]] = {}
    if args.dump_scores:
        with open(args.dump_scores, "w") as fs:
            for (i, system, code), sc in zip(owners, scores):
                fs.write(json.dumps({"i": i, "system": system, "code": code,
                                     "score": sc}) + "\n")
    for (i, system, code), sc in zip(owners, scores):
        key = (i, system)
        if key not in best or sc > best[key][0]:
            best[key] = (sc, code)

    stats = {s: {"n": 0, "correct": 0, "gold_in_options": 0} for s in ("icd", "snomed")}
    out = open(args.out, "w")
    for (i, system), (sc, pick) in sorted(best.items()):
        gold = metas[i]["gold"][system]
        present = gold in metas[i]["options"][system]
        st = stats[system]
        st["n"] += 1
        st["correct"] += int(pick == gold)
        st["gold_in_options"] += int(present)
        out.write(json.dumps({"i": i, "system": system, "gold": gold, "pick": pick,
                              "score": sc, "gold_in_options": present}) + "\n")
    out.close()

    for system in ("icd", "snomed"):
        st = stats[system]
        if not st["n"]:
            continue
        print(f" {system:6s} n={st['n']} acc={st['correct'] / st['n']:.3f} "
              f"({st['correct']}/{st['n']}) option_recall={st['gold_in_options'] / st['n']:.3f} "
              f"acc|present={st['correct'] / max(1, st['gold_in_options']):.3f}")
    print(f"[ultrabert-eval] wrote {args.out}")


if __name__ == "__main__":
    main()
