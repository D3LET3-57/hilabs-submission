"""Evaluate a Kev/SystemOne endpoint on exported decision JSONL + meta."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--meta", default=None, help="defaults to <jsonl>.meta.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8009")
    ap.add_argument("--model", default="kev-latest")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="optional predictions CSV")
    args = ap.parse_args()
    meta_path = args.meta or args.jsonl.replace(".jsonl", ".meta.jsonl")

    rows = [json.loads(line) for line in open(args.jsonl)]
    metas = [json.loads(line) for line in open(meta_path)]
    if args.limit:
        rows, metas = rows[: args.limit], metas[: args.limit]
    print(f"[kev-eval] {len(rows)} rows -> {args.url} model={args.model}")

    def ask(pair):
        row, meta = pair
        t0 = time.time()
        r = requests.post(f"{args.url}/v1/systemone",
                          json={"state": row["state"], "model": args.model,
                                "questions": row["questions"]},
                          timeout=300)
        r.raise_for_status()
        return r.json(), time.time() - t0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(ask, zip(rows, metas)))
    dt = time.time() - t0

    stats = {s: {"n": 0, "correct": 0, "gold_in_options": 0, "correct_when_present": 0}
             for s in ("icd", "snomed")}
    preds = []
    for (resp, _lat), meta in zip(results, metas):
        answers = resp.get("answers", {})
        for system, qid in (("icd", "icd_pick"), ("snomed", "snomed_pick")):
            if qid not in answers:
                continue
            gold = meta["gold"][system]
            opts = meta.get("short", meta["options"]).get(system, meta["options"].get(system, []))
            pick = answers[qid]["choice"]
            s = stats[system]
            s["n"] += 1
            present = gold in opts
            s["gold_in_options"] += int(present)
            s["correct"] += int(pick == gold)
            s["correct_when_present"] += int(present and pick == gold)
            preds.append({"i": meta["i"], "system": system, "gold": gold,
                          "pick": pick, "confidence": answers[qid].get("confidence"),
                          "gold_in_options": present})
    print(f"[kev-eval] {dt:.1f}s total ({sum(s['n'] for s in stats.values()) / dt:.1f} q/s)")
    for system in ("icd", "snomed"):
        s = stats[system]
        if not s["n"]:
            continue
        print(f" {system:6s} n={s['n']} acc={s['correct'] / s['n']:.3f} "
              f"({s['correct']}/{s['n']}) | option_recall={s['gold_in_options'] / s['n']:.3f} "
              f"| acc|present={s['correct_when_present'] / max(1, s['gold_in_options']):.3f}")
    if args.out:
        import csv

        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(preds[0]))
            w.writeheader()
            w.writerows(preds)
        print(f"[kev-eval] wrote {args.out}")


if __name__ == "__main__":
    main()
