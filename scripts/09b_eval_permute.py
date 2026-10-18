#!/usr/bin/env python3
import argparse, csv, json, collections, requests

ap = argparse.ArgumentParser()
ap.add_argument("--jsonl", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--csv", required=True, help="single-pass eval csv (gold/present/pick)")
ap.add_argument("--url", required=True)
ap.add_argument("--nperm", type=int, default=5)
ap.add_argument("--out", required=True)
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.jsonl)]
base = {}
for r in csv.DictReader(open(args.csv)):
    base[(int(r["i"]), r["system"])] = r

stats = collections.defaultdict(collections.Counter)
n_parse = 0
with open(args.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["i", "system", "gold", "pick", "stable", "margin", "spread", "single_pick", "changed"])
    for i, row in enumerate(rows):
        for qid, system in (("icd_pick", "icd"), ("snomed_pick", "snomed")):
            if qid not in row["questions"]:
                continue
            b = base.get((i, system))
            if not b:
                continue
            payload = {"request": {"state": row["state"], "model": "kev-latest",
                                   "questions": row["questions"]},
                       "question": qid, "n_perm": args.nperm, "seed": 0}
            for attempt in range(6):
                try:
                    r = requests.post(args.url + "/v1/systemone/permute", json=payload, timeout=300)
                    break
                except requests.ConnectionError:
                    if attempt == 5:
                        raise
                    import time as _t
                    _t.sleep(5)
                except requests.HTTPError:
                    if attempt == 5:
                        raise
                    import time as _t
                    _t.sleep(5)
            if r.status_code == 404:
                raise SystemExit("permute endpoint not available")
            r.raise_for_status()
            d = r.json()
            runs = d["runs"]
            keys = list(row["questions"][qid]["criteria"])
            mean = {k: sum(x["probabilities"].get(k, 0.0) for x in runs) / len(runs) for k in keys}
            order = sorted(keys, key=lambda k: -mean[k])
            pick = order[0]
            margin = mean[order[0]] - (mean[order[1]] if len(order) > 1 else 0.0)
            spread = d.get("spread", {}).get(pick, 0.0)
            stable = bool(d.get("argmax_stable"))
            gold = (b["gold"] or "").strip()
            present = b["gold_in_options"] == "True"
            s = stats[system]
            s["n"] += 1
            s["present"] += present
            ok = pick == gold
            s["ok"] += ok
            s["ok_present"] += ok and present
            s["stable"] += stable
            s["ok_stable"] += ok and stable
            s["stable_n"] += stable
            s["changed"] += pick != (b["pick"] or "").strip()
            s["fixed"] += (pick == gold and (b["pick"] or "").strip() != gold)
            s["broke"] += (pick != gold and (b["pick"] or "").strip() == gold)
            w.writerow([i, system, gold, pick, stable, f"{margin:.4f}", f"{spread:.4f}", b["pick"], pick != (b["pick"] or "").strip()])

for system in ("icd", "snomed"):
    s = stats[system]
    n = s["n"] or 1
    print(f"== {system} n={s['n']} present={s['present']}")
    print(f"  permute-ens acc = {s['ok']/n:.3f} ({s['ok']}/{s['n']})")
    print(f"  acc|present     = {s['ok_present']/max(s['present'],1):.3f}")
    print(f"  stable rows     = {s['stable']}/{s['n']} ({s['stable']/n:.1%}); acc on stable = {s['ok_stable']/max(s['stable_n'],1):.3f}")
    print(f"  changed pick    = {s['changed']} (fixed {s['fixed']}, broke {s['broke']})")
