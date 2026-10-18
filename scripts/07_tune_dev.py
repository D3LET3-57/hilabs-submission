#!/usr/bin/env python
"""Offline dev tuning of decision parameters (policy blend, rerank weight/gate).

Collects dev candidates once and precomputes MedCPT rerank scores once (GPU),
then evaluates decision-parameter combos through the exact engine decision path
(real PolicyModel + RulePack). Submit via Slurm:

  sbatch scripts/slurm/tune_dev.sh
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from term_map.adjudicate import confidence as conf  # noqa: E402
from term_map.adjudicate.policy import PolicyModel  # noqa: E402
from term_map.adjudicate.rerank import CrossEncoderReranker  # noqa: E402
from term_map.adjudicate.rules import RulePack  # noqa: E402
from term_map.candidates.base import Candidate  # noqa: E402
from term_map.config import load_config, resolve  # noqa: E402
from term_map.normalize import fmt_icd  # noqa: E402
from term_map.pipeline import Engine, load_resources  # noqa: E402
from term_map.terminology import load_terminology  # noqa: E402

CACHE = REPO / "out" / "tune_cache.pkl"


def load_cache(cfg, device: str, batch_size: int):
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    sp = pd.read_parquet(REPO / "data/processed/splits.parquet")
    dev = sp[sp["split"] == "dev"].reset_index(drop=True)
    dev["clinical_entity"] = dev["entity"]
    res = load_resources(cfg, enable_dense=True)
    eng = Engine(res, use_rules=False, use_policy=False, sources=None, use_rerank=False)
    data = eng.collect_candidates(dev)
    rr = CrossEncoderReranker(device=device, top_n=int(cfg["rerank"]["top_n"]),
                              batch_size=batch_size)
    rerank_scores = {}
    print("reranking ...", flush=True)
    disp = {"snomed": res.term.snomed_display, "icd": res.term.icd_display}
    for d in data:
        key = str(d["row"]["pair_id"])
        entity = str(d["row"]["clinical_entity"])
        ctx = d["context"]
        rerank_scores[key] = {}
        for system in ("snomed", "icd"):
            cs = deepcopy(d["sets"][system])
            pre = conf.fuse(cs)
            tr = rr.rerank(system, cs, pre, entity, ctx, disp[system])
            rerank_scores[key][system] = [(t["code"], t["score"]) for t in tr]
    CACHE.write_bytes(pickle.dumps((data, rerank_scores)))
    return data, rerank_scores


def run_pass(data, term, rules, rerank_scores, policies, rw, gate):
    """policies: system -> PolicyModel (or None)."""
    hits = {"snomed": [0, 0], "icd": [0, 0]}
    for d in data:
        row = d["row"]
        key = str(row["pair_id"])
        for system, gcol in (("snomed", "snomed_code"), ("icd", "icd10cm_code")):
            gold = row.get(gcol)
            if not isinstance(gold, str) or not gold:
                continue
            hits[system][1] += 1
            cs = deepcopy(d["sets"][system])
            if rw > 0:
                pre = conf.fuse(cs)
                top2 = sorted(pre.items(), key=lambda kv: -kv[1])[:2]
                margin = (top2[0][1] - top2[1][1]) if len(top2) > 1 else 1.0
                if gate is None or margin <= gate:
                    for code, p in rerank_scores.get(key, {}).get(system, []):
                        cs.add(Candidate(code=code, source="rerank", score=p * (rw / 0.9)))
            dec = conf.decide(cs, policy=policies.get(system), rules=rules[system],
                              context=d["context"], entity=str(row["clinical_entity"]), term=term)
            code = dec.code or ""
            if system == "icd" and code:
                code = fmt_icd(code)
            if code == gold:
                hits[system][0] += 1
    out = {s: hits[s][0] / max(1, hits[s][1]) for s in hits}
    out["avg"] = (out["snomed"] + out["icd"]) / 2
    return out


def make_policies(blend, real=None):
    if blend <= 0:
        return {"snomed": None, "icd": None}
    if real is None:
        real = {}
    for system in ("snomed", "icd"):
        if system not in real:
            p = PolicyModel.load(str(REPO / f"artifacts/policy/{system}_policy.pkl"))
            p = deepcopy(p)
            real[system] = p
        if blend is not None:
            real[system].blend = blend
    return real


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config()
    data, rerank_scores = load_cache(cfg, args.device, args.batch_size)
    term = load_terminology(resolve(cfg["terminology"]["snomed_parquet"]),
                            resolve(cfg["terminology"]["icd_parquet"]),
                            resolve(cfg["terminology"]["map_parquet"]))
    rules = {s: RulePack(s, term) for s in ("snomed", "icd")}

    results = []

    def report(tag, acc, **params):
        results.append({"tag": tag, **params, **acc})
        print(f"{tag} -> snomed={acc['snomed']:.4f} icd={acc['icd']:.4f} avg={acc['avg']:.4f}",
              flush=True)

    print("phase 1: policy blend sweep, rerank off", flush=True)
    for blend in (0.3, 0.45, 0.6):
        acc = run_pass(data, term, rules, rerank_scores, make_policies(blend), 0.0, None)
        report("blend_only", acc, blend=blend, rerank_w=0.0, gate=None)

    print("phase 2: policy blend sweep, rerank on (w=0.9)", flush=True)
    combos = []
    for blend in (0.3, 0.45, 0.6):
        acc = run_pass(data, term, rules, rerank_scores, make_policies(blend), 0.9, None)
        report("blend_rerank", acc, blend=blend, rerank_w=0.9, gate=None)
        combos.append((blend, acc["avg"]))

    bb = max(combos, key=lambda t: t[1])[0]
    print(f"phase 3: gate sweep at blend={bb}, w=0.9", flush=True)
    for gate in (0.06, 0.12):
        acc = run_pass(data, term, rules, rerank_scores, make_policies(bb), 0.9, gate)
        report("gate", acc, blend=bb, rerank_w=0.9, gate=gate)

    results.sort(key=lambda r: -r["avg"])
    (REPO / "out" / "tune_results.json").write_text(json.dumps(results, indent=2))
    print("[tune] ranking:")
    for r in results:
        print(f"  {r['tag']}: blend={r['blend']} rw={r['rerank_w']} gate={r['gate']} "
              f"avg={r['avg']:.4f} (snomed={r['snomed']:.4f} icd={r['icd']:.4f})")


if __name__ == "__main__":
    main()
