#!/usr/bin/env python3
"""Single-command inference: chunk_id, clinical_entity [, chunk_text] -> icd10cm_code, snomed_code.

Modes
-----
auto    (default) Reproduce the submitted predictions exactly from stored chooser
        picks (results/test_picks_permute8.csv) joined through the shortlist metadata.
kev     Live two-stage run: final shortlists (data/test_shortlist.jsonl) + Kev server
        with permutation ensembling over option orders.
engine  Deterministic mapping engine (scripts/04_run_engine.py) - fully offline once
        the retrieval artifacts are built (see README).

Examples
--------
python predict.py --input data/test_input.csv --output predictions/test_predictions.csv
python predict.py --input data/test_input.csv --output /tmp/live.csv --mode kev \
    --endpoint http://127.0.0.1:8013 --nperm 8
python predict.py --input data/test_input.csv --output /tmp/engine.csv --mode engine
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parent
SCHEMA = ["chunk_id", "clinical_entity", "icd10cm_code", "snomed_code"]


def read_input(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        import pandas as pd

        df = pd.read_parquet(p)
        return df.to_dict("records")
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def _row_key(r: dict) -> tuple[str, str]:
    return (str(r.get("chunk_id", "")), str(r.get("clinical_entity", r.get("entity", ""))))


def write_predictions(path: str, rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SCHEMA})
    print(f"[predict] wrote {out} rows={len(rows)}")


def load_shortlist(jsonl: str, meta: str):
    rows = [json.loads(l) for l in open(jsonl)]
    metas = [json.loads(l) for l in open(meta)]
    return rows, metas


def meta_index(metas: list[dict]) -> dict:
    idx = {}
    for i, m in enumerate(metas):
        idx[(str(m.get("chunk_id", "")), str(m.get("entity", "")))] = i
    return idx


def mode_auto(args, inputs: list[dict]) -> list[dict]:
    picks_path = args.picks or str(REPO / "results" / "test_picks_permute8.csv")
    picks = {}
    with open(picks_path, newline="") as f:
        for r in csv.DictReader(f):
            picks[(int(r["i"]), r["system"])] = (r.get("pick") or "").strip()
    metas_path = args.meta or str(REPO / "data" / "test_shortlist.meta.jsonl")
    metas = [json.loads(l) for l in open(metas_path)]
    idx = meta_index(metas)
    fill = {}
    fill_path = args.engine_fill or str(REPO / "predictions" / "test_predictions_engine.csv")
    if Path(fill_path).exists():
        with open(fill_path, newline="") as f:
            for r in csv.DictReader(f):
                fill[str(r.get("chunk_id", ""))] = r
    out = []
    for r in inputs:
        k = _row_key(r)
        i = idx.get(k)
        icd = picks.get((i, "icd"), "") if i is not None else ""
        sno = picks.get((i, "snomed"), "") if i is not None else ""
        f = fill.get(str(r.get("chunk_id", "")), {})
        if not icd and f:
            icd = f.get("icd10cm_code", "")
        if not sno and f:
            sno = f.get("snomed_code", "")
        out.append({**r, "icd10cm_code": icd, "snomed_code": sno})
    return out


def mode_kev(args, inputs: list[dict]) -> list[dict]:
    endpoint = (args.endpoint or os.environ.get("KEV_ENDPOINT") or "").rstrip("/")
    if not endpoint:
        raise SystemExit("mode=kev needs --endpoint or KEV_ENDPOINT")
    rows, metas = load_shortlist(args.shortlist, args.meta)
    idx = meta_index(metas)
    model = args.model or os.environ.get("KEV_MODEL", "kev-latest")
    picks: dict[tuple[int, str], str] = {}
    for i, row in enumerate(rows):
        for qid, system in (("icd_pick", "icd"), ("snomed_pick", "snomed")):
            if qid not in row["questions"]:
                continue
            payload = {"request": {"state": row["state"], "model": model,
                                   "questions": row["questions"]},
                       "question": qid, "n_perm": args.nperm, "seed": 0}
            d = requests.post(f"{endpoint}/v1/systemone/permute", json=payload, timeout=300).json()
            keys = list(row["questions"][qid]["criteria"])
            mean = {k: sum(x["probabilities"].get(k, 0.0) for x in d["runs"]) / len(d["runs"])
                    for k in keys}
            picks[(i, system)] = max(keys, key=mean.get)
    out = []
    for r in inputs:
        i = idx.get(_row_key(r))
        out.append({**r,
                    "icd10cm_code": picks.get((i, "icd"), "") if i is not None else "",
                    "snomed_code": picks.get((i, "snomed"), "") if i is not None else ""})
    return out


def mode_engine(args, inputs: list[dict]) -> list[dict]:
    tmp_in = Path("/tmp/predict_input.csv")
    tmp_out = Path("/tmp/predict_engine_out.csv")
    with open(tmp_in, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["chunk_id", "clinical_entity", "chunk_text"])
        w.writeheader()
        for i, r in enumerate(inputs):
            w.writerow({"chunk_id": r.get("chunk_id", i),
                        "clinical_entity": r.get("clinical_entity", r.get("entity", "")),
                        "chunk_text": r.get("chunk_text", r.get("context", ""))})
    cmd = [sys.executable, str(REPO / "scripts" / "04_run_engine.py"),
           "--input", str(tmp_in), "--output", str(tmp_out)]
    if args.config:
        cmd += ["--config", args.config]
    print("[predict] running engine:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    pred = {}
    with open(tmp_out, newline="") as f:
        for r in csv.DictReader(f):
            pred[str(r["chunk_id"])] = r
    out = []
    for i, r in enumerate(inputs):
        p = pred.get(str(r.get("chunk_id", i)), {})
        out.append({**r,
                    "icd10cm_code": p.get("predicted_icd10cm_code", ""),
                    "snomed_code": p.get("predicted_snomed_code", "")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["auto", "kev", "engine"], default="auto")
    ap.add_argument("--picks", default=None, help="picks CSV for mode=auto")
    ap.add_argument("--shortlist", default=str(REPO / "data" / "test_shortlist.jsonl"))
    ap.add_argument("--meta", default=None)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--nperm", type=int, default=8)
    ap.add_argument("--config", default=None)
    ap.add_argument("--engine-fill", default=None, help="engine predictions CSV used to fill rows without a chooser question")
    args = ap.parse_args()
    if args.mode == "auto" and args.meta is None:
        args.meta = str(REPO / "data" / "test_shortlist.meta.jsonl")

    inputs = read_input(args.input)
    if not inputs:
        raise SystemExit("empty input")
    fn = {"auto": mode_auto, "kev": mode_kev, "engine": mode_engine}[args.mode]
    rows = fn(args, inputs)
    missing = sum(1 for r in rows if not r.get("icd10cm_code") or not r.get("snomed_code"))
    if missing:
        print(f"[predict] warning: {missing} rows have an empty code "
              f"(row not present in the shortlist meta; use --mode engine to fill)")
    write_predictions(args.output, rows)


if __name__ == "__main__":
    main()
