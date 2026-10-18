#!/usr/bin/env python
"""Evaluation harness: metrics, buckets, chapter report, error dump, sanity checks.

Usage:
  python scripts/05_eval.py --pred out/dev_m0.csv --split dev
  python scripts/05_eval.py --pred out/dev.csv --split dev --report-dir eval/reports

Reads gold from data/processed/splits.parquet; joins on (chunk_id, clinical_entity).
Reports acc_icd, acc_snomed, acc_joint, acc_avg + chapter/tag and bucket tables.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from term_map.config import load_config, resolve  # noqa: E402
from term_map.normalize import basic_norm, fmt_icd, strip_trailing_punct  # noqa: E402

ICD_RE = re.compile(r"^[A-Z][0-9][0-9A-Z](\.[0-9A-Z]{1,4})?$")
SNOMED_RE = re.compile(r"^[0-9]{4,18}$")

ICD_CHAPTERS = [
    ("A00-B99", "Infectious"),
    ("C00-D49", "Neoplasms"),
    ("D50-D89", "Blood/immune"),
    ("E00-E89", "Endocrine/metabolic"),
    ("F01-F99", "Mental"),
    ("G00-G99", "Nervous"),
    ("H00-H59", "Eye"),
    ("H60-H95", "Ear"),
    ("I00-I99", "Circulatory"),
    ("J00-J99", "Respiratory"),
    ("K00-K95", "Digestive"),
    ("L00-L99", "Skin"),
    ("M00-M99", "Musculoskeletal"),
    ("N00-N99", "Genitourinary"),
    ("O00-O9A", "Pregnancy"),
    ("P00-P96", "Perinatal"),
    ("Q00-Q99", "Congenital"),
    ("R00-R99", "Symptoms"),
    ("S00-T88", "Injury/poisoning"),
    ("V00-Y99", "External causes"),
    ("Z00-Z99", "Factors/health status"),
]


def icd_chapter(code: str) -> str:
    c = fmt_icd(code)
    if not c or not c[0].isalpha():
        return "other"
    from string import ascii_uppercase

    letter = c[0]
    idx = ascii_uppercase.index(letter) if letter in ascii_uppercase else -1
    if idx < 0:
        return "other"
    for rng, name in ICD_CHAPTERS:
        lo, hi = rng.split("-")
        if lo[0] <= letter <= hi[0]:
            return name
    return "other"


def snomed_tag(display: str) -> str:
    if not isinstance(display, str):
        return "unknown"
    m = re.search(r"\(([a-z][a-z ]+)\)\s*$", display.lower())
    return m.group(1) if m else "unknown"


def norm_code_icd(c) -> str:
    if pd.isna(c):
        return ""
    return fmt_icd(str(c).strip())


def norm_code_snomed(c) -> str:
    if pd.isna(c):
        return ""
    s = str(c).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def load_gold(split: str) -> pd.DataFrame:
    cfg = load_config()
    sp = pd.read_parquet(resolve(cfg["splits"]["path"]))
    sp = sp[sp["split"] == split].copy()
    return sp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--report-dir", default=str(REPO_ROOT / "eval" / "reports"))
    ap.add_argument("--context-chars", type=int, default=300)
    ap.add_argument("--sanity", action="store_true", default=True)
    ap.add_argument("--no-sanity", dest="sanity", action="store_false")
    args = ap.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.pred) if not args.pred.endswith(".parquet") else pd.read_parquet(args.pred)
    gold = load_gold(args.split)
    print(f"[eval] pred rows={len(pred)} gold rows={len(gold)}")

    # canonicalize
    pred = pred.copy()
    pred["clinical_entity"] = pred["clinical_entity"].astype(str)
    pred["key_entity"] = pred["clinical_entity"].map(lambda e: basic_norm(e))
    gold["key_entity"] = gold["entity"].map(lambda e: basic_norm(str(e)))
    pred["key"] = pred["chunk_id"].astype(str) + "\u241f" + pred["key_entity"]
    gold["key"] = gold["chunk_id"].astype(str) + "\u241f" + gold["key_entity"]

    dup = pred["key"].duplicated().sum()
    if dup:
        print(f"[eval] warning: {dup} duplicate prediction keys; keeping first")
        pred = pred.drop_duplicates("key")

    m = gold.merge(pred, on="key", how="left", suffixes=("", "_pred"))
    missing = m["predicted_icd10cm_code"].isna() & m["predicted_snomed_code"].isna()
    if missing.any():
        print(f"[eval] warning: {int(missing.sum())} gold rows have no prediction")

    m["pred_icd"] = m["predicted_icd10cm_code"].map(norm_code_icd)
    m["pred_snomed"] = m["predicted_snomed_code"].map(norm_code_snomed)
    m["gold_icd"] = m["icd10cm_code"].map(norm_code_icd)
    m["gold_snomed"] = m["snomed_code"].map(norm_code_snomed)

    has_icd = m["gold_icd"] != ""
    has_snomed = m["gold_snomed"] != ""
    m["hit_icd"] = (m["pred_icd"] == m["gold_icd"]) & has_icd
    m["hit_snomed"] = (m["pred_snomed"] == m["gold_snomed"]) & has_snomed

    n = len(m)
    acc_icd = m.loc[has_icd, "hit_icd"].mean() if has_icd.any() else 0.0
    acc_snomed = m.loc[has_snomed, "hit_snomed"].mean() if has_snomed.any() else 0.0
    both = has_icd & has_snomed
    acc_joint = (m.loc[both, "hit_icd"] & m.loc[both, "hit_snomed"]).mean() if both.any() else 0.0
    acc_avg = (acc_icd + acc_snomed) / 2

    summary = {
        "split": args.split,
        "rows": int(n),
        "rows_with_icd_gold": int(has_icd.sum()),
        "rows_with_snomed_gold": int(has_snomed.sum()),
        "rows_with_both_gold": int(both.sum()),
        "acc_icd": round(float(acc_icd), 4),
        "acc_snomed": round(float(acc_snomed), 4),
        "acc_joint": round(float(acc_joint), 4),
        "acc_avg": round(float(acc_avg), 4),
        "icd_errors": int((has_icd & ~m["hit_icd"]).sum()),
        "snomed_errors": int((has_snomed & ~m["hit_snomed"]).sum()),
    }
    print(json.dumps(summary, indent=2))

    # ------------------------------------------------------------ buckets
    cfg = load_config()
    tag_rows = []
    try:
        from term_map.terminology import load_terminology
        from rapidfuzz import fuzz

        term = load_terminology(
            resolve(cfg["terminology"]["snomed_parquet"]),
            resolve(cfg["terminology"]["icd_parquet"]),
            resolve(cfg["terminology"]["map_parquet"]),
        )
        sn_index = term.snomed_norm_index
        icd_index = term.icd_norm_index
        gold_disp = {"snomed": term.snomed_display, "icd": term.icd_display}
        have_term = True
    except Exception as e:  # pragma: no cover
        print(f"[eval] terminology unavailable for buckets: {e}")
        have_term = False

    try:
        abbr_graph = json.loads((resolve(cfg["lexicon"]["abbrev_graph"])).read_text())
        collide = set(list(abbr_graph.keys()))
    except Exception:
        collide = set()

    qual_re = re.compile(
        r"\b(possible|suspected|likely|history of|h/o|hx|mild|moderate|severe|left|right|bilateral|"
        r"acute|chronic|\d+\s*\+|\d-\d|\bone\b|\btwo\b)\b", re.I)
    train_gold_counts = {"icd": {}, "snomed": {}}
    sp = pd.read_parquet(resolve(cfg["splits"]["path"]))
    tr = sp[sp["split"] == "train"]
    train_gold_counts["icd"] = tr["icd10cm_code"].value_counts().to_dict()
    train_gold_counts["snomed"] = tr["snomed_code"].value_counts().to_dict()

    def bucket_tags(row) -> list[str]:
        tags = []
        ent = basic_norm(str(row["entity"]))
        ent_clean = strip_trailing_punct(ent)
        for system in ("snomed", "icd"):
            goldc = row["gold_snomed"] if system == "snomed" else row["gold_icd"]
            if not goldc:
                continue
            norm_idx = sn_index if system == "snomed" else icd_index
            if ent_clean in norm_idx and goldc in norm_idx.get(ent_clean, []):
                tags.append(f"exact_dictionary:{system}")
            if ent_clean not in norm_idx:
                tags.append(f"out_of_dictionary:{system}")
            # misspelling proxy: similarity of entity to gold display
            disp = gold_disp[system].get(goldc, "") if have_term else ""
            sim = fuzz.token_set_ratio(ent, basic_norm(disp)) / 100 if disp else 0.0
            if 0.60 < sim < 0.90:
                tags.append(f"misspelling_ocr:{system}")
            if len(goldc) and (train_gold_counts[system].get(goldc, 0) <= 2):
                tags.append(f"long_tail_gold:{system}")
            if system == "icd":
                tags.append(f"chapter:{icd_chapter(goldc)}")
            else:
                tags.append(f"tag:{snomed_tag(disp)}")
        if (not ent_clean) or (len(ent) <= 6 and " " not in ent):
            tags.append("abbreviation")
        if ent in collide or strip_trailing_punct(ent) in collide:
            tags.append("collision_prone_abbrev")
        if qual_re.search(ent):
            tags.append("qualifier_laden")
        if not has_icd_or_snomed(row):
            tags.append("single_gold_row")
        return tags

    def has_icd_or_snomed(row):
        return bool(row["gold_icd"]) and bool(row["gold_snomed"])

    m["buckets"] = m.apply(bucket_tags, axis=1)
    m["buckets_str"] = m["buckets"].map(lambda b: ";".join(b))

    rows = []
    all_tags = set()
    for b in m["buckets"]:
        all_tags.update(b)
    for tag in sorted(all_tags):
        mask = m["buckets"].map(lambda b, tag=tag: tag in b)
        sub = m[mask]
        sys_name = tag.split(":")[-1] if ":" in tag else None
        if sys_name == "icd":
            ok = sub["hit_icd"].mean() if sub["gold_icd"].ne("").any() else 0.0
            rows.append({"bucket": tag, "rows": len(sub), "acc_icd": round(float(ok), 4),
                         "acc_snomed": None})
        elif sys_name == "snomed":
            ok = sub["hit_snomed"].mean() if sub["gold_snomed"].ne("").any() else 0.0
            rows.append({"bucket": tag, "rows": len(sub), "acc_icd": None,
                         "acc_snomed": round(float(ok), 4)})
        else:
            sub_icd = sub[sub["gold_icd"].ne("")]
            sub_sn = sub[sub["gold_snomed"].ne("")]
            rows.append({
                "bucket": tag, "rows": len(sub),
                "acc_icd": round(float(sub_icd["hit_icd"].mean()), 4) if len(sub_icd) else None,
                "acc_snomed": round(float(sub_sn["hit_snomed"].mean()), 4) if len(sub_sn) else None,
            })
    buckets_df = pd.DataFrame(rows)
    buckets_df.to_csv(report_dir / f"{args.split}_buckets.csv", index=False)

    # chapter / tag report
    chap = []
    for c, sub in m[has_icd].groupby(m[has_icd]["gold_icd"].map(icd_chapter)):
        chap.append({"group": c, "system": "icd", "rows": len(sub),
                     "acc": round(float(sub["hit_icd"].mean()), 4)})
    for t, sub in m[has_snomed].groupby(m[has_snomed].apply(
            lambda r: snomed_tag(gold_disp["snomed"].get(r["gold_snomed"], "") if have_term else ""),
            axis=1)):
        chap.append({"group": t, "system": "snomed", "rows": len(sub),
                     "acc": round(float(sub["hit_snomed"].mean()), 4)})
    pd.DataFrame(chap).to_csv(report_dir / f"{args.split}_chapter.csv", index=False)

    # errors dump
    err = m[(has_icd & ~m["hit_icd"]) | (has_snomed & ~m["hit_snomed"])].copy()
    keep = ["chunk_id", "entity", "chunk_text", "gold_icd", "gold_snomed", "pred_icd",
            "pred_snomed", "icd_confidence", "snomed_confidence", "icd_provenance",
            "snomed_provenance", "buckets_str"]
    keep = [c for c in keep if c in err.columns]
    err[keep].to_csv(report_dir / f"{args.split}_errors.csv", index=False)

    # ------------------------------------------------------------ sanity checks
    sanity = {}
    if args.sanity:
        valid_icd = set()
        valid_sn = set()
        try:
            from term_map.terminology import load_terminology

            term2 = load_terminology(
                resolve(cfg["terminology"]["snomed_parquet"]),
                resolve(cfg["terminology"]["icd_parquet"]), None)
            valid_icd = set(term2.icd_codes)
            valid_sn = set(term2.snomed_codes)
        except Exception:
            pass
        n_rows = len(pred)
        p_icd = pred["predicted_icd10cm_code"].map(norm_code_icd)
        p_sn = pred["predicted_snomed_code"].map(norm_code_snomed)
        empty_icd = int((p_icd == "").sum())
        empty_sn = int((p_sn == "").sum())
        bad_fmt_icd = int((~p_icd.str.match(ICD_RE)).sum())
        bad_fmt_sn = int((~p_sn.str.match(SNOMED_RE)).sum())
        not_in_icd = int((~p_icd.map(lambda c: c in valid_icd or c == "")).sum()) if valid_icd else None
        not_in_sn = int((~p_sn.map(lambda c: c in valid_sn or c == "")).sum()) if valid_sn else None
        # leakage audit re-run
        sp2 = pd.read_parquet(resolve(cfg["splits"]["path"]))
        tr2 = sp2[sp2["split"] == "train"]
        te2 = sp2[sp2["split"] == args.split]
        ent_overlap = len(set(tr2["entity"].map(lambda e: basic_norm(str(e)))) &
                          set(te2["entity"].map(lambda e: basic_norm(str(e)))))
        chunk_overlap = len(set(tr2["chunk_id"]) & set(te2["chunk_id"]))
        sanity = {
            "rows": n_rows,
            "empty_icd": empty_icd,
            "empty_snomed": empty_sn,
            "bad_format_icd": bad_fmt_icd,
            "bad_format_snomed": bad_fmt_sn,
            "icd_codes_not_in_terminology": not_in_icd,
            "snomed_codes_not_in_terminology": not_in_sn,
            "entity_overlap_with_train": ent_overlap,
            "chunk_overlap_with_train": chunk_overlap,
        }
        print(f"[eval] sanity: {json.dumps(sanity, indent=1)}")

    summary["sanity"] = sanity
    (report_dir / f"{args.split}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval] reports -> {report_dir}/{args.split}_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
