#!/usr/bin/env python
"""M0 step 1: raw xlsx/CSV -> parquet tables + manifest.

Outputs (data/processed/):
  snomed_desc.parquet   snomed_code, description, description_order
  icd_desc.parquet      code, description  (canonical dotted ICD format)
  map.parquet           snomed_code, icd_code, map_priority_rule, advice, active, status_message
  train_pairs.parquet   one row per (entity, chunk_text) with per-system golds
  manifest.json         release labels, file hashes, counts, data-quality flags

Run: python scripts/00_prepare_data.py [--skip-xlsx]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from term_map.normalize import fmt_icd  # noqa: E402

DATA = REPO_ROOT / "data"
OUT = DATA / "processed"

DEFAULT_TRAIN = DATA / "hackathon_train.csv"
DEFAULT_SNOMED = DATA / "SNOMEDCT_US_codes_and_descriptions.xlsx"
DEFAULT_ICD = DATA / "icd10cm_codes.xlsx"
DEFAULT_MAP_CANDIDATES = [
    REPO_ROOT.parent / "snomed_icd_10_map.csv_0_0_0.csv.gz",
    DATA / "snomed_icd_10_map.csv_0_0_0.csv.gz",
    DATA / "raw" / "snomed_icd_10_map.csv_0_0_0.csv.gz",
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def find_map() -> Path | None:
    for p in DEFAULT_MAP_CANDIDATES:
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=str(DEFAULT_TRAIN))
    ap.add_argument("--snomed-xlsx", default=str(DEFAULT_SNOMED))
    ap.add_argument("--icd-xlsx", default=str(DEFAULT_ICD))
    ap.add_argument("--map-csv", default=None)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--skip-xlsx", action="store_true", help="reuse existing parquet tables")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    report: dict = {"counts": {}, "flags": [], "files": {}}

    # ---------------------------------------------------------------- SNOMED
    snomed_path = out / "snomed_desc.parquet"
    if not args.skip_xlsx or not snomed_path.exists():
        print("[snomed] reading xlsx ...", flush=True)
        sn = pd.read_excel(args.snomed_xlsx, dtype={"snomed_code": str}, engine="openpyxl")
        sn = sn.rename(columns={"snomed_code": "snomed_code"})
        cols = [c for c in ["snomed_code", "description", "description_order"] if c in sn.columns]
        sn = sn[cols].copy()
        sn["snomed_code"] = sn["snomed_code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        sn = sn[sn["snomed_code"].str.fullmatch(r"\d+", na=False)]
        if "description_order" in sn.columns:
            sn["description_order"] = pd.to_numeric(sn["description_order"], errors="coerce").fillna(0).astype(int)
        else:
            sn["description_order"] = 1
        sn["description"] = sn["description"].astype(str)
        sn = sn.drop_duplicates().reset_index(drop=True)
        sn.to_parquet(snomed_path, index=False)
    sn = pd.read_parquet(snomed_path)
    report["counts"]["snomed_rows"] = int(len(sn))
    report["counts"]["snomed_unique_codes"] = int(sn["snomed_code"].nunique())
    report["counts"]["snomed_preferred_rows"] = int((sn["description_order"] == 1).sum())
    print(f"[snomed] rows={len(sn):,} unique codes={sn['snomed_code'].nunique():,}")

    # ---------------------------------------------------------------- ICD
    icd_path = out / "icd_desc.parquet"
    if not args.skip_xlsx or not icd_path.exists():
        print("[icd] reading xlsx ...", flush=True)
        ic = pd.read_excel(args.icd_xlsx, dtype=str, engine="openpyxl")
        ic.columns = [str(c).strip() for c in ic.columns]
        code_col = "Code" if "Code" in ic.columns else ic.columns[0]
        desc_col = "Description" if "Description" in ic.columns else ic.columns[1]
        ic = ic.rename(columns={code_col: "code", desc_col: "description"})[["code", "description"]]
        ic["description"] = ic["description"].fillna("").astype(str)
        ic["code"] = ic["code"].fillna("").astype(str).str.strip()
        headers = ic["code"].str.contains("-", regex=False)
        n_headers = int(headers.sum())
        ic = ic[~headers & ic["code"].str.len().gt(0)].copy()
        ic["code"] = ic["code"].map(fmt_icd)
        ic = ic.drop_duplicates().reset_index(drop=True)
        ic.to_parquet(icd_path, index=False)
        report["counts"]["icd_header_rows_dropped"] = n_headers
    ic = pd.read_parquet(icd_path)
    report["counts"]["icd_rows"] = int(len(ic))
    report["counts"]["icd_unique_codes"] = int(ic["code"].nunique())
    print(f"[icd] rows={len(ic):,} unique codes={ic['code'].nunique():,}")

    # ---------------------------------------------------------------- map
    map_path = out / "map.parquet"
    map_src = Path(args.map_csv) if args.map_csv else find_map()
    if map_src and map_src.exists() and (not args.skip_xlsx or not map_path.exists()):
        print(f"[map] reading {map_src} ...", flush=True)
        # NB: this file ships WITHOUT a header row despite the .csv name.
        names = ["uuid", "release", "active", "module_1", "module_2", "snomed_code",
                 "snomed_display", "a", "b", "map_priority_rule", "advice", "icd_code",
                 "icd_display", "map_category_id", "map_rule_id", "status_message"]
        mp = pd.read_csv(map_src, dtype=str, keep_default_na=False, header=None,
                         names=names, compression=None)
        mp = mp[["release", "active", "snomed_code", "snomed_display", "map_priority_rule",
                 "advice", "icd_code", "icd_display", "map_category_id",
                 "map_rule_id", "status_message"]].copy()
        mp["snomed_code"] = mp["snomed_code"].astype(str).str.strip()
        mp["icd_code"] = mp["icd_code"].map(fmt_icd)
        mp = mp.drop_duplicates().reset_index(drop=True)
        mp.to_parquet(map_path, index=False)
    if map_path.exists():
        mp = pd.read_parquet(map_path)
        report["counts"]["map_rows"] = int(len(mp))
        print(f"[map] rows={len(mp):,}")
    else:
        report["flags"].append("map file missing; crosswalk source disabled")

    # ---------------------------------------------------------------- train pairs
    print("[train] reshaping ...", flush=True)
    tr = pd.read_csv(args.train_csv)
    dx = tr[tr["subconcept"] == "DiagnosisCode"].copy()
    report["counts"]["dx_rows"] = int(len(dx))
    report["counts"]["dx_unique_entities"] = int(dx["entity"].nunique())
    report["counts"]["dx_unique_chunks"] = int(dx["chunk_text"].nunique())

    sn_rows = dx[dx["coding_system"] == "SNOMEDCT_US"][["entity", "chunk_text", "code", "display"]]
    ic_rows = dx[dx["coding_system"] == "ICD10CM"][["entity", "chunk_text", "code", "display"]]
    sn_rows = sn_rows.rename(columns={"code": "snomed_code", "display": "snomed_display"})
    ic_rows = ic_rows.rename(columns={"code": "icd10cm_code", "display": "icd_display"})
    ic_rows["icd10cm_code"] = ic_rows["icd10cm_code"].map(fmt_icd)

    pairs = sn_rows.merge(ic_rows, on=["entity", "chunk_text"], how="outer")
    pairs["snomed_code"] = pairs["snomed_code"].astype("string")
    pairs["icd10cm_code"] = pairs["icd10cm_code"].astype("string")
    pairs["both_gold"] = pairs["snomed_code"].notna() & pairs["icd10cm_code"].notna()
    pairs = pairs.reset_index(drop=True)
    pairs.to_parquet(out / "train_pairs.parquet", index=False)

    n_both = int(pairs["both_gold"].sum())
    n_sn_only = int((pairs["snomed_code"].notna() & pairs["icd10cm_code"].isna()).sum())
    n_ic_only = int((pairs["icd10cm_code"].notna() & pairs["snomed_code"].isna()).sum())
    report["counts"].update(pairs=len(pairs), both_gold=n_both,
                            snomed_only=n_sn_only, icd_only=n_ic_only)
    print(f"[train] pairs={len(pairs)} both={n_both} snomed_only={n_sn_only} icd_only={n_ic_only}")

    # checks (soft assertions per plan docs; recorded, not fatal)
    # note: plan counted 2,879 pairs with non-null chunk_text; the raw file also
    # has 3 chunkless dx rows (2 icd-only, 1 snomed-only) which we keep.
    n_chunkless = int(pairs["chunk_text"].isna().sum())
    report["counts"]["chunkless_pairs"] = n_chunkless
    checks = []
    if len(pairs) != 2882:
        checks.append(f"pair count {len(pairs)} != expected 2882 (2879 chunked + 3 chunkless)")
    if n_both != 2123:
        checks.append(f"both-gold {n_both} != expected 2123")
    if n_sn_only != 577:
        checks.append(f"snomed-only {n_sn_only} != expected 577")
    if n_ic_only != 182:
        checks.append(f"icd-only {n_ic_only} != expected 182")
    if n_chunkless != 3:
        checks.append(f"chunkless rows {n_chunkless} != expected 3")
    report["flags"].extend(checks)

    # label-noise flags
    sn_bad = pairs[pairs["snomed_code"].notna() &
                   ~pairs["snomed_code"].isin(set(sn["snomed_code"]))]
    ic_bad = pairs[pairs["icd10cm_code"].notna() &
                   ~pairs["icd10cm_code"].isin(set(ic["code"]))]
    report["counts"]["snomed_gold_missing_in_term"] = int(len(sn_bad))
    report["counts"]["icd_gold_missing_in_term"] = int(len(ic_bad))
    if len(ic_bad):
        report["flags"].append(
            f"ICD gold codes not in terminology: {sorted(set(ic_bad['icd10cm_code']))}")

    # ---------------------------------------------------------------- manifest
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "releases": {
            "snomed": "20260901 (US Edition export 24 Sep 2026)",
            "icd": "FY2026 snapshot",
            "map": "20230901",
        },
        "files": {},
        "report": report,
    }
    for pth in [Path(args.train_csv), Path(args.snomed_xlsx), Path(args.icd_xlsx)] + \
               ([map_src] if map_src else []):
        if pth and pth.exists():
            manifest["files"][pth.name] = {"sha256": sha256_file(pth), "size": pth.stat().st_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "pairs_report.json").write_text(json.dumps(report, indent=2))
    print(f"[done] elapsed {time.time() - t0:.1f}s -> {out}")
    if checks:
        print("[warn] " + "; ".join(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
