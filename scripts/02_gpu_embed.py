#!/usr/bin/env python
"""Embedding generation on the GPU (run via Slurm: scripts/slurm/embed_all.sh).

python scripts/02_gpu_embed.py \
  --model thenlper/gte-base --name gte_base \
  --descriptions data/processed/snomed_desc.parquet \
  --queries data/processed/queries.parquet \
  --out-dir artifacts/embeddings/gte_base/snomed \
  --batch-size 1024 --fp16 --prefix-passage "passage: " --prefix-query "query: "

Outputs in out-dir: desc_emb.f16.npy, desc_codes.npy, query_emb.f16.npy,
query_texts.json, manifest.json. Chunked + resumable (part files).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_model(hf_id: str, fp16: bool, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(hf_id, device=device)
    if fp16:
        model.half()
    model.eval()
    return model


def encode_all(model, texts: list[str], batch_size: int, chunk_size: int, out_dir: Path,
               tag: str, resume: bool = True, show: bool = True) -> np.ndarray:
    import torch

    parts_dir = out_dir / f"parts_{tag}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    part_files = []
    i = 0
    while i < n:
        j = min(n, i + chunk_size)
        p = parts_dir / f"part_{i:09d}.npy"
        part_files.append(p)
        if resume and p.exists():
            i = j
            continue
        with torch.inference_mode():
            emb = model.encode(
                texts[i:j], batch_size=batch_size, show_progress_bar=show,
                normalize_embeddings=True, convert_to_numpy=True,
            )
        if not np.isfinite(emb).all():
            raise RuntimeError(f"non-finite embeddings in {tag} chunk {i}")
        np.save(p, emb.astype(np.float16))
        i = j
    arrs = [np.load(p) for p in part_files]
    out = np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 768), dtype=np.float16)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--name", required=True, help="short name (gte_base, sapbert, biolord)")
    ap.add_argument("--descriptions", required=True)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--chunk-size", type=int, default=200_000)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--prefix-passage", default="")
    ap.add_argument("--prefix-query", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", default=True)
    ap.add_argument("--no-smoke", dest="smoke", action="store_false")
    ap.add_argument("--queries-only", action="store_true",
                    help="encode only the queries file into out-dir (no descriptions)")
    args = ap.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[embed] CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.queries_only:
        model = load_model(args.model, args.fp16, args.device)
        dim = model.get_sentence_embedding_dimension()
        qdf = pd.read_parquet(args.queries)
        q_texts = [args.prefix_query + str(t) for t in qdf["text"].tolist()]
        q_emb = encode_all(model, q_texts, args.batch_size, args.chunk_size,
                           out_dir, "query", resume=True)
        np.save(out_dir / "query_emb.f16.npy", q_emb)
        (out_dir / "query_texts.json").write_text(json.dumps(q_texts))
        manifest = {
            "model": args.model, "name": args.name, "dim": int(dim),
            "prefix_query": args.prefix_query, "n_queries": int(len(q_texts)),
            "normalized": True, "queries_only": True,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - t0, 1), "device": args.device,
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"[embed] queries done in {manifest['elapsed_sec']}s -> {out_dir}")
        return 0

    # ------------------------------------------------------------ descriptions
    df = pd.read_parquet(args.descriptions)
    if "snomed_code" in df.columns:
        df = df.rename(columns={"snomed_code": "code"})
    if "icd_code" in df.columns:
        df = df.rename(columns={"icd_code": "code"})
    df = df[["code", "description"]].dropna()
    df = df.drop_duplicates(subset=["description", "code"]).reset_index(drop=True)
    n_total = len(df)
    texts = df["description"].astype(str).tolist()
    codes = df["code"].astype(str).to_numpy()
    print(f"[embed] {n_total:,} unique (description, code) rows")

    model = load_model(args.model, args.fp16, args.device)
    dim = model.get_sentence_embedding_dimension()

    passage_texts = [args.prefix_passage + t for t in texts]
    desc_emb = encode_all(model, passage_texts, args.batch_size, args.chunk_size,
                          out_dir, "desc", resume=True)
    np.save(out_dir / "desc_emb.f16.npy", desc_emb)
    np.save(out_dir / "desc_codes.npy", codes)
    print(f"[embed] desc embeddings saved {desc_emb.shape}")

    # ------------------------------------------------------------ queries
    if args.queries:
        qdf = pd.read_parquet(args.queries)
        q_texts = [args.prefix_query + str(t) for t in qdf["text"].tolist()]
        q_emb = encode_all(model, q_texts, args.batch_size, args.chunk_size,
                           out_dir, "query", resume=True)
        np.save(out_dir / "query_emb.f16.npy", q_emb)
        (out_dir / "query_texts.json").write_text(json.dumps(q_texts))
        print(f"[embed] query embeddings saved {q_emb.shape}")

    # ------------------------------------------------------------ smoke test
    smoke = {}
    if args.smoke:
        # entity vs its gold display should score higher than entity vs random display
        lookup_p = REPO_ROOT / "data" / "processed" / "train_lookup.parquet"
        smoke = {"status": "skipped"}
        if lookup_p.exists():
            lookup = pd.read_parquet(lookup_p)
            code_col = "snomed_code" if "snomed" in args.descriptions else "icd10cm_code"
            lookup = lookup[lookup[code_col].notna()].head(200)
            code2desc = dict(zip(df["code"].astype(str), df["description"].astype(str)))
            ents = lookup["entity"].astype(str).tolist()
            golds = [code2desc.get(str(c), "") for c in lookup[code_col]]
            rng = np.random.default_rng(17)
            rnds = [code2desc.get(str(c), "") for c in
                    rng.choice(df["code"].astype(str).to_numpy(), size=len(ents), replace=True)]
            with torch.inference_mode():
                e_emb = model.encode([args.prefix_query + t for t in ents],
                                     normalize_embeddings=True, convert_to_numpy=True)
                g_emb = model.encode([args.prefix_passage + t for t in golds],
                                     normalize_embeddings=True, convert_to_numpy=True)
                r_emb = model.encode([args.prefix_passage + t for t in rnds],
                                     normalize_embeddings=True, convert_to_numpy=True)
            gold_cos = float(np.mean(np.sum(e_emb * g_emb, axis=1)))
            rand_cos = float(np.mean(np.sum(e_emb * r_emb, axis=1)))
            smoke = {"n": len(ents), "gold_pair_cos_mean": gold_cos,
                     "random_pair_cos_mean": rand_cos, "status": "ok"}
            print(f"[embed] smoke: gold={gold_cos:.3f} random={rand_cos:.3f}")
            if gold_cos <= rand_cos:
                smoke["status"] = "FAILED"
                raise RuntimeError("smoke test failed: gold pairs not closer than random")

    # ------------------------------------------------------------ manifest
    manifest = {
        "model": args.model,
        "name": args.name,
        "dim": int(dim),
        "fp16": bool(args.fp16),
        "prefix_passage": args.prefix_passage,
        "prefix_query": args.prefix_query,
        "n_desc": int(desc_emb.shape[0]),
        "normalized": True,
        "smoke": smoke,
        "descriptions_file": str(args.descriptions),
        "descriptions_sha256": sha256_file(args.descriptions),
        "script": "02_gpu_embed.py",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "device": args.device,
        "torch": torch.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[embed] done in {manifest['elapsed_sec']}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
