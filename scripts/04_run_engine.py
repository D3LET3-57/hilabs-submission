#!/usr/bin/env python
"""End-to-end engine CLI.

python scripts/04_run_engine.py --input eval/dev_input.csv --output out/dev.csv \
    [--no-dense] [--llm] [--sources exact,memory,lexical,dense,fuzzy,crosswalk] \
    [--no-rules] [--no-policy] [--limit N] [--resume] [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from term_map.config import load_config  # noqa: E402
from term_map.io import read_input, write_predictions  # noqa: E402
from term_map.pipeline import Engine, load_resources  # noqa: E402
from term_map.version import engine_version  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--llm", action="store_true", help="enable LLM adjudication (needs LLM_* env)")
    ap.add_argument("--llm-always", action="store_true", help="call LLM for every row")
    ap.add_argument("--sources", default=None,
                    help="comma list of candidate sources: exact,memory,lexical,dense,fuzzy,crosswalk")
    ap.add_argument("--lexical-only", action="store_true",
                    help="floor reproduction: sources=lexical only")
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--no-policy", action="store_true")
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                    help="enable cross-encoder reranker (default: config)")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--rerank-device", default=None)
    ap.add_argument("--judge", dest="judge", action="store_true", default=None,
                    help="enable fine-tuned encoder judge (default: config)")
    ap.add_argument("--no-judge", dest="judge", action="store_false")
    ap.add_argument("--judge-device", default=None)
    ap.add_argument("--char-weight", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--context-window", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sources = None
    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if args.lexical_only:
        sources = ["lexical"]

    t0 = time.time()
    enable_dense = not args.no_dense and sources is not None and "dense" in sources \
        if sources is not None else not args.no_dense
    if sources is not None and "dense" not in sources:
        enable_dense = False
    print(f"[engine] loading resources (dense={enable_dense}, llm={args.llm}) ...", flush=True)
    res = load_resources(cfg, enable_dense=enable_dense, enable_llm=args.llm, device=args.device)
    # device preference for encoders
    if args.device:
        for enc in res.dense_encoders.values():
            enc.device = args.device

    df = read_input(args.input)
    if args.limit:
        df = df.head(args.limit)
    print(f"[engine] input rows={len(df)} sources={sources or 'all'}", flush=True)

    engine = Engine(
        res,
        use_rules=not args.no_rules,
        use_policy=not args.no_policy,
        use_llm=args.llm,
        sources=sources,
        row_cache_path=(str(REPO_ROOT / "artifacts" / "cache" / "rows.sqlite") if args.resume else None),
        context_window=args.context_window,
        llm_always=args.llm_always,
        use_rerank=args.rerank,
        rerank_device=args.rerank_device,
        use_judge=args.judge,
        judge_device=args.judge_device,
    )
    if args.char_weight is not None:
        engine.char_weight = args.char_weight
    if args.lexical_only:
        engine.char_weight = 0.0  # floor reproduction: word TF-IDF only
    out = engine.process_batch(df, out_path=args.output, resume=args.resume)
    if "engine_version" in out.columns:
        out["engine_version"] = engine_version()
    write_predictions(out, args.output)
    dt = time.time() - t0
    print(f"[engine] wrote {args.output} rows={len(out)} in {dt:.1f}s ({len(out) / max(dt, 1e-9):.1f} rows/s)")
    print(f"[engine] stats: {engine.stats}")
    if res.llm is not None:
        print(f"[engine] llm stats: {res.llm.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
