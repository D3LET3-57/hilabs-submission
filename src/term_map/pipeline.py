"""Pipeline orchestration: stages 0-5 of doc 03 section 2, batched for throughput."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm

from .adjudicate import confidence as conf_mod
from .adjudicate.llm import LLMClient
from .adjudicate.policy import PolicyModel
from .adjudicate.rerank import CrossEncoderReranker
from .adjudicate.rules import RulePack
from .candidates import crosswalk as cw_src
from .candidates import exact as exact_src
from .candidates import fuzzy as fuzzy_src
from .candidates import lexical as lex_src
from .candidates import memory as mem_src
from .candidates.base import Candidate, CandidateSet
from .candidates.dense import DenseEncoder, DenseModel
from .config import resolve
from .io import RowCache, row_hash
from .normalize import Variants, all_query_forms, build_variants, clean_chunk, extract_context, fmt_icd
from .terminology import Terminology, load_terminology


@dataclass
class EngineResources:
    cfg: dict
    term: Terminology
    abbrev_graph: dict = field(default_factory=dict)
    lexical: dict = field(default_factory=dict)          # system -> LexicalIndex
    fuzzy: dict = field(default_factory=dict)            # system -> FuzzyTermIndex
    train_lookup: object = None
    dense_models: dict = field(default_factory=dict)     # system -> [DenseModel]
    dense_encoders: dict = field(default_factory=dict)   # model name -> DenseEncoder
    policies: dict = field(default_factory=dict)         # system -> PolicyModel
    llm: LLMClient | None = None
    typo_corrector: object = None


def load_resources(cfg: dict, enable_dense: bool = True, enable_llm: bool = False,
                   device: str | None = None) -> EngineResources:
    term = load_terminology(
        resolve(cfg["terminology"]["snomed_parquet"]),
        resolve(cfg["terminology"]["icd_parquet"]),
        resolve(cfg["terminology"]["map_parquet"]),
    )
    res = EngineResources(cfg=cfg, term=term)

    abbr_path = resolve(cfg["lexicon"]["abbrev_graph"])
    if abbr_path.exists():
        res.abbrev_graph = json.loads(abbr_path.read_text())

    typo_path = resolve(cfg["lexicon"]["typo_pairs"])
    if typo_path.exists():
        try:
            from .textfix import SymSpellCorrector

            res.typo_corrector = SymSpellCorrector.from_json(typo_path)
        except Exception:
            res.typo_corrector = None

    idx_root = resolve(cfg["dense"]["index_dir"])
    for system in ("snomed", "icd"):
        p = idx_root / f"lexical_{system}.joblib"
        if p.exists():
            res.lexical[system] = lex_src.LexicalIndex.load(str(p))
        fp = idx_root / f"fuzzy_{system}.joblib"
        if fp.exists():
            import joblib

            res.fuzzy[system] = joblib.load(str(fp))

    res.train_lookup = mem_src.load_train_lookup(resolve(cfg["lexicon"]["train_lookup"]))

    if enable_dense:
        for system in ("snomed", "icd"):
            models = []
            for mcfg in cfg["dense"]["models"]:
                m = DenseModel(
                    name=mcfg["name"], hf_id=mcfg["hf_id"], dim=mcfg.get("dim", 768),
                    query_prefix=mcfg.get("query_prefix", ""),
                    passage_prefix=mcfg.get("passage_prefix", ""),
                    weight=float(mcfg.get("weight", 1.0)),
                )
                m.load(str(idx_root / mcfg["name"] / system))
                models.append(m)
            res.dense_models[system] = models

    if enable_llm:
        lcfg = cfg["llm"]
        res.llm = LLMClient(
            cache_path=str(resolve(lcfg["cache_path"])),
            timeout=float(lcfg.get("default_timeout", 60)),
            max_retries=int(lcfg.get("max_retries", 3)),
            max_tokens=int(lcfg.get("max_tokens", 512)),
            per_source_k=int(lcfg.get("per_source_k", 0)),
            max_concurrency=int(lcfg.get("default_max_concurrency", 4)),
            self_consistency_n=int(lcfg.get("self_consistency_n", 3)),
            self_consistency_temperature=float(lcfg.get("self_consistency_temperature", 0.3)),
            self_consistency_margin=float(lcfg.get("self_consistency_margin", 0.15)),
            prompt_version=lcfg.get("prompt_version", "v1"),
        )

    for system in ("snomed", "icd"):
        res.policies[system] = PolicyModel.load(str(resolve(cfg["rules"][f"policy_model_{system}"])))
    return res


class Engine:
    def __init__(
        self,
        res: EngineResources,
        use_rules: bool = True,
        use_policy: bool = True,
        use_llm: bool = False,
        sources: list[str] | None = None,
        row_cache_path: str | None = None,
        lexical_min_score: float = 0.05,
        context_window: int | None = None,
        llm_always: bool = False,
        use_rerank: bool | None = None,
        rerank_device: str | None = None,
        use_judge: bool | None = None,
        judge_device: str | None = None,
    ):
        self.res = res
        self.use_rules = use_rules
        self.use_policy = use_policy
        self.use_llm = bool(use_llm and res.llm is not None and res.llm.base_url)
        self.llm_always = llm_always
        self.sources = sources or ["exact", "memory", "lexical", "dense", "fuzzy", "crosswalk"]
        rcfg = res.cfg.get("rerank", {}) or {}
        enabled = bool(rcfg.get("enabled", False)) if use_rerank is None else bool(use_rerank)
        self.reranker: CrossEncoderReranker | None = None
        if enabled:
            self.reranker = CrossEncoderReranker(
                model_id=rcfg.get("model", CrossEncoderReranker.model_id),
                device=rerank_device,
                top_n=int(rcfg.get("top_n", 30)),
                max_length=int(rcfg.get("max_length", 256)),
            )
        jcfg = res.cfg.get("judge", {}) or {}
        judge_on = bool(jcfg.get("enabled", False)) if use_judge is None else bool(use_judge)
        self.judge = None
        if judge_on:
            from .adjudicate.encoder_judge import EncoderJudge

            self.judge = EncoderJudge(
                str(resolve(jcfg.get("model", "artifacts/ultrabert_judge3"))),
                device=judge_device,
                per_source_k=int(jcfg.get("per_source_k", 20)),
                top=int(jcfg.get("top", 120)),
                batch=int(jcfg.get("batch", 128)),
                max_len=int(jcfg.get("max_len", 256)),
            )
        self.context_window = context_window or res.cfg["run"]["context_window"]
        self.cache = RowCache(row_cache_path if row_cache_path else None)
        self.lexical_min_score = lexical_min_score
        self.char_weight = 0.35
        fw = (res.cfg.get("fusion", {}) or {}).get("source_weights", {}) or {}
        self.source_weights = {**conf_mod.DEFAULT_SOURCE_WEIGHTS, **fw}
        self.policy_blend = float((res.cfg.get("rules", {}) or {}).get("policy_blend", 0.45))
        self.rerank_weight = float(rcfg.get("weight", 0.9))
        self.source_weights["rerank"] = self.rerank_weight
        self.source_weights["judge"] = float(jcfg.get("weight", 0.75))
        self.rerank_gate = float(rcfg.get("gate_margin", 0.0) or 0.0)
        self.rerank_gate_system = rcfg.get("gate_system", None)
        self.rule_packs = {s: RulePack(s, res.term) for s in ("snomed", "icd")}
        self.stats = {"rows": 0, "llm_calls": 0, "llm_skipped": 0, "empty_candidates": 0,
                      "exact_unique": 0, "memory_exact": 0, "agreement": 0, "fused": 0}

    # ------------------------------------------------------------------ helpers
    def _make_variants(self, entity: str, chunk_text: str) -> tuple[Variants, str, str]:
        cleaned = clean_chunk(chunk_text)
        ctx = extract_context(cleaned, entity, self.context_window) if cleaned else ""
        v = build_variants(entity, self.res.abbrev_graph or {}, self.res.typo_corrector)
        return v, ctx, cleaned

    def _lexical_texts(self, v: Variants) -> list[str]:
        return all_query_forms(v)

    # ------------------------------------------------------------------ batch
    def _prepare_and_query(self, df: pd.DataFrame, resume: bool = False
                           ) -> tuple[list[dict], dict[int, dict], dict, dict]:
        rows = df.to_dict("records")
        n = len(rows)
        prepared: list[dict] = []
        skipped: dict[int, dict] = {}
        for i, r in enumerate(rows):
            key = row_hash(str(r["chunk_id"]), str(r["clinical_entity"]), str(r["chunk_text"]))
            if resume and self.cache.enabled:
                cached = self.cache.get(key)
                if cached is not None:
                    skipped[i] = cached
                    continue
            v, ctx, cleaned = self._make_variants(str(r["clinical_entity"]), str(r["chunk_text"]))
            prepared.append({"i": i, "pos": len(prepared), "row": r, "key": key,
                             "variants": v, "context": ctx, "cleaned": cleaned})

        # ---- batch lexical queries (aligned with prepared order)
        lex_results: dict[str, list[dict[str, float]]] = {}
        if "lexical" in self.sources:
            for system, idx in self.res.lexical.items():
                if idx is None:
                    continue
                all_texts: list[str] = []
                spans: list[tuple[int, int]] = []
                for p in prepared:
                    texts = self._lexical_texts(p["variants"])
                    spans.append((len(all_texts), len(all_texts) + len(texts)))
                    all_texts.extend(texts)
                top_k = self.res.cfg["candidates"]["lexical_top_k"]
                hits = idx.query(all_texts, top_k=top_k, char_weight=self.char_weight) if all_texts else []
                per_row: list[dict[str, float]] = []
                for (a, b) in spans:
                    merged: dict[str, float] = {}
                    for res_row in hits[a:b]:
                        for code, score in res_row:
                            if score > merged.get(code, -1):
                                merged[code] = score
                    per_row.append(merged)
                lex_results[system] = per_row

        # ---- batch dense queries
        dense_results: dict[str, dict[str, list[dict[str, float]]]] = {}
        if "dense" in self.sources and self.res.dense_models:
            for system, models in self.res.dense_models.items():
                usable = [m for m in models if m.loaded]
                if not usable:
                    continue
                views: list[list[str]] = []
                for p in prepared:
                    v = p["variants"]
                    qv: list[str] = [v.no_dot]
                    if p["context"]:
                        qv.append(f"{v.cleaned} || {p['context']}")
                    if v.corrected:
                        qv.append(v.corrected)
                    for e in v.expanded[:2]:
                        qv.append(e)
                    dedup: list[str] = []
                    for q in qv:
                        if q and q not in dedup:
                            dedup.append(q)
                    views.append(dedup[:4])
                per_model: dict[str, list[dict[str, float]]] = {}
                top_k = self.res.cfg["candidates"]["dense_top_k"]
                for m in usable:
                    enc = self.res.dense_encoders.get(m.name)
                    if enc is None:
                        cache_dir = None
                        if m.index_dir:
                            cand = os.path.join(os.path.dirname(os.path.dirname(m.index_dir)),
                                                "embeddings", m.name, "queries")
                            # <index_root>/<model>/<system> -> <repo>/artifacts/embeddings/<model>/queries
                            cand = str(resolve(self.res.cfg["dense"]["embedding_dir"]) /
                                       m.name / "queries")
                            if os.path.isdir(cand):
                                cache_dir = cand
                        enc = DenseEncoder(m.hf_id, m.query_prefix, device=None,
                                           cache_dir=cache_dir)
                        self.res.dense_encoders[m.name] = enc
                    flat: list[str] = []
                    spans: list[tuple[int, int]] = []
                    for qv in views:
                        spans.append((len(flat), len(flat) + len(qv)))
                        flat.extend(qv)
                    if not flat:
                        continue
                    emb = enc.encode(flat)
                    hits = m.search(emb, top_k)
                    per_row: list[dict[str, float]] = []
                    for (a, b) in spans:
                        merged: dict[str, float] = {}
                        for res_row in hits[a:b]:
                            for code, score in res_row:
                                if code not in merged or score > merged[code]:
                                    merged[code] = score
                        per_row.append(merged)
                    per_model[m.name] = per_row
                dense_results[system] = per_model

        return prepared, skipped, lex_results, dense_results

    def process_batch(
        self,
        df: pd.DataFrame,
        out_path: str | None = None,
        resume: bool = False,
        log_every: int = 0,
    ) -> pd.DataFrame:
        n = len(df)
        prepared, skipped, lex_results, dense_results = self._prepare_and_query(df, resume)

        # ---- per-row assembly + decisions
        out_rows: list[dict | None] = [None] * n
        for i, cached in skipped.items():
            out_rows[i] = cached
        todo = [p for p in prepared]
        workers = 1
        if self.use_llm:
            workers = int(getattr(self.res.llm, "max_concurrency", 0)
                          or self.res.cfg["llm"].get("default_max_concurrency", 4))
        if workers > 1 and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(self._process_prepared, p, lex_results, dense_results): p
                        for p in todo}
                for fut in tqdm(as_completed(futs), total=len(futs), desc="engine",
                                mininterval=5.0, disable=len(todo) < 50):
                    p = futs[fut]
                    out = fut.result()
                    self.cache.put(p["key"], out)
                    out_rows[p["i"]] = out
                    if log_every and out_path and (len([x for x in out_rows if x]) % log_every == 0):
                        pd.DataFrame([x for x in out_rows if x]).to_csv(out_path, index=False)
        else:
            for p in tqdm(todo, total=len(todo), desc="engine", mininterval=5.0, disable=len(todo) < 50):
                out = self._process_prepared(p, lex_results, dense_results)
                self.cache.put(p["key"], out)
                out_rows[p["i"]] = out
                if log_every and out_path and (len([x for x in out_rows if x]) % log_every == 0):
                    pd.DataFrame([x for x in out_rows if x]).to_csv(out_path, index=False)
        result = pd.DataFrame(out_rows)
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            result.to_csv(out_path, index=False)
        return result

    # ------------------------------------------------------------------ row
    def collect_candidates(self, df: pd.DataFrame, limit: int = 0
                           ) -> list[dict]:
        """Stage 1-4 only (no decision): used to train the pointwise policy with
        exactly the same candidate/feature distribution as inference."""
        prepared, _skipped, lex_results, dense_results = self._prepare_and_query(df)
        if limit:
            prepared = prepared[:limit]
        out = []
        for p in tqdm(prepared, total=len(prepared), desc="collect", mininterval=5.0):
            sets = self._assemble_sets(p, lex_results, dense_results)
            out.append({"row": p["row"], "variants": p["variants"],
                        "context": p["context"], "sets": sets, "pos": p["pos"]})
        return out

    def _assemble_sets(self, p: dict, lex_results: dict,
                       dense_results: dict) -> dict[str, CandidateSet]:
        v = p["variants"]
        sets: dict[str, CandidateSet] = {"snomed": CandidateSet(system="snomed"),
                                         "icd": CandidateSet(system="icd")}

        if "exact" in self.sources:
            for system in sets:
                sets[system].extend(exact_src.generate(system, v, self.res.term).candidates.values())

        if "memory" in self.sources and self.res.train_lookup is not None:
            for system in sets:
                cs = mem_src.generate(
                    system, v, self.res.train_lookup,
                    self.res.cfg["candidates"]["fuzzy_top_k"],
                    self.res.cfg["candidates"]["fuzzy_min_score"],
                )
                sets[system].extend(cs.candidates.values())

        pos = p["pos"]
        if "lexical" in self.sources:
            for system, per_row in lex_results.items():
                for code, score in per_row[pos].items():
                    sets[system].add(Candidate(code=code, source="lexical", score=score))

        if "dense" in self.sources:
            for system, per_model in dense_results.items():
                for model_name, per_row in per_model.items():
                    for code, score in per_row[pos].items():
                        sets[system].add(Candidate(code=code, source=f"dense_{model_name}", score=score))

        # crosswalk: ICD candidates from SNOMED candidates; SNOMED presence flag
        if "crosswalk" in self.sources and self.res.term.has_map:
            snomed_codes = [c for c in sorted(
                sets["snomed"].candidates.values(), key=lambda c: -c.score)][:50]
            cw_icd = cw_src.generate_for_icd([c.code for c in snomed_codes], self.res.term)
            sets["icd"].extend(cw_icd.candidates.values())
            cw_sn = cw_src.generate_for_snomed(list(sets["snomed"].candidates.keys()),
                                               self.res.term)
            sets["snomed"].extend(cw_sn.candidates.values())

        # fuzzy arm: only when a system has no exact/abbrev hit and weak lexical
        if "fuzzy" in self.sources:
            for system, idx in self.res.fuzzy.items():
                if idx is None:
                    continue
                cs = sets[system]
                weak = True
                if cs.has_source("exact") or cs.has_source("abbrev") or cs.has_source("memory_exact"):
                    weak = False
                lex_scores = [c.score for c in cs.provenance.get("lexical", [])]
                if lex_scores and max(lex_scores) >= 0.45:
                    weak = False
                if weak:
                    fcs = fuzzy_src.generate(system, all_query_forms(v), idx,
                                             top_k=self.res.cfg["candidates"]["fuzzy_top_k"],
                                             min_score=88.0)
                    cs.extend(fcs.candidates.values())
                    if cs.has_source("fuzzy"):
                        classic_weak = False

        # cross-encoder rerank stage (same path in training collect and inference)
        if self.reranker is not None:
            entity = str(p["row"]["clinical_entity"])
            ctx = p["context"]
            for system, cs in sets.items():
                disp_map = (self.res.term.icd_display if system == "icd"
                            else self.res.term.snomed_display)
                pre = conf_mod.fuse(cs, self.source_weights)
                gate = self.rerank_gate if self.rerank_gate_system in (None, system) else 0.0
                if gate > 0:
                    top2 = sorted(pre.items(), key=lambda kv: -kv[1])[:2]
                    margin = (top2[0][1] - top2[1][1]) if len(top2) > 1 else 1.0
                    if margin > gate:
                        continue
                trace = self.reranker.rerank(system, cs, pre, entity, ctx, disp_map)
                if trace:
                    cs.rerank_trace = trace
                    self.stats["rerank_rows"] = self.stats.get("rerank_rows", 0) + 1

        # encoder judge stage (fine-tuned pairwise model -> extra candidate source)
        if self.judge is not None:
            state = f"{p['row']['clinical_entity']} || {p['context']}"
            for system, cs in sets.items():
                disp_map = (self.res.term.icd_display if system == "icd"
                            else self.res.term.snomed_display)
                for code, prob in self.judge.scores(state, cs, disp_map).items():
                    cs.add(Candidate(code=code, source="judge", score=prob))
            self.stats["judge_rows"] = self.stats.get("judge_rows", 0) + 1
        return sets

    def _process_prepared(self, p: dict, lex_results: dict, dense_results: dict) -> dict:
        r = p["row"]
        entity = str(r["clinical_entity"])
        ctx = p["context"]
        v = p["variants"]
        chunk_id = str(r["chunk_id"])
        self.stats["rows"] += 1

        sets = self._assemble_sets(p, lex_results, dense_results)

        # decide per system
        out: dict = {"chunk_id": chunk_id, "clinical_entity": entity}
        for system, code_col, conf_col, prov_col in (
            ("icd", "predicted_icd10cm_code", "icd_confidence", "icd_provenance"),
            ("snomed", "predicted_snomed_code", "snomed_confidence", "snomed_provenance"),
        ):
            cs = sets[system]
            decision = self._decide_system(system, cs, entity, ctx, variants=v)
            out[code_col] = decision["code"]
            out[conf_col] = decision["confidence"]
            out[prov_col] = json.dumps(decision["provenance"], default=str)
        self.stats["llm_calls"] += int(self.use_llm) * 0
        out["engine_version"] = self.res.cfg["output"]["engine_version"]
        return out

    def _decide_system(self, system: str, cs: CandidateSet, entity: str, ctx: str,
                       variants: Variants | None = None) -> dict:
        disp_map = self.res.term.icd_display if system == "icd" else self.res.term.snomed_display

        # LLM adjudication first when enabled (with deterministic fallback)
        llm_info = None
        if self.use_llm and len(cs) > 0:
            should_call = self.llm_always
            margin = self._margin(cs)
            if not should_call:
                # call when not trivially resolvable
                exact_unique = any(c.source == "exact" and c.score >= 1.0 for c in cs.candidates.values())
                mem_exact = any(c.source == "memory_exact" for c in cs.candidates.values())
                should_call = not (exact_unique or mem_exact)
            if should_call:
                use_sc = margin < self.res.cfg["llm"]["self_consistency_margin"]
                result = self.res.llm.choose(system, entity, ctx, cs, disp_map, use_sc)
                if result is not None:
                    self.stats["llm_calls"] += 1
                    self.res.llm.merge_into_candidates(cs, result)
                    llm_info = {"code": result.code, "reason": result.reason,
                                "n_votes": result.n_votes, "cached": result.cached}
                else:
                    self.stats["llm_skipped"] += 1

        rerank_trace = getattr(cs, "rerank_trace", None)

        rules = self.rule_packs[system] if self.use_rules else None
        policy = self.res.policies.get(system) if self.use_policy else None
        if policy is not None:
            policy.blend = self.policy_blend
        dec = conf_mod.decide(cs, policy=policy, rules=rules, context=ctx, entity=entity,
                              term=self.res.term, weights=self.source_weights)

        code = dec.code
        if code is None:
            # last resort chain: lexical top-1 -> any candidate -> fuzzy best (never empty
            # when an index exists; doc 03 section 5)
            lex = sorted(cs.provenance.get("lexical", []), key=lambda c: -c.score)
            if lex:
                code = lex[0].code
                dec.reason = "lexical_last_resort"
                dec.confidence = 0.1
            elif cs.candidates:
                code = cs.top(1)[0].code
                dec.reason = "candidate_last_resort"
                dec.confidence = 0.05
            else:
                fidx = self.res.fuzzy.get(system)
                hit = None
                if fidx is not None and variants is not None:
                    for t in all_query_forms(variants)[:4]:
                        got = fidx.query(t, top_k=5, min_score=0.0)
                        if got:
                            hit = got[0]
                            break
                if hit is not None:
                    code = hit[0]
                    cs.add(Candidate(code=hit[0], source="fuzzy", score=hit[1], rank=0))
                    dec.reason = "fuzzy_last_resort"
                    dec.confidence = 0.02
                else:
                    self.stats["empty_candidates"] += 1
                    code = ""
                    dec.reason = "empty"
                    dec.confidence = 0.0

        if system == "icd" and code:
            code = fmt_icd(code)
        valid = self.res.term.icd_codes if system == "icd" else self.res.term.snomed_codes
        if code and code not in valid:
            # defensive: emitted code must exist in the target terminology
            replacement = next((c.code for c in cs.top(50) if c.code in valid), None)
            if replacement is None:
                replacement = next(
                    (c.code for c in sorted(cs.provenance.get("lexical", []),
                                             key=lambda c: -c.score) if c.code in valid), None)
            if replacement is not None:
                dec.reason = f"{dec.reason}+out_of_terminology_filtered"
                dec.confidence = min(dec.confidence, 0.10)
                code = fmt_icd(replacement) if system == "icd" else replacement
            else:
                code = ""
        self.stats[("exact_unique" if dec.reason == "exact_unique" else
                    "memory_exact" if dec.reason == "memory_exact" else
                    "agreement" if dec.reason.startswith("agreement") else "fused")] += 1

        prov = {
            "reason": dec.reason,
            "confidence": dec.confidence,
            "top": [
                {"code": c.code, "score": round(c.score, 4), "sources": cs.sources_for(c.code)}
                for c in cs.top(10)
            ],
            "trace": dec.trace,
        }
        if llm_info:
            prov["llm"] = llm_info
        if rerank_trace:
            prov["rerank"] = rerank_trace[:10]
        return {"code": code, "confidence": dec.confidence, "provenance": prov}

    @staticmethod
    def _margin(cs: CandidateSet) -> float:
        top = cs.top(2)
        if len(top) < 2:
            return 1.0
        return top[0].score - top[1].score
