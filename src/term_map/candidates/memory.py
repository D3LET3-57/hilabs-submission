"""Train memorization source: exact entity hit / fuzzy entity hit / dense-NN over train."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz, process

from ..normalize import Variants, basic_norm
from .base import Candidate, CandidateSet


@dataclass
class TrainLookup:
    """entity_norm -> [(icd, snomed)] + list for fuzzy search."""

    entity_norm: list[str]
    icd: list[str | None]
    snomed: list[str | None]
    chunk_id: list[str | None] = None    # for leave-one-component-out policy training

    def build_index(self):
        self._index = {}
        for i, e in enumerate(self.entity_norm):
            self._index.setdefault(e, []).append(i)
        return self

    @property
    def index(self):
        return self._index


def load_train_lookup(path) -> TrainLookup | None:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if len(df) == 0:
        return None
    def _codes(col: str) -> list[str | None]:
        if col not in df:
            return [None] * len(df)
        return [x if isinstance(x, str) and x else None for x in df[col].tolist()]

    tl = TrainLookup(
        entity_norm=[basic_norm(e) for e in df["entity"].tolist()],
        icd=_codes("icd10cm_code"),
        snomed=_codes("snomed_code"),
        chunk_id=(df["chunk_id"].tolist() if "chunk_id" in df else None),
    )
    return tl.build_index()


def generate(system: str, variants: Variants, lookup: TrainLookup | None,
             fuzzy_top_k: int = 5, fuzzy_min_score: float = 88.0) -> CandidateSet:
    cs = CandidateSet(system=system)
    if lookup is None:
        return cs
    gold_col = lookup.icd if system == "icd" else lookup.snomed

    # exact hits
    for form in variants.normalized:
        rows = lookup.index.get(form)
        if rows:
            for i in rows:
                if gold_col[i]:
                    cs.add(Candidate(code=gold_col[i], source="memory_exact", score=1.0, rank=0,
                                      meta={"train_idx": int(i)}))

    # fuzzy hits over train entity forms (only if short-ish query and no exact hit)
    if not cs.has_source("memory_exact") and variants.normalized:
        q = variants.normalized[0]
        if len(q) >= 3:
            matches = process.extract(
                q, lookup.entity_norm, scorer=fuzz.token_set_ratio,
                limit=fuzzy_top_k, score_cutoff=fuzzy_min_score,
            )
            for rank, (match, score, idx) in enumerate(matches):
                gi = lookup.index.get(match, [idx])
                for i in gi[:3]:
                    if gold_col[i]:
                        cs.add(Candidate(code=gold_col[i], source="memory_fuzzy",
                                          score=min(1.0, score / 100.0), rank=rank,
                                          meta={"train_idx": int(i)}))
    return cs
