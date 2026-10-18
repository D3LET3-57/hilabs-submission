"""Crosswalk source: official SNOMED CT -> ICD-10-CM map.

Used strictly as a candidate source / consistency validator (never a derivation
by itself -- the dataset shows systems are labelled independently).
"""
from __future__ import annotations

from ..terminology import Terminology
from .base import Candidate, CandidateSet


def generate_for_snomed(codes: list[str], term: Terminology, top_k: int = 50,
                        source_tag: str = "crosswalk") -> CandidateSet:
    cs = CandidateSet(system="snomed")
    if not term.has_map:
        return cs
    rank = 0
    for code in codes:
        # the map is SNOMED -> ICD; for SNOMED candidates it validates presence
        if code in term.map_by_snomed:
            cs.add(Candidate(code=code, source=source_tag, score=0.7, rank=rank))
            rank += 1
    return cs


def generate_for_icd(snomed_candidates: list[str], term: Terminology, top_k: int = 50,
                     source_tag: str = "crosswalk") -> CandidateSet:
    """ICD candidates from map targets of candidate SNOMED concepts."""
    cs = CandidateSet(system="icd")
    if not term.has_map:
        return cs
    best: dict[str, float] = {}
    for i, sc in enumerate(snomed_candidates[:top_k]):
        targets = term.map_by_snomed.get(sc, [])
        for j, ic in enumerate(targets):
            s = 0.75 - 0.005 * i - 0.001 * j   # earlier/primary sources first
            if ic not in best or s > best[ic]:
                best[ic] = max(s, 0.3)
    for rank, (code, score) in enumerate(sorted(best.items(), key=lambda kv: -kv[1])):
        cs.add(Candidate(code=code, source=source_tag, score=score, rank=rank))
    return cs
