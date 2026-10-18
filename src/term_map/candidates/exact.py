"""Exact dictionary candidate source: normalized form == description or == ABBR part."""
from __future__ import annotations

from ..normalize import Variants, strip_trailing_punct
from ..terminology import Terminology
from .base import Candidate, CandidateSet


def generate(system: str, variants: Variants, term: Terminology) -> CandidateSet:
    cs = CandidateSet(system=system)
    if system == "icd":
        norm_index, abbrev_index = term.icd_norm_index, term.icd_abbrev_index
    else:
        norm_index, abbrev_index = term.snomed_norm_index, term.snomed_abbrev_index

    forms = list(variants.normalized)
    if variants.corrected:
        forms.append(variants.corrected)

    rank = 0
    for form in forms:
        key = strip_trailing_punct(form)
        if not key:
            continue
        hits = norm_index.get(key)
        if hits:
            for code in hits:
                # ambiguity penalized: 1 hit = 1.0, many hits share score
                score = 1.0 if len(hits) == 1 else max(0.5, 1.0 - 0.01 * (len(hits) - 1))
                cs.add(Candidate(code=code, source="exact", score=score, rank=rank, display=""))
            rank += 1
        hits = abbrev_index.get(key)
        if hits:
            for code in hits:
                score = 0.9 if len(hits) < 4 else 0.6
                cs.add(Candidate(code=code, source="abbrev", score=score, rank=rank, display=""))
            rank += 1
    return cs
