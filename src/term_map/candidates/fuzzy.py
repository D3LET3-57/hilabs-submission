"""Fuzzy candidate generation with rapidfuzz over terminology description strings.

Used as a precision arm for near-miss spellings; guarded by a minimum score and
a length guard to avoid expensive full scans on long phrases.
"""
from __future__ import annotations

from rapidfuzz import fuzz, process

from ..normalize import basic_norm, strip_trailing_punct
from .base import Candidate, CandidateSet


class FuzzyTermIndex:
    def __init__(self, descriptions: list[str], codes: list[str], system: str):
        self.system = system
        self.keys: list[str] = []
        self.keys_code: list[str] = []
        seen = set()
        for d, c in zip(descriptions, codes):
            n = strip_trailing_punct(basic_norm(d))
            if not n or len(n) > 120:
                continue
            k = (n, c)
            if k in seen:
                continue
            seen.add(k)
            self.keys.append(n)
            self.keys_code.append(c)

    def query(self, text: str, top_k: int = 50, min_score: float = 88.0):
        q = strip_trailing_punct(basic_norm(text))
        if not q:
            return []
        matches = process.extract(
            q, self.keys, scorer=fuzz.token_set_ratio, limit=top_k, score_cutoff=min_score
        )
        out: list[tuple[str, float]] = []
        seen: dict[str, float] = {}
        for m, score, _idx in matches:
            i = self.keys.index(m) if False else None
        # process.extract returns (match, score, index) when choices is a list
        out = []
        seen = {}
        for m, score, i in matches:
            c = self.keys_code[i]
            s = score / 100.0
            if c not in seen or s > seen[c]:
                seen[c] = s
        for c, s in sorted(seen.items(), key=lambda kv: -kv[1]):
            out.append((c, s))
        return out


def generate(system: str, texts: list[str], index: FuzzyTermIndex | None,
             top_k: int = 50, min_score: float = 88.0) -> CandidateSet:
    cs = CandidateSet(system=system)
    if index is None:
        return cs
    best: dict[str, float] = {}
    for t in texts[:4]:
        for code, score in index.query(t, top_k=top_k, min_score=min_score):
            if score > best.get(code, -1):
                best[code] = score
    for rank, (code, score) in enumerate(sorted(best.items(), key=lambda kv: -kv[1])):
        cs.add(Candidate(code=code, source="fuzzy", score=score, rank=rank))
    return cs
