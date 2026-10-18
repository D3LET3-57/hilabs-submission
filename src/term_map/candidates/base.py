"""Candidate datatypes shared by all sources."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    code: str
    source: str
    score: float = 0.0          # normalized 0..1 within source (higher = better)
    rank: int = 0               # 0-based rank within source
    display: str = ""
    meta: dict | None = None    # e.g. {"train_idx": i} for leave-one-component-out training


@dataclass
class CandidateSet:
    system: str                 # "icd" | "snomed"
    candidates: dict[str, Candidate] = field(default_factory=dict)
    provenance: dict[str, list[Candidate]] = field(default_factory=dict)

    def add(self, cand: Candidate) -> None:
        existing = self.candidates.get(cand.code)
        if existing is None or cand.score > existing.score:
            self.candidates[cand.code] = cand
        self.provenance.setdefault(cand.source, []).append(cand)

    def extend(self, cands: list[Candidate]) -> None:
        for c in cands:
            self.add(c)

    def has_source(self, source: str) -> bool:
        return source in self.provenance and bool(self.provenance[source])

    def sources_for(self, code: str) -> list[str]:
        out = []
        for src, cands in self.provenance.items():
            if any(c.code == code for c in cands):
                out.append(src)
        return out

    def n_sources(self, code: str) -> int:
        return len(self.sources_for(code))

    def top(self, k: int = 20) -> list[Candidate]:
        ranked = sorted(self.candidates.values(), key=lambda c: (-c.score, c.source, c.code))
        return ranked[:k]

    def __len__(self) -> int:
        return len(self.candidates)

    def __contains__(self, code: str) -> bool:
        return code in self.candidates


def merge_candidates(*sets: CandidateSet) -> CandidateSet:
    out = CandidateSet(system=sets[0].system if sets else "unknown")
    for s in sets:
        for code, cand in s.candidates.items():
            out.add(cand)
        for src, cands in s.provenance.items():
            out.provenance.setdefault(src, []).extend(cands)
    return out
