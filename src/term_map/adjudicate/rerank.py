"""Cross-encoder reranking stage (doc 02 section 3: "reranking (high precision):
cross-encoder or LLM adjudication with context").

The reranker scores (entity + context window) against each top candidate's display
and re-injects the scores as a `rerank` candidate source, so fusion, rules, policy,
and provenance all see it uniformly. Degrades to a no-op when the model is
unavailable (offline env) so the engine keeps working deterministically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..candidates.base import Candidate, CandidateSet

DEFAULT_MODEL = "ncbi/MedCPT-Cross-Encoder"


@dataclass
class CrossEncoderReranker:
    model_id: str = DEFAULT_MODEL
    device: str | None = None
    top_n: int = 30
    max_length: int = 256
    batch_size: int = 64
    _model: object = field(default=None, repr=False)
    _failed: bool = field(default=False, repr=False)

    def available(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            self._ensure()
            return True
        except Exception:
            self._failed = True
            return False

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, device=self.device,
                                       max_length=self.max_length)
        return self._model

    @staticmethod
    def _to_prob(x: float) -> float:
        # logistic map; identity when the model already returns probabilities
        if 0.0 <= x <= 1.0:
            return x
        return 1.0 / (1.0 + math.exp(-x))

    def rerank(self, system: str, cs: CandidateSet, scores: dict[str, float],
               entity: str, context: str, disp_map: dict[str, str],
               min_top_n: int = 3) -> list[dict]:
        """Add `rerank` provenance candidates; returns fired-trace rows."""
        if not scores or not self.available():
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: self.top_n]
        top_n = len(ranked)
        query = (entity or "").strip()
        if context:
            query = f"{query} . {context[:400]}"
        pairs = []
        codes = []
        for code, _ in ranked[:top_n]:
            disp = disp_map.get(code, "") or code
            pairs.append((query, disp))
            codes.append(code)
        if not pairs:
            return []
        model = self._ensure()
        raw = model.predict(pairs, batch_size=self.batch_size)
        trace = []
        for code, r in zip(codes, raw):
            p = self._to_prob(float(r))
            cs.add(Candidate(code=code, source="rerank", score=p, rank=0))
            trace.append({"code": code, "score": round(p, 4)})
        trace.sort(key=lambda d: -d["score"])
        return trace
