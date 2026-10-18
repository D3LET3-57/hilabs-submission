"""Dense candidate source: FAISS index over terminology description embeddings.

Index layout (per model, under artifacts/indexes/<model_name>/):
  desc.index        FAISS IndexFlatIP (or IVF) over unique description vectors
  codes.npy         code per unique description row (np.str_)
  manifest.json     model id, dim, counts, terminology manifest hash

Query embeddings are produced by a sentence-transformers encoder (lazy, cached).
When torch/ST are unavailable, this source degrades to a no-op.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .base import Candidate, CandidateSet


@dataclass
class DenseModel:
    name: str
    hf_id: str
    dim: int
    query_prefix: str = ""
    passage_prefix: str = ""
    weight: float = 1.0
    index_dir: str | None = None
    _index: object = field(default=None, repr=False)
    _codes: np.ndarray | None = field(default=None, repr=False)

    def load(self, index_root: str) -> bool:
        import faiss

        # accept either <models_root> (index under <models_root>/<name>/) or the
        # system dir itself (<...>/<name>/<system>/desc.index)
        for d in (index_root, os.path.join(index_root, self.name)):
            index_path = os.path.join(d, "desc.index")
            codes_path = os.path.join(d, "codes.npy")
            if os.path.exists(index_path) and os.path.exists(codes_path):
                self._index = faiss.read_index(index_path)
                self._codes = np.load(codes_path, allow_pickle=True)
                self.index_dir = d
                return True
        return False

    @property
    def loaded(self) -> bool:
        return self._index is not None

    def search(self, q: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        if self._index is None:
            return []
        q = np.ascontiguousarray(q.astype(np.float32))
        sims, idxs = self._index.search(q, top_k)
        out: list[list[tuple[str, float]]] = []
        for row_sims, row_idxs in zip(sims, idxs):
            seen: dict[str, float] = {}
            for s, i in zip(row_sims, row_idxs):
                if i < 0:
                    continue
                c = self._codes[i]
                if c not in seen or s > seen[c]:
                    seen[c] = float(s)
            out.append(sorted(seen.items(), key=lambda kv: -kv[1]))
        return out


class DenseEncoder:
    """Lazy sentence-transformers encoder with optional precomputed query cache."""

    def __init__(self, hf_id: str, query_prefix: str = "", device: str | None = None,
                 cache_dir: str | None = None):
        self.hf_id = hf_id
        self.query_prefix = query_prefix
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        if cache_dir:
            self._load_cache()

    def _load_cache(self) -> None:
        import json
        import os

        texts_p = os.path.join(self.cache_dir, "query_texts.json")
        emb_p = os.path.join(self.cache_dir, "query_emb.f16.npy")
        if os.path.exists(texts_p) and os.path.exists(emb_p):
            texts = json.loads(open(texts_p).read())
            emb = np.load(emb_p).astype(np.float32)
            if emb.shape[0] == len(texts):
                self._cache = dict(zip(texts, emb))

    def _ensure(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.hf_id, device=self.device)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        pref = [self.query_prefix + t for t in texts]
        out = np.zeros((len(pref), 0), dtype=np.float32)
        miss_idx: list[int] = []
        miss_texts: list[str] = []
        if self._cache:
            rows = []
            for i, t in enumerate(pref):
                v = self._cache.get(t)
                if v is None:
                    miss_idx.append(i)
                    miss_texts.append(t)
                rows.append(v)
            dim = next((r.shape[0] for r in rows if r is not None), None)
            if dim is not None:
                out = np.zeros((len(pref), dim), dtype=np.float32)
                for i, v in enumerate(rows):
                    if v is not None:
                        out[i] = v
        else:
            miss_idx = list(range(len(pref)))
            miss_texts = pref
        if miss_texts:
            model = self._ensure()
            emb = model.encode(
                miss_texts, batch_size=batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            ).astype(np.float32)
            out = emb if out.shape[1] == 0 else out
            for k, i in enumerate(miss_idx):
                out[i] = emb[k]
        return out


def generate(
    system: str,
    query_texts: list[str],
    models: list[DenseModel],
    encoder_map: dict[str, DenseEncoder],
    top_k: int = 200,
) -> CandidateSet:
    cs = CandidateSet(system=system)
    if not models:
        return cs
    best: dict[str, tuple[float, str]] = {}   # code -> (score, source)
    for m in models:
        if not m.loaded:
            continue
        enc = encoder_map.get(m.name)
        if enc is None:
            continue
        q = enc.encode(query_texts)
        for res in m.search(q, top_k):
            for code, score in res:
                wscore = float(score) * m.weight
                if code not in best or wscore > best[code][0]:
                    best[code] = (wscore, f"dense_{m.name}")
    for rank, (code, (score, src)) in enumerate(sorted(best.items(), key=lambda kv: -kv[1][0])):
        cs.add(Candidate(code=code, source=src, score=score, rank=rank))
    return cs
