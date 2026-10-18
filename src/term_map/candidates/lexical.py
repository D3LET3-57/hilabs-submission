"""Lexical candidate generation: word TF-IDF (1-2 gram) + optional char 3-5 gram TF-IDF.

Batched sparse dot products (scipy) give exact cosine top-K for ~1M descriptions
quickly on CPU. Char n-grams add misspelling/OCR robustness.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from ..normalize import basic_norm, strip_trailing_punct


@dataclass
class LexicalIndex:
    system: str
    codes: np.ndarray                    # code per unique description row
    word_vectorizer: TfidfVectorizer
    word_matrix: sp.csr_matrix
    char_vectorizer: TfidfVectorizer | None = None
    char_matrix: sp.csr_matrix | None = None

    def save(self, path: str, compress: int = 0) -> None:
        import joblib

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path, compress=compress)

    @staticmethod
    def load(path: str) -> "LexicalIndex":
        import joblib

        return joblib.load(path)

    # ------------------------------------------------------------------ query
    def query(self, texts: list[str], top_k: int = 200,
              char_weight: float = 0.35) -> list[list[tuple[str, float]]]:
        if not texts:
            return []
        q = self._transform(self.word_vectorizer, texts)
        sims = (q @ self.word_matrix.T).toarray()  # (nq, ndesc) dense chunk
        if self.char_matrix is not None and char_weight > 0:
            qc = self._transform(self.char_vectorizer, texts)
            simsc = (qc @ self.char_matrix.T).toarray()
            sims = (1.0 - char_weight) * sims + char_weight * simsc
        out: list[list[tuple[str, float]]] = []
        k = min(top_k, sims.shape[1])
        for i in range(sims.shape[0]):
            row = sims[i]
            if k >= row.shape[0]:
                idx = np.argsort(-row)[:k]
            else:
                part = np.argpartition(-row, k)[:k]
                idx = part[np.argsort(-row[part])]
            seen: dict[str, float] = {}
            for j in idx:
                s = float(row[j])
                if s <= 0:
                    continue
                c = self.codes[j]
                if c not in seen or s > seen[c]:
                    seen[c] = s
            out.append(sorted(seen.items(), key=lambda kv: -kv[1]))
        return out

    @staticmethod
    def _transform(vec: TfidfVectorizer, texts: list[str]) -> sp.csr_matrix:
        normed = [strip_trailing_punct(basic_norm(t)) or " " for t in texts]
        return vec.transform(normed)


def build_lexical_index(
    descriptions: list[str],
    codes: list[str],
    system: str,
    with_char: bool = True,
    word_max_features: int | None = None,
    char_max_features: int = 400_000,
    min_df: int = 2,
    verbose: bool = True,
) -> LexicalIndex:
    # dedupe identical normalized descriptions, keep all codes
    uniq: dict[str, list[str]] = {}
    for d, c in zip(descriptions, codes):
        n = strip_trailing_punct(basic_norm(d))
        if not n:
            continue
        uniq.setdefault(n, []).append(c)
    texts = list(uniq.keys())
    row_codes: list[str] = []
    for t in texts:
        row_codes.append(uniq[t][0])
    if verbose:
        print(f"[lexical:{system}] unique descriptions: {len(texts):,}")

    def _fit(vec: TfidfVectorizer, docs: list[str]) -> "sp.csr_matrix":
        try:
            return vec.fit_transform(docs)
        except ValueError:
            # tiny test corpora prune everything at min_df>1
            vec.set_params(min_df=1)
            return vec.fit_transform(docs)

    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
        min_df=min_df, max_features=word_max_features, dtype=np.float32,
        strip_accents="unicode", token_pattern=r"[a-z0-9][a-z0-9\-\.]{1,}",
    )
    mat = _fit(vec, texts)
    if verbose:
        print(f"[lexical:{system}] word vocab={len(vec.vocabulary_):,} nnz={mat.nnz:,}")

    cvec = None
    cmat = None
    if with_char:
        cvec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
            min_df=min_df, max_features=char_max_features, dtype=np.float32,
            strip_accents="unicode",
        )
        cmat = _fit(cvec, texts)
        if verbose:
            print(f"[lexical:{system}] char vocab={len(cvec.vocabulary_):,} nnz={cmat.nnz:,}")

    return LexicalIndex(
        system=system,
        codes=np.array(row_codes, dtype=object),
        word_vectorizer=vec,
        word_matrix=mat.tocsr(),
        char_vectorizer=cvec,
        char_matrix=cmat.tocsr() if cmat is not None else None,
    )


def generate(system: str, texts: list[str], index: LexicalIndex, top_k: int,
             source_tag: str = "lexical", min_score: float = 0.05):
    from .base import Candidate, CandidateSet

    cs = CandidateSet(system=system)
    if index is None or not texts:
        return cs
    results = index.query(texts, top_k=top_k)
    best: dict[str, float] = {}
    for res in results:
        for code, score in res:
            if score >= min_score and score > best.get(code, -1):
                best[code] = score
    for rank, (code, score) in enumerate(sorted(best.items(), key=lambda kv: -kv[1])):
        cs.add(Candidate(code=code, source=source_tag, score=score, rank=rank))
    return cs
