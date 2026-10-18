"""Learned policy model: pointwise candidate ranker trained on train-split labels.

Improvement over the doc's narrow unspecified-vs-specific calibration (doc 06
section 3): a pointwise gradient-boosted ranker over full candidate features
implements annotator-policy imitation in one learned layer, while remaining
trainable per system and frozen with the split hash.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from ..candidates.base import CandidateSet
from ..normalize import basic_norm
from .rules import _HISTORY, _LATERALITY, _SEVERITY, _SUSPECT, _SNOMED_TAG_RE, _UNSPEC_TERM

SOURCE_FLAGS = [
    "exact", "abbrev", "memory_exact", "memory_fuzzy", "lexical", "lexical_char",
    "fuzzy", "crosswalk", "rerank", "dense_gte_base", "dense_sapbert", "dense_biolord",
]

_SNOMED_TAGS = [
    "disorder", "finding", "procedure", "situation", "observable entity",
    "qualifier value", "substance", "product", "morphologic abnormality",
    "laboratory procedure", "disease", "syndrome",
]

NUM_RE = re.compile(r"\b\d+\b")


def _ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    from rapidfuzz import fuzz

    return fuzz.token_set_ratio(a, b) / 100.0


def extract_features(
    cs: CandidateSet,
    scores: dict[str, float],
    entity: str,
    context: str,
    term,
    system: str,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (codes, feature matrix, feature names) for the candidate set."""
    ent_norm = basic_norm(entity)
    ctx_norm = basic_norm(context or "")
    ent_tokens = set(ent_norm.split())
    ent_susp = 1.0 if _SUSPECT.search(ent_norm) else 0.0
    ent_hist = 1.0 if _HISTORY.search(ent_norm) else 0.0
    ent_lat = 1.0 if _LATERALITY.search(ent_norm) else 0.0
    ent_sev = 1.0 if _SEVERITY.search(ent_norm) else 0.0
    ctx_susp = 1.0 if _SUSPECT.search(ctx_norm) else 0.0
    ent_nums = len(NUM_RE.findall(ent_norm))

    disp_map = (term.icd_display if system == "icd" else term.snomed_display) if term is not None else {}

    # per-source ranks
    ranks: dict[str, dict[str, int]] = {}
    for src, cands in cs.provenance.items():
        for c in cands:
            r = ranks.setdefault(c.code, {})
            if src not in r or c.rank < r[src]:
                r[src] = c.rank
    src_scores: dict[str, dict[str, float]] = {}
    for src, cands in cs.provenance.items():
        for c in cands:
            m = src_scores.setdefault(c.code, {})
            if c.score > m.get(src, -1):
                m[src] = c.score

    codes: list[str] = []
    rows: list[list[float]] = []
    for code in scores:
        disp_raw = disp_map.get(code, "")
        disp = basic_norm(disp_raw)
        disp_tokens = set(disp.split())
        overlap = len(ent_tokens & disp_tokens) / max(1, len(ent_tokens))
        f: list[float] = []
        f.append(scores[code])                                   # fused score
        f.append(1.0 if code in cs.candidates else 0.0)
        for src in SOURCE_FLAGS:
            f.append(src_scores.get(code, {}).get(src, 0.0))     # source score
        f.append(sum(1 for src in SOURCE_FLAGS if code in src_scores and src in cs.provenance))  # n sources
        f.append(min(ranks.get(code, {}).values(), default=999) / 100.0)   # best rank
        f.append(_ratio(ent_norm, disp))                         # fuzzy similarity entity vs display
        f.append(_ratio(ent_norm, disp) ** 2)
        f.append(overlap)
        f.append(len(disp_tokens) / 20.0)
        f.append(len(ent_tokens) / 20.0)
        f.append(1.0 if ent_norm == disp else 0.0)
        fol = [t for t in disp_tokens if t in ent_norm]
        f.append(len(fol) / max(1, len(disp_tokens)))            # long overlap fraction
        f.append(len(code) / 10.0)
        if system == "icd":
            f.append(1.0 if "." in code else 0.0)
            f.append(1.0 if len(code) > 5 else 0.0)              # 7th char present
            f.append(ord(code[0]) - ord("A"))                    # chapter letter ordinal
            f.append(1.0 if code.endswith(".9") or code.endswith(".90") else 0.0)
            f.append(1.0 if code.startswith("Z") else 0.0)
            f.append(1.0 if code.startswith("R") else 0.0)
            f.append(1.0 if _UNSPEC_TERM.search(disp) else 0.0)
            f.append(1.0 if re.search(r"unspecified", disp) else 0.0)
            f.append(0.0)
            f.append(0.0)
            f.append(0.0)
        else:
            m = _SNOMED_TAG_RE.search(disp_raw.lower()) if disp_raw else None
            tag = m.group(1) if m else ""
            for t in ["disorder", "finding", "procedure", "situation", "disease", "syndrome"]:
                f.append(1.0 if tag == t else 0.0)
            f.append(1.0 if re.search(r"\b(history of|h/o|personal history)\b", disp) else 0.0)
            f.append(0.0)
            f.append(0.0)
            f.append(0.0)
            f.append(0.0)
        f.append(ent_susp)
        f.append(ent_hist)
        f.append(ent_lat)
        f.append(ent_sev)
        f.append(ctx_susp)
        f.append(float(ent_nums))
        # qualifier agreement: entity has susp/hist and display carries same word
        f.append(1.0 if (ent_susp and _SUSPECT.search(disp)) else 0.0)
        f.append(1.0 if (ent_hist and re.search(r"\b(history of|h/o)\b", disp)) else 0.0)
        codes.append(code)
        rows.append(f)
    X = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
    return codes, X, _feature_names(system)


def _feature_names(system: str) -> list[str]:
    names = ["fused_score", "in_candidates"]
    names += [f"src_score_{s}" for s in SOURCE_FLAGS]
    names += ["n_sources", "best_rank", "similarity", "similarity_sq", "overlap", "disp_len",
              "ent_len", "exact_match", "long_overlap", "code_len"]
    if system == "icd":
        names += ["has_dot", "long_code", "chapter", "ends_9", "z_chapter", "r_chapter",
                  "unspec_term", "unspec_word", "pad1", "pad2", "pad3"]
    else:
        names += ["tag_disorder", "tag_finding", "tag_procedure", "tag_situation",
                  "tag_disease", "tag_syndrome", "history_display", "pad1", "pad2", "pad3", "pad4"]
    names += ["ent_susp", "ent_hist", "ent_lat", "ent_sev", "ctx_susp", "ent_nums",
              "agree_susp", "agree_hist"]
    return names


@dataclass
class PolicyModel:
    system: str
    model: object
    feature_names: list[str]
    blend: float = 0.45

    def predict_quality(self, codes: list[str], X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.zeros(0)
        proba = self.model.predict_proba(X)[:, 1]
        return proba

    def adjust(self, cs: CandidateSet, scores: dict[str, float], entity: str,
               context: str, term=None) -> dict[str, float] | None:
        codes, X, names = extract_features(cs, scores, entity, context, term, self.system)
        if X.shape[0] == 0:
            return None
        proba = self.predict_quality(codes, X)
        # blend fused score with learned probability (prob in [0,1])
        out = {}
        for code, p in zip(codes, proba):
            out[code] = (1.0 - self.blend) * scores[code] + self.blend * float(p)
        return out

    def save(self, path: str) -> None:
        import joblib

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "PolicyModel | None":
        try:
            import joblib

            return joblib.load(path)
        except Exception:
            return None
