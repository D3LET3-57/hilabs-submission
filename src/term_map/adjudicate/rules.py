"""Ordered rule pack per coding system (doc 06 section 2).

Rules are scored preferences (additive adjustments), never hard overrides.
They reference concept-name patterns resolved against loaded terminology,
not hardcoded codes. Every fired rule is recorded for provenance.
"""
from __future__ import annotations

import re

from ..normalize import basic_norm

# qualifier vocabulary
_LATERALITY = re.compile(r"\b(left|right|bilateral|bilat|unilateral|l\b|r\b)\b", re.I)
_SEVERITY = re.compile(r"\b(mild|moderate|severe|acute|chronic|subacute|intermittent|persistent|paroxysmal)\b", re.I)
_SUSPECT = re.compile(r"\b(possible|suspected|suspect|likely|probable|rule out|r/o|concern for)\b", re.I)
_HISTORY = re.compile(r"\b(history of|h/o|hx of|personal history|prior|previous)\b", re.I)
_DIABETES = re.compile(r"\b(diabet\w*|dm|iddm|niddm|t1dm|t2dm)\b", re.I)

_DIAB_NEURO = re.compile(r"\b(neuropathy|polyneuropathy|mononeuropathy|amyotrophy|autonomic)\b", re.I)

_UNSPEC_TERM = re.compile(r"\bunspecified\b|\bnot otherwise specified\b|\bnos\b", re.I)

# SNOMED semantic tags as trailing parenthetical
_SNOMED_TAG_RE = re.compile(r"\(([a-z][a-z ]+)\)\s*$")
_PREFERRED_TAGS = {"disorder", "finding", "disease", "syndrome"}


class RulePack:
    def __init__(self, system: str, term=None):
        self.system = system
        self.term = term
        self.last_fired: list[dict] = []

    # ------------------------------------------------------------------ utils
    def _display(self, code: str) -> str:
        if self.term is None:
            return ""
        disp = self.term.icd_display if self.system == "icd" else self.term.snomed_display
        return disp.get(code, "")

    def _has_specificity(self, text: str) -> bool:
        return bool(_LATERALITY.search(text) or _SEVERITY.search(text))

    # ------------------------------------------------------------------ rerank
    def rerank(self, scores: dict[str, float], entity: str, context: str,
               cs=None, term=None, weight: float = 1.0) -> dict[str, float]:
        self.last_fired = []
        if not scores:
            return scores
        ent_norm = basic_norm(entity)
        ctx_norm = basic_norm(context or "")
        combined = f"{ent_norm} {ctx_norm}"
        out = dict(scores)

        if self.system == "icd":
            self._icd_rules(out, ent_norm, ctx_norm, combined)
        else:
            self._snomed_rules(out, ent_norm, ctx_norm, combined)
        return out

    def _fire(self, rule: str, code: str, delta: float, note: str = ""):
        self.last_fired.append({"rule": rule, "code": code, "delta": round(delta, 4), "note": note})

    # ------------------------------------------------------------------ ICD
    def _icd_rules(self, scores: dict[str, float], ent: str, ctx: str, combined: str):
        specific = self._has_specificity(ent)
        hist = bool(_HISTORY.search(ent))
        susp = bool(_SUSPECT.search(ent))
        is_diab = bool(_DIABETES.search(ent)) and bool(_DIAB_NEURO.search(ent))

        for code in list(scores):
            disp = basic_norm(self._display(code))

            # R1: unspecified default preference when entity lacks specificity words
            if disp and not specific and not hist:
                if _UNSPEC_TERM.search(disp):
                    scores[code] += 0.045
                    self._fire("icd_unspecified_default", code, 0.045)
                elif _LATERALITY.search(disp) or _SEVERITY.search(disp):
                    scores[code] -= 0.02
                    self._fire("icd_penalize_specific_when_generic", code, -0.02)

            # R2: laterality preference only when entity names a side
            if disp:
                if _LATERALITY.search(ent):
                    ent_side = set(m.group(1).lower() for m in _LATERALITY.finditer(ent))
                    disp_side = set(m.group(1).lower() for m in _LATERALITY.finditer(disp))
                    if disp_side and not (ent_side & disp_side):
                        scores[code] -= 0.05
                        self._fire("icd_laterality_mismatch", code, -0.05)
                    elif disp_side and (ent_side & disp_side):
                        scores[code] += 0.03
                        self._fire("icd_laterality_match", code, 0.03)
                elif _LATERALITY.search(disp):
                    scores[code] -= 0.03
                    self._fire("icd_laterality_invented", code, -0.03)

            # R3: history-of -> Z-chapter preference (code-structural; no display needed)
            if hist:
                if code.startswith("Z"):
                    scores[code] += 0.08
                    self._fire("icd_history_zcode", code, 0.08)
                else:
                    scores[code] -= 0.04
                    self._fire("icd_history_non_z", code, -0.04)

            # R4: suspected -> Z03 family bias when encounter phrasing present
            if susp and code.startswith("Z03"):
                scores[code] += 0.06
                self._fire("icd_suspected_z03", code, 0.06)

            # R5: diabetic neuropathy templates
            if is_diab and disp and re.search(r"\bneuropathy\b", disp) and _DIAB_NEURO.search(disp):
                scores[code] += 0.05
                self._fire("icd_diabetic_neuropathy_family", code, 0.05)
        return scores

    # ------------------------------------------------------------------ SNOMED
    def _snomed_rules(self, scores: dict[str, float], ent: str, ctx: str, combined: str):
        susp = bool(_SUSPECT.search(ent))
        hist = bool(_HISTORY.search(ent))

        for code in list(scores):
            disp = self._display(code)
            disp_norm = basic_norm(disp)
            if not disp_norm:
                continue

            # R1: qualifier dropping (possible/suspected/likely) -> penalize qualifier concepts
            if susp and _SUSPECT.search(disp_norm):
                scores[code] -= 0.04
                self._fire("snomed_qualifier_drop", code, -0.04)

            # R2: do not force history concepts; gold prefers plain disorder
            if hist and re.search(r"\b(history of|h/o|personal history)\b", disp_norm):
                scores[code] -= 0.05
                self._fire("snomed_history_penalty", code, -0.05)

            # R3: semantic tag preference for disorder/finding over procedure/lab
            m = _SNOMED_TAG_RE.search(disp.lower())
            if m:
                tag = m.group(1)
                if tag in _PREFERRED_TAGS:
                    scores[code] += 0.02
                    self._fire("snomed_tag_preferred", code, 0.02)
                elif tag in {"procedure", "laboratory procedure", "situation", "observable entity",
                             "qualifier value", "substance", "product", "specimen", "morphologic abnormality"}:
                    scores[code] -= 0.03
                    self._fire("snomed_tag_deprioritized", code, -0.03)
        return scores
