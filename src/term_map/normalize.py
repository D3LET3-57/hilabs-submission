"""Normalization cascade: chunk cleaning, entity normalization, abbreviation handling.

All functions are deterministic; raw text is never mutated (variants are produced).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------- chunk cleaning

_HEADER_RE = re.compile(r"header_\d+\s*:")
_PICTURE_RE = re.compile(r"==>.*?<==", flags=re.IGNORECASE | re.DOTALL)
_BR_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_COLONSEP_RE = re.compile(r"\s*::\s*")
_HTML_RE = re.compile(r"<[^>]{0,80}>")
_WS_RE = re.compile(r"\s+")


def clean_chunk(text: str) -> str:
    """Strip chart scaffolding: header_N: markers, :: separators, <br>, picture tags."""
    if not isinstance(text, str):
        return ""
    t = _PICTURE_RE.sub(" ", text)
    t = _BR_RE.sub(" ", t)
    t = _COLONSEP_RE.sub(" ", t)
    t = _HEADER_RE.sub(" ", t)
    t = _HTML_RE.sub(" ", t)
    t = t.replace("\u00a0", " ")
    return _WS_RE.sub(" ", t).strip()


def extract_context(chunk: str, entity: str, window: int = 300) -> str:
    """Cleaned ±window-char context around the first (case-insensitive) entity occurrence."""
    cleaned = clean_chunk(chunk)
    if not cleaned:
        return ""
    idx = cleaned.lower().find(entity.lower())
    if idx < 0:
        return cleaned[: 2 * window]
    start = max(0, idx - window)
    end = min(len(cleaned), idx + len(entity) + window)
    return cleaned[start:end]


# ---------------------------------------------------------------- entity norms

_UNICODE_FIX = {
    "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-", "\u00b5": "u",
}

# numeric forms seen in charts: "1-2+", "2+", "1,2", "2/2" (= secondary to)
_PLUS_NUM_RE = re.compile(r"(\d+)\s*-\s*(\d+)\s*\+")
_NUM_PLUS_RE = re.compile(r"(\d+)\s*\+")
_SLASH_SECONDARY_RE = re.compile(r"\b2\s*/\s*2\b")

_PUNCT_KEEP_INTERNAL = re.compile(r"[^a-z0-9\-\+'/. ]")
_MULTI_DASH = re.compile(r"-{2,}")


def basic_norm(text: str) -> str:
    """Lowercase, unicode-normalize, collapse punctuation & whitespace. Keeps dots (codes)."""
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    for k, v in _UNICODE_FIX.items():
        t = t.replace(k, v)
    t = t.lower()
    t = _SLASH_SECONDARY_RE.sub(" secondary to ", t)
    t = _PLUS_NUM_RE.sub(r"\1-\2 +", t)
    t = _NUM_PLUS_RE.sub(r"\1 +", t)
    t = _PUNCT_KEEP_INTERNAL.sub(" ", t)
    t = _MULTI_DASH.sub("-", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def strip_trailing_punct(t: str) -> str:
    return t.strip(" -.'/+")


def fmt_icd(code: str) -> str:
    """Canonical ICD-10-CM format: uppercase, dotted after the 3rd char when longer."""
    if not isinstance(code, str):
        return ""
    c = code.strip().upper()
    if not c:
        return c
    if "." in c or "-" in c:
        return c
    if len(c) > 3:
        return c[:3] + "." + c[3:]
    return c


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen and it:
            seen.add(it)
            out.append(it)
    return out


# ---------------------------------------------------------------- abbreviations

_ABBR_SPLIT_RE = re.compile(r"^(.{1,14}?)\s*-\s+(\S.*)$")


def split_abbr_description(desc: str) -> tuple[str | None, str]:
    """'COPD - Chronic obstructive pulmonary disease' -> ('copd', 'chronic ...').

    Conservative: prefix must be an acronym-like token (no lowercase letters beyond
    a leading 'pH'-style pattern is rejected: we require >=1 uppercase and <=14 chars,
    and the long form must start with an uppercase letter).
    """
    if not isinstance(desc, str):
        return None, ""
    m = _ABBR_SPLIT_RE.match(desc.strip())
    if not m:
        return None, desc
    abbr, long = m.group(1).strip(), m.group(2).strip()
    if not long or not long[0].isalnum():
        return None, desc
    letters = [c for c in abbr if c.isalpha()]
    if not letters:
        return None, desc
    n_upper = sum(1 for c in abbr if c.isupper())
    n_lower = sum(1 for c in abbr if c.islower())
    if n_upper == 0:
        return None, desc
    # reject word-like prefixes ("Fall", "Acting out") unless clearly acronym-shaped
    if n_lower > 0 and not (n_upper >= 2 and len(abbr) <= 6):
        return None, desc
    if " " in abbr and len(abbr.split()) > 2:
        return None, desc
    return basic_norm(abbr), long


# ---------------------------------------------------------------- variant builder

@dataclass
class Variants:
    raw: str
    cleaned: str
    normalized: str                 # basic_norm
    no_dot: str                     # dots removed
    expanded: list[str] = field(default_factory=list)   # abbreviation expansions
    corrected: str | None = None    # typo-corrected form


def build_variants(entity: str, abbrev_graph: dict[str, list[str]] | None = None,
                   corrector=None) -> Variants:
    raw = entity
    cleaned = _WS_RE.sub(" ", str(entity).strip()) if isinstance(entity, str) else ""
    norm = basic_norm(cleaned)
    dedot = norm.replace(".", "")
    expanded: list[str] = []
    if abbrev_graph:
        # whole-string abbreviation hit
        for key in (norm, dedot, strip_trailing_punct(norm)):
            if key in abbrev_graph:
                expanded.extend(basic_norm(x) for x in abbrev_graph[key])
        # token-level: expand the last token if it is an abbreviation seen in the graph
        toks = dedot.split()
        if toks:
            last = toks[-1]
            if last in abbrev_graph and len(toks) > 1:
                head = " ".join(toks[:-1])
                expanded.extend(f"{head} {basic_norm(x)}" for x in abbrev_graph[last])
    corrected = None
    if corrector is not None:
        try:
            corrected = corrector(dedot)
        except Exception:
            corrected = None
    return Variants(
        raw=raw,
        cleaned=cleaned,
        normalized=dedupe_keep_order([norm, dedot]),
        no_dot=dedot,
        expanded=dedupe_keep_order(expanded),
        corrected=corrected if corrected and corrected != dedot else None,
    )


def all_query_forms(v: Variants) -> list[str]:
    out: list[str] = []
    out.extend(v.normalized)
    out.extend(v.expanded[:4])
    if v.corrected:
        out.append(v.corrected)
    return dedupe_keep_order(out)
