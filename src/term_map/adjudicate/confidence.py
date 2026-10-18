"""Score fusion, acceptance conditions, and the deterministic choice.

Fusion = weighted max per source + agreement bonus. Acceptance is layered:
  exact unique > memory exact > multi-source agreement > rules/policy > fused max.
Every decision returns a trace dict used for provenance and error analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..candidates.base import CandidateSet

DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "exact": 1.00,
    "abbrev": 0.93,
    "memory_exact": 1.00,
    "memory_fuzzy": 0.88,
    "memory_dense": 0.85,
    "dense_gte_base": 0.86,
    "dense_sapbert": 0.84,
    "dense_ultrabert_embed": 0.84,
    "dense_biolord": 0.84,
    "lexical": 0.72,
    "lexical_char": 0.68,
    "fuzzy": 0.74,
    "crosswalk": 0.70,
    "rerank": 0.90,
    "structural": 0.55,
    "judge": 0.75,
    "llm": 1.20,
    "llm_sc": 1.25,
}

INDEPENDENT_SOURCES = {
    "exact", "abbrev", "memory_exact", "memory_fuzzy", "lexical", "lexical_char",
    "fuzzy", "crosswalk", "rerank", "judge", "dense_gte_base", "dense_sapbert",
    "dense_ultrabert_embed", "dense_biolord",
}


@dataclass
class Decision:
    code: str | None
    confidence: float
    reason: str
    fused: dict[str, float] = field(default_factory=dict)
    trace: dict = field(default_factory=dict)


def fuse(cs: CandidateSet, weights: dict[str, float] | None = None) -> dict[str, float]:
    w = weights or DEFAULT_SOURCE_WEIGHTS
    scores: dict[str, float] = {}
    support: dict[str, set[str]] = {}
    for src, cands in cs.provenance.items():
        sw = w.get(src, 0.5)
        for c in cands:
            s = sw * c.score
            if s > scores.get(c.code, -1):
                scores[c.code] = s
            support.setdefault(c.code, set()).add(src)
    # agreement bonus: up to +0.06 for 3+ independent sources
    for code, srcs in support.items():
        n = len(srcs & INDEPENDENT_SOURCES)
        if n >= 2:
            scores[code] = scores[code] + min(0.06, 0.03 * (n - 1))
    return scores


def decide(cs: CandidateSet, policy=None, rules=None, context: str = "",
           entity: str = "", weights: dict[str, float] | None = None,
           margin: float = 0.05, term=None) -> Decision:
    """Deterministic layered choice. `policy`/`rules` optional rank adjusters (M2)."""
    scores = fuse(cs, weights)
    if not scores:
        return Decision(code=None, confidence=0.0, reason="empty", fused=scores)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_code, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin_abs = top_score - second_score
    trace = {
        "n_candidates": len(scores),
        "top3": [(c, round(s, 4)) for c, s in ranked[:3]],
        "margin": round(margin_abs, 4),
    }

    exact_codes = [c.code for c in cs.provenance.get("exact", []) if c.score >= 1.0]
    if len(exact_codes) == 1:
        return Decision(code=exact_codes[0], confidence=min(1.0, 0.99),
                        reason="exact_unique", fused=scores, trace=trace)

    if "memory_exact" in cs.provenance:
        mem = [c.code for c in cs.provenance["memory_exact"] if c.score >= 1.0]
        if len(mem) == 1:
            return Decision(code=mem[0], confidence=0.97, reason="memory_exact",
                            fused=scores, trace=trace)

    # multi-source agreement: top has >=2 independent sources and clear margin
    if margin_abs >= margin:
        srcs = cs.sources_for(top_code)
        n_ind = len(set(srcs) & INDEPENDENT_SOURCES)
        if n_ind >= 3 and top_score >= 0.7:
            return Decision(code=top_code, confidence=min(0.99, 0.8 + 0.05 * n_ind),
                            reason=f"agreement_{n_ind}", fused=scores, trace=trace)

    # rules + policy re-rank
    if rules is not None:
        scores = rules.rerank(scores, entity=entity, context=context, cs=cs, term=getattr(rules, "term", None))
        trace["rules"] = getattr(rules, "last_fired", [])
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        top_code, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin_abs = top_score - second_score
        trace["margin_after_rules"] = round(margin_abs, 4)

    if policy is not None:
        adj = policy.adjust(cs, scores, entity=entity, context=context, term=term)
        if adj:
            scores = adj
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
            top_code, top_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            margin_abs = top_score - second_score
            trace["policy_applied"] = True

    conf = top_score
    if margin_abs >= margin:
        conf = min(1.0, top_score + 0.05)
    return Decision(code=top_code, confidence=round(conf, 4),
                    reason="fused_max", fused=scores, trace=trace)
