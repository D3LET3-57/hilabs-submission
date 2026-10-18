from term_map.adjudicate import confidence as conf
from term_map.adjudicate.policy import extract_features
from term_map.adjudicate.rules import RulePack
from term_map.candidates.base import Candidate, CandidateSet


def cs_with(*items):
    cs = CandidateSet(system="icd")
    for code, source, score in items:
        cs.add(Candidate(code=code, source=source, score=score))
    return cs


def test_exact_unique_accept():
    cs = cs_with(("J44.9", "exact", 1.0), ("J44.8", "lexical", 0.4))
    d = conf.decide(cs)
    assert d.code == "J44.9"
    assert d.reason == "exact_unique"


def test_agreement_accept():
    cs = cs_with(("J44.9", "dense_gte_base", 0.9), ("J44.9", "lexical", 0.8),
                 ("J44.9", "memory_fuzzy", 0.9), ("J44.8", "fuzzy", 0.3))
    d = conf.decide(cs, margin=0.05)
    assert d.code == "J44.9"
    assert d.reason.startswith("agreement")


def test_fused_fallback():
    cs = cs_with(("I10", "lexical", 0.6), ("I11.9", "lexical", 0.5))
    d = conf.decide(cs)
    assert d.code == "I10"
    assert d.reason == "fused_max"


def test_rules_history_zcode(term):
    rp = RulePack("icd", term)
    cs = cs_with(("M51.36", "lexical", 0.6), ("Z87.39", "lexical", 0.58))
    scores = conf.fuse(cs)
    out = rp.rerank(scores, entity="history of degenerative disc disease", context="", cs=cs)
    assert out["Z87.39"] > out["M51.36"]
    assert any(f["rule"] == "icd_history_zcode" for f in rp.last_fired)


def test_rules_snomed_tag_penalty(term):
    rp = RulePack("snomed", term)
    cs = cs_with(("555555", "lexical", 0.6), ("77547008", "lexical", 0.6))
    scores = conf.fuse(cs)
    out = rp.rerank(scores, entity="degenerative disc disease", context="", cs=cs)
    assert out["77547008"] > out["555555"]
    assert any(f["rule"] == "snomed_tag_deprioritized" for f in rp.last_fired)


def test_policy_features_shape(term):
    cs = cs_with(("J44.9", "exact", 1.0), ("J44.8", "lexical", 0.4))
    scores = conf.fuse(cs)
    codes, X, names = extract_features(cs, scores, "copd", "patient with copd", term, "icd")
    assert len(codes) == 2
    assert X.shape[0] == 2
    assert X.shape[1] == len(names)
    assert X.dtype.kind == "f"
