import pandas as pd

from term_map.candidates import crosswalk as cw
from term_map.candidates import exact as exact_src
from term_map.candidates import lexical as lex_src
from term_map.candidates import memory as mem_src
from term_map.candidates.fuzzy import FuzzyTermIndex, generate as fuzzy_generate
from term_map.normalize import build_variants
from term_map.terminology import Terminology


def test_exact_and_abbrev(term):
    v = build_variants("COPD", {})
    cs = exact_src.generate("snomed", v, term)
    codes = set(cs.candidates)
    assert "13645005" in codes
    cs_abbr = exact_src.generate("snomed", v, term)
    assert "abbrev" in cs_abbr.provenance
    assert "13645005" in {c.code for c in cs_abbr.provenance["abbrev"]}


def test_exact_icd(term):
    v = build_variants("chronic obstructive pulmonary disease, unspecified", {})
    cs = exact_src.generate("icd", v, term)
    assert "J44.9" in cs.candidates


def test_memory_exact_and_fuzzy():
    lookup = mem_src.TrainLookup(
        entity_norm=["htn", "degenerative disc disease"],
        icd=["I10", "M51.36"],
        snomed=["38341003", "77547008"],
    ).build_index()
    v = build_variants("HTN", {})
    cs = mem_src.generate("icd", v, lookup)
    assert "I10" in cs.candidates
    v2 = build_variants("degeneratve disc disease", {})
    cs2 = mem_src.generate("snomed", v2, lookup, fuzzy_min_score=80)
    assert any(c.source == "memory_fuzzy" for c in cs2.candidates.values())


def test_lexical_small():
    idx = lex_src.build_lexical_index(
        ["chronic obstructive pulmonary disease", "atrial fibrillation", "hypertension"],
        ["J44.9", "I48.91", "I10"],
        "icd",
        with_char=True,
        verbose=False,
    )
    res = idx.query(["copd chronic obstructive pulmonary disease"], top_k=2)
    assert res and res[0][0][0] == "J44.9"


def test_fuzzy():
    fidx = FuzzyTermIndex(["chronic obstructive pulmonary disease"], ["J44.9"], "icd")
    cs = fuzzy_generate("icd", ["chronic obstructiv pulmonry disease"], fidx,
                        top_k=5, min_score=80)
    assert "J44.9" in cs.candidates


def test_crosswalk(term):
    cs = cw.generate_for_icd(["13645005"], term)
    assert "J44.9" in cs.candidates
