from term_map.normalize import (
    Variants, basic_norm, build_variants, clean_chunk, extract_context, fmt_icd,
    split_abbr_description,
)


def test_clean_chunk_strips_scaffolding():
    raw = "header_1: :: header_2: htn:: ==>: picture (some) <== <br> started 10/23. No edema."
    out = clean_chunk(raw)
    assert "header_" not in out
    assert "::" not in out
    assert "<br>" not in out
    assert "picture" not in out
    assert "htn" in out


def test_basic_norm_keeps_dot_and_unifies_numeric():
    assert basic_norm("1-2+ BILATERAL edema") == "1-2 + bilateral edema"
    assert basic_norm("E11.40") == "e11.40"
    assert basic_norm("2/2 CHF") == "secondary to chf"


def test_fmt_icd():
    assert fmt_icd("I4891") == "I48.91"
    assert fmt_icd("i48.91") == "I48.91"
    assert fmt_icd("I10") == "I10"
    assert fmt_icd("S72.001A") == "S72.001A"


def test_split_abbr_description():
    abbr, long = split_abbr_description("COPD - Chronic obstructive pulmonary disease")
    assert abbr == "copd"
    assert long.startswith("Chronic")
    assert split_abbr_description("No abbreviation here")[0] is None
    assert split_abbr_description("Fall - accidental")[0] is None
    abbr2, _ = split_abbr_description("AF - Paroxysmal atrial fibrillation")
    assert abbr2 == "af"


def test_extract_context_windows():
    entity = "htn"
    chunk = "a" * 500 + " patient with htn and chf " + "b" * 500
    ctx = extract_context(clean_chunk(chunk), entity, window=30)
    assert "htn" in ctx and "chf" in ctx
    assert len(ctx) < 120


def test_build_variants_expands_abbreviation():
    graph = {"copd": ["chronic obstructive pulmonary disease"]}
    v = build_variants("COPD", graph)
    assert v.no_dot == "copd"
    assert any("chronic obstructive" in e for e in v.expanded)
