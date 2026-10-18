import pandas as pd

from term_map.io import read_input, row_hash, RowCache


def test_read_input_aliases(tmp_path):
    p = tmp_path / "in.csv"
    pd.DataFrame({"chunk": ["c1"], "text": ["htn"], "context": ["severe htn"]}).to_csv(p, index=False)
    df = read_input(p)
    assert list(df.columns) == ["chunk_id", "clinical_entity", "chunk_text"]
    assert df.iloc[0]["clinical_entity"] == "htn"


def test_read_input_missing_entity(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"chunk_id": ["c1"]}).to_csv(p, index=False)
    try:
        read_input(p)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_row_cache(tmp_path):
    cache = RowCache(str(tmp_path / "rows.sqlite"))
    key = row_hash("c1", "htn", "context")
    assert cache.get(key) is None
    cache.put(key, {"code": "I10"})
    assert cache.get(key) == {"code": "I10"}


def test_splits_no_leakage_synthetic():
    from term_map.splits import build_splits

    pairs = pd.DataFrame({
        "entity_norm": ["htn", "htn", "af", "copd", "copd", "dm", "dm", "chf"],
        "chunk_id": ["c1", "c2", "c2", "c3", "c4", "c5", "c6", "c7"],
        "entity": ["HTN", "HTN", "AF", "COPD", "COPD", "DM", "DM", "CHF"],
    })
    out = build_splits(pairs)
    for s1 in ("train", "dev"):
        for s2 in ("dev", "test"):
            if s1 == s2:
                continue
            e1 = set(out[out["split"] == s1]["entity_norm"])
            e2 = set(out[out["split"] == s2]["entity_norm"])
            assert not (e1 & e2)
