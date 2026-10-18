import numpy as np
import pandas as pd

from term_map.candidates.dense import DenseModel


def _make_index(root, name, system):
    import faiss

    d = root / name / system
    d.mkdir(parents=True)
    index = faiss.IndexFlatIP(4)
    index.add(np.eye(4, dtype=np.float32))
    faiss.write_index(index, str(d / "desc.index"))
    np.save(d / "codes.npy", np.asarray(["A00", "B00", "C00", "D00"]))
    return d


def test_dense_load_accepts_system_dir_and_model_root(tmp_path):
    _make_index(tmp_path, "m1", "snomed")
    dm = DenseModel(name="m1", hf_id="x", dim=4)
    assert dm.load(str(tmp_path / "m1" / "snomed")) is True
    assert dm.loaded

    dm2 = DenseModel(name="m1", hf_id="x", dim=4)
    assert dm2.load(str(tmp_path / "m1")) is False  # no desc.index at model root

    dm3 = DenseModel(name="m1", hf_id="x", dim=4)
    assert dm3.load(str(tmp_path)) is False


def test_engine_config_overrides():
    from term_map.pipeline import Engine, EngineResources

    cfg = {
        "run": {"context_window": 40},
        "fusion": {"source_weights": {"lexical": 0.5}},
        "rules": {"policy_blend": 0.6},
        "rerank": {"enabled": False, "weight": 0.4, "gate_margin": 0.1},
    }
    eng = Engine(EngineResources(cfg=cfg, term=None), use_rerank=None)
    assert eng.source_weights["lexical"] == 0.5
    assert eng.source_weights["exact"] == 1.0            # defaults preserved
    assert eng.source_weights["rerank"] == 0.4
    assert eng.policy_blend == 0.6
    assert eng.rerank_gate == 0.1
    assert eng.reranker is None                          # disabled in config


def test_terminology_map_filters_junk_targets(tmp_path):
    from term_map.terminology import load_terminology

    snomed = pd.DataFrame({"code": ["100", "101"], "description": ["X", "Y"]})
    icd = pd.DataFrame({"code": ["A00", "B00.1"], "description": ["a", "b"]})
    m = pd.DataFrame({
        "snomed_code": ["100", "100", "101", "101"],
        "icd_code": ["A00", "\\N", "B001", "T07.XXX?"],
    })
    sp, ip, mp = (tmp_path / "s.parquet"), (tmp_path / "i.parquet"), (tmp_path / "m.parquet")
    snomed.to_parquet(sp)
    icd.to_parquet(ip)
    m.to_parquet(mp)
    term = load_terminology(sp, ip, mp)
    assert len(term.map_df) == 2
    assert term.map_by_snomed["101"] == ["B00.1"]
    assert "T07.XXX?" not in term.map_by_snomed.get("101", [])
