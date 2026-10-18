from term_map.adjudicate import confidence as conf
from term_map.adjudicate.rerank import CrossEncoderReranker
from term_map.candidates.base import Candidate, CandidateSet


class DummyModel:
    def predict(self, pairs, batch_size=64):
        return [5.0 if "atrial fibrillation" in doc.lower() else -3.0 for _q, doc in pairs]


def test_rerank_reorders(term):
    r = CrossEncoderReranker()
    r._model = DummyModel()
    cs = CandidateSet(system="icd")
    cs.add(Candidate(code="I10", source="lexical", score=0.65))
    cs.add(Candidate(code="I48.91", source="lexical", score=0.60))
    disp = {"I48.91": "Unspecified atrial fibrillation", "I10": "Essential hypertension"}
    scores = conf.fuse(cs)
    trace = r.rerank("icd", cs, scores, "af", "patient with af", disp)
    assert trace and trace[0]["code"] == "I48.91"
    fused = conf.fuse(cs)
    assert fused["I48.91"] > fused["I10"]
    assert cs.sources_for("I48.91") == ["lexical", "rerank"] or \
        sorted(cs.sources_for("I48.91")) == ["lexical", "rerank"]


def test_rerank_unavailable_noop(term):
    r = CrossEncoderReranker()
    r._failed = True
    cs = CandidateSet(system="icd")
    cs.add(Candidate(code="I10", source="lexical", score=0.65))
    scores = conf.fuse(cs)
    assert r.rerank("icd", cs, scores, "htn", "", {"I10": "Hypertension"}) == []
    assert "rerank" not in cs.provenance
