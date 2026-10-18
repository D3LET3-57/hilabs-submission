import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from term_map.adjudicate.llm import LLMClient
from term_map.candidates.base import Candidate, CandidateSet


def make_cs(*items):
    cs = CandidateSet(system="icd")
    for code, score in items:
        cs.add(Candidate(code=code, source="lexical", score=score))
    return cs


class MockHandler(BaseHTTPRequestHandler):
    responses = []
    calls = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        MockHandler.calls.append(body)
        code = MockHandler.responses.pop(0)
        payload = {
            "choices": [{"message": {"content": json.dumps({"code": code, "reason": "mock"})}}]
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_choose_and_cache(mock_server, tmp_path):
    MockHandler.responses = ["J44.9", "J44.9"]
    MockHandler.calls = []
    client = LLMClient(
        base_url=mock_server, api_key="x", model="mock-1",
        cache_path=str(tmp_path / "cache.sqlite"),
    )
    cs = make_cs(("J44.9", 0.9), ("J44.8", 0.3))
    r1 = client.choose("icd", "copd", "context", cs)
    assert r1 is not None and r1.code == "J44.9" and not r1.cached
    n_calls = len(MockHandler.calls)
    r2 = client.choose("icd", "copd", "context", cs)
    assert r2 is not None and r2.code == "J44.9" and r2.cached
    assert len(MockHandler.calls) == n_calls  # cache hit, no new HTTP call


def test_reject_and_reask(mock_server, tmp_path):
    MockHandler.responses = ["NOT_A_CODE", "J44.8"]
    MockHandler.calls = []
    client = LLMClient(
        base_url=mock_server, api_key="x", model="mock-1",
        cache_path=str(tmp_path / "cache2.sqlite"),
    )
    cs = make_cs(("J44.9", 0.9), ("J44.8", 0.3))
    r = client.choose("icd", "copd", "context", cs)
    assert r is not None and r.code == "J44.8"
    assert len(MockHandler.calls) == 2


def test_self_consistency_majority(mock_server, tmp_path):
    MockHandler.responses = ["J44.9", "J44.8", "J44.9"]
    client = LLMClient(
        base_url=mock_server, api_key="x", model="mock-1",
        cache_path=str(tmp_path / "cache3.sqlite"),
    )
    cs = make_cs(("J44.9", 0.9), ("J44.8", 0.88))
    r = client.choose("icd", "copd", "context", cs, use_self_consistency=True)
    assert r is not None and r.code == "J44.9" and r.n_votes == 2


def test_no_base_url_returns_none(tmp_path):
    client = LLMClient(base_url="", cache_path=str(tmp_path / "c.sqlite"))
    cs = make_cs(("J44.9", 0.9))
    assert client.choose("icd", "copd", "", cs) is None
