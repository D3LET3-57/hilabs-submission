"""OpenAI-compatible LLM adjudication client.

- provider-agnostic (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL)
- candidate-constrained: output code must be in the candidate list, else re-ask once
- JSON validated; retries with backoff; SQLite cache keyed by content hash
- temperature 0; optional self-consistency (n samples at higher temperature)
- thread-safe enough for executor use (cache writes serialized)
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass

import requests

from ..candidates.base import CandidateSet

PROMPT_VERSION = "v1"

SYSTEM_PROMPTS = {
    "icd": (
        "You are a certified medical coder. You choose exactly ONE ICD-10-CM code for a "
        "clinical term from the provided candidate list. You never invent codes."
    ),
    "snomed": (
        "You are a certified medical terminologist. You choose exactly ONE SNOMED CT "
        "concept id for a clinical term from the provided candidate list. You never invent ids."
    ),
}

RULES_EXCERPT = {
    "icd": (
        "ICD conventions: unqualified conditions map to their unspecified/default leaf; "
        "do not invent laterality or severity not present in the term/context; "
        "'history of X' maps to a personal-history Z code; 'suspected X' in an observation "
        "encounter maps to Z03.-; symptoms map to R-chapter codes."
    ),
    "snomed": (
        "SNOMED conventions: prefer disorder/finding concepts over procedure/lab concepts "
        "for diagnosis text; qualifiers like possible/suspected are usually dropped if no "
        "explicit suspected-context concept exists; history-of text usually maps to the "
        "plain disorder unless a genuine history concept is the best fit."
    ),
}


@dataclass
class LLMResult:
    code: str
    reason: str
    n_votes: int
    raw: dict | None = None
    cached: bool = False


def _select_candidates(cs: CandidateSet, top: int, per_source_k: int = 0):
    """Top candidates for the judge. per_source_k>0 unions each source's own
    top-k (source-diverse: much better recall per option than fused ranking)."""
    if not per_source_k:
        return cs.top(top)
    chosen: list = []
    seen: set[str] = set()
    for src_cands in cs.provenance.values():
        for c in sorted(src_cands, key=lambda c: -c.score)[:per_source_k]:
            if c.code not in seen:
                seen.add(c.code)
                chosen.append(c)
    return chosen[: max(top, 120)]


def _candidate_block(cs: CandidateSet, top: int = 20, disp_map: dict | None = None,
                     per_source_k: int = 0) -> tuple[str, set[str]]:
    cands = _select_candidates(cs, top, per_source_k)
    lines = []
    codes = set()
    for i, c in enumerate(cands, 1):
        codes.add(c.code)
        disp = (disp_map or {}).get(c.code, "")
        srcs = ",".join(sorted(set(cs.sources_for(c.code))))
        lines.append(f"{i}. {c.code} | {disp} | sources={srcs} | score={c.score:.3f}")
    return "\n".join(lines), codes


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        cache_path: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_tokens: int = 512,
        per_source_k: int = 0,
        max_concurrency: int = 4,
        self_consistency_n: int = 3,
        self_consistency_temperature: float = 0.3,
        self_consistency_margin: float = 0.15,
        prompt_version: str = PROMPT_VERSION,
    ):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.timeout = float(os.environ.get("LLM_TIMEOUT", timeout))
        self.max_retries = max_retries
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", max_tokens))
        self.per_source_k = int(os.environ.get("LLM_PER_SOURCE_K", per_source_k))
        self.max_concurrency = int(os.environ.get("LLM_MAX_CONCURRENCY", max_concurrency))
        self.self_consistency_n = int(os.environ.get("LLM_SC_N", self_consistency_n))
        self.self_consistency_temperature = self_consistency_temperature
        self.self_consistency_margin = float(os.environ.get("LLM_SC_MARGIN", self_consistency_margin))
        self.prompt_version = prompt_version
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self.stats = {"calls": 0, "cache_hits": 0, "errors": 0, "reasks": 0}

    # ------------------------------------------------------------------ cache
    def _cache(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            conn = sqlite3.connect(self.cache_path, check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, response TEXT, created_at REAL)"
            )
            conn.commit()
            self._conn = conn
        return self._conn

    @staticmethod
    def cache_key(entity: str, context: str, system: str, cand_codes: list[str],
                  prompt_version: str, model: str) -> str:
        ctx_hash = hashlib.sha256((context or "").encode()).hexdigest()[:16]
        cand_hash = hashlib.sha256("|".join(sorted(cand_codes)).encode()).hexdigest()[:16]
        raw = f"{entity}\x1f{ctx_hash}\x1f{system}\x1f{cand_hash}\x1f{prompt_version}\x1f{model}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        with self._lock:
            cur = self._cache().execute("SELECT response FROM llm_cache WHERE key=?", (key,))
            row = cur.fetchone()
        if row:
            self.stats["cache_hits"] += 1
            return json.loads(row[0])
        return None

    def _cache_put(self, key: str, value: dict) -> None:
        with self._lock:
            self._cache().execute(
                "INSERT OR REPLACE INTO llm_cache (key, response, created_at) VALUES (?,?,?)",
                (key, json.dumps(value), time.time()),
            )
            self._cache().commit()

    # ------------------------------------------------------------------ chat
    def _chat(self, system: str, entity: str, context: str, cand_block: str,
              temperature: float) -> str | None:
        url = f"{self.base_url}/chat/completions"
        user = (
            f"TASK: Choose exactly ONE {'ICD-10-CM code' if system == 'icd' else 'SNOMED CT concept id'} "
            f"for the term.\n"
            f'TERM: "{entity}"\n'
            f'CONTEXT: "{context[:600]}"\n'
            f"CANDIDATES:\n{cand_block}\n"
            f"RULES: {RULES_EXCERPT[system]}\n"
            'OUTPUT: JSON {"code": "<one of the candidate codes>", "reason": "<=20 words"}\n'
            "Never invent a code that is not in CANDIDATES."
        )
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPTS[system]},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                self.stats["calls"] += 1
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if r.status_code >= 500 or r.status_code == 429:
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception:
                self.stats["errors"] += 1
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(delay)
                delay *= 2
        return None

    @staticmethod
    def _parse_code(text: str | None) -> tuple[str, str]:
        if not text:
            return "", ""
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            t = t[t.find("{"):]
        try:
            obj = json.loads(t)
            return str(obj.get("code", "")).strip(), str(obj.get("reason", ""))[:120]
        except Exception:
            import re

            m = re.search(r"\{.*\}", t, re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    return str(obj.get("code", "")).strip(), str(obj.get("reason", ""))[:120]
                except Exception:
                    pass
        return "", ""

    # ------------------------------------------------------------------ public
    def choose(
        self,
        system: str,
        entity: str,
        context: str,
        cs: CandidateSet,
        disp_map: dict | None = None,
        use_self_consistency: bool = False,
    ) -> LLMResult | None:
        if not self.base_url or not self.model:
            return None
        cand_block, valid_codes = _candidate_block(cs, top=20, disp_map=disp_map,
                                                   per_source_k=self.per_source_k)
        if not valid_codes:
            return None
        key = self.cache_key(entity, context, system, sorted(valid_codes),
                             self.prompt_version, self.model)
        cached = self._cache_get(key)
        if cached is not None:
            if cached.get("code"):
                return LLMResult(code=cached["code"], reason=cached.get("reason", ""),
                                 n_votes=cached.get("n_votes", 1), cached=True)

        n = self.self_consistency_n if use_self_consistency else 1
        temp = self.self_consistency_temperature if use_self_consistency else 0.0
        votes: dict[str, int] = {}
        reasons: dict[str, str] = {}
        samples = 0
        for _ in range(n):
            text = self._chat(system, entity, context, cand_block, temp)
            code, reason = self._parse_code(text)
            if code and code not in valid_codes:
                # re-ask once (plan: reject and re-ask once)
                self.stats["reasks"] += 1
                text = self._chat(system, entity, context, cand_block, temp)
                code, reason = self._parse_code(text)
            if code in valid_codes:
                votes[code] = votes.get(code, 0) + 1
                reasons.setdefault(code, reason)
                samples += 1
        if not votes:
            return None
        best = max(votes.items(), key=lambda kv: (kv[1], kv[0]))
        result = {"code": best[0], "reason": reasons.get(best[0], ""), "n_votes": best[1],
                  "samples": samples, "votes": votes}
        self._cache_put(key, result)
        return LLMResult(code=best[0], reason=result["reason"], n_votes=best[1], raw=result)

    def merge_into_candidates(self, cs: CandidateSet, result: LLMResult,
                              top_score: float = 1.0) -> None:
        from ..candidates.base import Candidate

        cs.add(Candidate(code=result.code, source="llm" if result.n_votes <= 1 else "llm_sc",
                         score=min(1.0, top_score + 0.1 + 0.02 * result.n_votes), rank=0,
                         display=f"llm:{result.reason}" if result.reason else "llm"))
