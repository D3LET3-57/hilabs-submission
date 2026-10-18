"""Fine-tuned encoder judge: scores candidate (code | display) pairs against a row.

Used as an extra candidate source so fusion/rules/policy treat its preference
like any other signal. The option list is source-diverse (same builder as the
LLM prompt) so recall per option is much higher than fused-score top-k.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .llm import _select_candidates

if TYPE_CHECKING:
    from ..candidates.base import CandidateSet


class EncoderJudge:
    def __init__(self, model_path: str, device: str | None = None,
                 per_source_k: int = 20, top: int = 120, batch: int = 128,
                 max_len: int = 256):
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)

        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=1, trust_remote_code=True, dtype=torch.bfloat16)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.per_source_k = per_source_k
        self.top = top
        self.batch = batch
        self.max_len = max_len

    @torch.no_grad()
    def scores(self, state: str, cs: "CandidateSet", disp_map: dict | None = None
               ) -> dict[str, float]:
        cands = _select_candidates(cs, self.top, self.per_source_k)
        if not cands:
            return {}
        texts = [state] * len(cands)
        pairs = [f"{c.code} | {((disp_map or {}).get(c.code) or c.display or c.code)}"
                 for c in cands]
        out: list[float] = []
        for s in range(0, len(texts), self.batch):
            b = self.tok(texts[s:s + self.batch], pairs[s:s + self.batch],
                         truncation=True, max_length=self.max_len,
                         padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**b).logits.squeeze(-1).float()
            out.extend(torch.sigmoid(logits).cpu().tolist())
        return {c.code: float(p) for c, p in zip(cands, out)}
