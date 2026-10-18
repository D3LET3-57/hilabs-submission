"""SymSpell-based conservative typo correction over a terminology lexicon."""
from __future__ import annotations

import json
import re
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z][a-z\-']{1,}")


class SymSpellCorrector:
    def __init__(self, pairs: dict[str, str], vocab: dict[str, int],
                 max_edit_distance: int = 1, min_len: int = 5, min_freq: int = 5):
        self.pairs = pairs
        self.max_edit_distance = max_edit_distance
        self.min_len = min_len
        self.min_freq = min_freq
        self._spell = None
        self._vocab = vocab
        self._unigrams = None

    def _ensure(self):
        if self._spell is None:
            from symspellpy import SymSpell, Verbosity

            sp = SymSpell(max_dictionary_edit_distance=self.max_edit_distance, prefix_length=7)
            for tok, freq in self._vocab.items():
                if len(tok) >= 2:
                    sp.create_dictionary_entry(tok, max(1, int(freq)))
            self._spell = sp
            self._verbosity = Verbosity
        return self._spell

    def __call__(self, text: str) -> str:
        if not text:
            return text
        toks = text.split()
        out = []
        changed = False
        for tok in toks:
            rep = self.pairs.get(tok)
            if rep is not None:
                out.append(rep)
                changed = changed or rep != tok
                continue
            plain = tok.strip("-'")
            if len(plain) >= self.min_len and plain.isalpha() and plain not in self._vocab:
                sp = self._ensure()
                sug = sp.lookup(plain, self._verbosity.TOP, max_edit_distance=self.max_edit_distance)
                if sug and sug[0].distance > 0 and self._vocab.get(sug[0].term, 0) >= self.min_freq:
                    out.append(sug[0].term)
                    changed = True
                    continue
            out.append(tok)
        return " ".join(out) if changed else text

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"pairs": self.pairs, "vocab": self._vocab}))

    @staticmethod
    def from_json(path: str | Path) -> "SymSpellCorrector":
        data = json.loads(Path(path).read_text())
        return SymSpellCorrector(pairs=data.get("pairs", {}), vocab=data.get("vocab", {}))
