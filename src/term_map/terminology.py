"""Terminology loading: SNOMED CT, ICD-10-CM, official map + lookup structures."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .normalize import basic_norm, split_abbr_description, strip_trailing_punct


def _build_norm_index(descriptions: pd.Series) -> dict[str, list[str]]:
    """normalized description -> list of codes (deduped, order-preserving)."""
    idx: dict[str, list[str]] = {}
    seen_pair: set[tuple[str, str]] = set()
    for code, desc in zip(descriptions.index, descriptions.values):
        norm = strip_trailing_punct(basic_norm(desc))
        if not norm:
            continue
        key = (norm, code)
        if key in seen_pair:
            continue
        seen_pair.add(key)
        idx.setdefault(norm, []).append(code)
    return idx


@dataclass
class Terminology:
    snomed: pd.DataFrame                     # code, description, description_order
    icd: pd.DataFrame                        # code, description
    map_df: pd.DataFrame | None = None       # snomed_code, icd_code, map_priority_rule, advice, active
    # built lazily
    _snomed_norm_index: dict[str, list[str]] | None = field(default=None, repr=False)
    _icd_norm_index: dict[str, list[str]] | None = field(default=None, repr=False)
    _snomed_abbrev_index: dict[str, list[str]] | None = field(default=None, repr=False)
    _icd_abbrev_index: dict[str, list[str]] | None = field(default=None, repr=False)
    _snomed_display: dict[str, str] | None = field(default=None, repr=False)
    _icd_display: dict[str, str] | None = field(default=None, repr=False)
    _map_by_snomed: dict[str, list[str]] | None = field(default=None, repr=False)
    _map_by_icd: dict[str, list[str]] | None = field(default=None, repr=False)
    _snomed_codes: set | None = field(default=None, repr=False)
    _icd_codes: set | None = field(default=None, repr=False)

    # ---------------------------------------------------------- properties
    @property
    def snomed_norm_index(self) -> dict[str, list[str]]:
        if self._snomed_norm_index is None:
            s = self.snomed
            self._snomed_norm_index = _build_norm_index(
                s.set_index("code")["description"]
            )
        return self._snomed_norm_index

    @property
    def icd_norm_index(self) -> dict[str, list[str]]:
        if self._icd_norm_index is None:
            self._icd_norm_index = _build_norm_index(
                self.icd.set_index("code")["description"]
            )
        return self._icd_norm_index

    @property
    def snomed_abbrev_index(self) -> dict[str, list[str]]:
        if self._snomed_abbrev_index is None:
            idx: dict[str, list[str]] = {}
            for code, desc in zip(self.snomed["code"], self.snomed["description"]):
                abbr, _ = split_abbr_description(desc)
                if abbr:
                    idx.setdefault(abbr, []).append(code)
            self._snomed_abbrev_index = idx
        return self._snomed_abbrev_index

    @property
    def icd_abbrev_index(self) -> dict[str, list[str]]:
        if self._icd_abbrev_index is None:
            idx: dict[str, list[str]] = {}
            for code, desc in zip(self.icd["code"], self.icd["description"]):
                abbr, _ = split_abbr_description(desc)
                if abbr:
                    idx.setdefault(abbr, []).append(code)
            self._icd_abbrev_index = idx
        return self._icd_abbrev_index

    @property
    def snomed_display(self) -> dict[str, str]:
        if self._snomed_display is None:
            d = self.snomed
            pref = d[d["description_order"] == 1][["code", "description"]]
            first = d[["code", "description"]].drop_duplicates("code")
            disp = dict(zip(first["code"], first["description"]))
            disp.update(dict(zip(pref["code"], pref["description"])))
            self._snomed_display = disp
        return self._snomed_display

    @property
    def icd_display(self) -> dict[str, str]:
        if self._icd_display is None:
            d = self.icd.drop_duplicates("code")
            self._icd_display = dict(zip(d["code"], d["description"]))
        return self._icd_display

    @property
    def snomed_codes(self) -> set:
        if self._snomed_codes is None:
            self._snomed_codes = set(self.snomed["code"].unique())
        return self._snomed_codes

    @property
    def icd_codes(self) -> set:
        if self._icd_codes is None:
            self._icd_codes = set(self.icd["code"].unique())
        return self._icd_codes

    @property
    def has_map(self) -> bool:
        return self.map_df is not None and len(self.map_df) > 0

    @property
    def map_by_snomed(self) -> dict[str, list[str]]:
        if self._map_by_snomed is None:
            idx: dict[str, list[str]] = {}
            if self.has_map:
                df = self.map_df
                for src, tgt in zip(df["snomed_code"], df["icd_code"]):
                    idx.setdefault(src, [])
                    if tgt not in idx[src]:
                        idx[src].append(tgt)
            self._map_by_snomed = idx
        return self._map_by_snomed

    @property
    def map_by_icd(self) -> dict[str, list[str]]:
        if self._map_by_icd is None:
            idx: dict[str, list[str]] = {}
            if self.has_map:
                df = self.map_df
                for src, tgt in zip(df["snomed_code"], df["icd_code"]):
                    idx.setdefault(tgt, [])
                    if src not in idx[tgt]:
                        idx[tgt].append(src)
            self._map_by_icd = idx
        return self._map_by_icd

    def family(self, system: str, category: str, rows: int = 50) -> list[str]:
        """Return codes of a family whose display matches category (case-insens. substring)."""
        disp = self.icd_display if system == "icd" else self.snomed_display
        cat = category.lower()
        out = [c for c, d in disp.items() if cat in d.lower()]
        return out[:rows]


def load_terminology(snomed_path, icd_path, map_path=None) -> Terminology:
    snomed = pd.read_parquet(snomed_path)
    if "snomed_code" in snomed.columns:
        snomed = snomed.rename(columns={"snomed_code": "code"})
    icd = pd.read_parquet(icd_path)
    if "icd_code" in icd.columns:
        icd = icd.rename(columns={"icd_code": "code"})
    map_df = None
    try:
        map_df = pd.read_parquet(map_path)
    except Exception:
        map_df = None
    if map_df is not None and len(map_df) > 0:
        # keep only map targets that are real ICD-10-CM codes: the source map
        # contains WHO sentinel values ("\N") and non-billable placeholders
        # like "T07.XXX?" which must never be emitted or reranked.
        from .normalize import fmt_icd

        valid = set(icd["code"].astype(str))
        icd_code = map_df["icd_code"].astype(str).map(fmt_icd)
        snomed_code = map_df["snomed_code"].astype(str)
        keep = icd_code.isin(valid) & ~snomed_code.str.contains(r"[?\\]|XXX", regex=True)
        map_df = map_df[keep].assign(icd_code=icd_code[keep])
    return Terminology(snomed=snomed, icd=icd, map_df=map_df)
