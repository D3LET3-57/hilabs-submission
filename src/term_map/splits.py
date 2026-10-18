"""Leakage-proof split construction: connected components of the entity<->chunk
bipartite graph are assigned whole to splits (doc 04 section 2)."""
from __future__ import annotations

import pandas as pd


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_splits(pairs: pd.DataFrame, train_frac: float = 0.65, dev_frac: float = 0.15,
                 seed: int = 17) -> pd.DataFrame:
    """pairs needs columns entity_norm and chunk_id. Deterministic greedy assignment."""
    uf = UnionFind()
    for entity, cid in zip(pairs["entity_norm"], pairs["chunk_id"]):
        uf.union(("e", entity), ("c", cid))
    comps: dict = {}
    for i, (entity, cid) in enumerate(zip(pairs["entity_norm"], pairs["chunk_id"])):
        comps.setdefault(uf.find(("e", entity)), []).append(i)

    n = len(pairs)
    targets = {"train": train_frac * n, "dev": dev_frac * n,
               "test": (1 - train_frac - dev_frac) * n}
    counts = {"train": 0, "dev": 0, "test": 0}
    order = ["train", "dev", "test"]
    assign = ["train"] * n
    comp_items = sorted(comps.items(), key=lambda kv: (-len(kv[1]), str(kv[0])))
    for _root, idxs in comp_items:
        deficits = {s: targets[s] - counts[s] for s in order}
        best = max(order, key=lambda s: (deficits[s], -order.index(s)))
        for i in idxs:
            assign[i] = best
        counts[best] += len(idxs)
    out = pairs.copy()
    out["split"] = assign
    return out
