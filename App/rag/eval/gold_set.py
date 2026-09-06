"""Phase 9: hand-built gold set for the NetApp MTS Intern JD (n=7).

The labels live in a VERSIONED JSON data file (not in code) next to this
module; this module is a thin loader with typed accessors so metrics code and
tests never manipulate raw dicts by hand.

SAMPLE SIZE CONTRACT
--------------------
There are 7 candidates and 161 (requirement x candidate) labels. Per-pair
metrics over 161 labels are reportable. Any RANKING-level result (Kendall tau
vs the hand ranking) is illustration, not a statistically meaningful benchmark;
""KENDALL_CAVEAT"" is the canonical sentence for that and MUST be appended to
every ranking-level report.
"""

from __future__ import annotations

import json
import os

DATA_FILENAME = "gold_set_netapp_mts_intern_v1.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_HERE, "data", DATA_FILENAME)

KENDALL_CAVEAT = (
    "(n=7; illustrative only, not a statistically meaningful ranking benchmark)"
)

REQUIRED_TIERS_VERDICT_VALUES = {"demonstrated", "claimed_only", "absent"}


class GoldSet:
    def __init__(self, path: str = _DATA_PATH, chunks_index: dict | None = None):
        with open(path, encoding="utf-8") as f:
            self._data = json.load(f)
        # chunks_index maps candidate slug -> {chunk_id: raw_text}; used only to
        # resolve gold chunk ids at eval time. If not provided, spanned labels
        # carry chunk ids but no pre-resolved text.
        self._chunks_index = chunks_index if chunks_index is not None else {}

    # -- accessors ---------------------------------------------------------

    @property
    def meta(self) -> dict:
        return self._data["meta"]

    @property
    def requirements(self) -> list[dict]:
        return self._data["requirements"]

    @property
    def candidates(self) -> list[dict]:
        return self._data["candidates"]

    @property
    def labels(self) -> list[dict]:
        return self._data["labels"]

    @property
    def hand_ranking(self) -> list[str]:
        return self._data["hand_ranking"]

    @property
    def system_skill_alias(self) -> dict[str, list[str]]:
        """Gold requirement id -> system extracted-skill names that can stand in
        for it (derived from the real rule_based_fallback_extraction output)."""
        return self._data["meta"]["system_skill_alias"]

    @property
    def system_skills(self) -> list[str]:
        return self._data["meta"]["system_skills_measured"]["skills"]

    def slug_for_filename(self, filename: str) -> str | None:
        for c in self.candidates:
            if c["filename"] == filename:
                return c["slug"]
        return None

    def requirement_text(self, rid: str) -> str:
        for r in self.requirements:
            if r["id"] == rid:
                return r["text"]
        raise KeyError(rid)

    def label(self, rid: str, slug: str) -> dict:
        for L in self.labels:
            if L["r"] == rid and L["c"] == slug:
                return L
        raise KeyError((rid, slug))

    def gold_verdict(self, rid: str, slug: str) -> str:
        return self.label(rid, slug)["v"]

    def gold_pairs(self) -> list[dict]:
        """(requirement, candidate, gold_chunk_id, gold_quote) for every
        demonstrated label — the retrieval recall@3 ground truth."""
        pairs = []
        for L in self.labels:
            if L["v"] != "demonstrated":
                continue
            text = self._chunks_index.get(L["c"], {}).get(L["chunk"], "")
            pairs.append(
                {
                    "requirement_id": L["r"],
                    "requirement": self.requirement_text(L["r"]),
                    "candidate_id": self.candidate_id_for_slug(L["c"]),
                    "candidate_slug": L["c"],
                    "gold_chunk_id": L["chunk"],
                    "gold_quote": L.get("quote", ""),
                    "gold_chunk_text": text,
                }
            )
        return pairs

    def candidate_id_for_slug(self, slug: str) -> str:
        for c in self.candidates:
            if c["slug"] == slug:
                return c["candidate_id"]
        raise KeyError(slug)

    def demonstrated_requirement_ids(self, slug: str) -> set[str]:
        return {L["r"] for L in self.labels if L["c"] == slug and L["v"] == "demonstrated"}

    # -- integrity ---------------------------------------------------------

    def check_integrity(self) -> bool:
        req_ids = {r["id"] for r in self.requirements}
        slugs = {c["slug"] for c in self.candidates}
        assert len(self.requirements) == 23, "expected 23 requirements"
        assert len(self.candidates) == 7, "expected 7 candidates"
        seen = set()
        for L in self.labels:
            assert L["r"] in req_ids and L["c"] in slugs, (L["r"], L["c"])
            assert L["v"] in REQUIRED_TIERS_VERDICT_VALUES, L["v"]
            key = (L["r"], L["c"])
            assert key not in seen, "duplicate label %s" % (key,)
            seen.add(key)
        assert len(seen) == 23 * 7, "expected exactly 161 labels, got %d" % len(seen)
        assert sorted(self.hand_ranking) == sorted(slugs), "hand ranking invalid"
        return True


def load_gold_set(chunks_index: dict | None = None) -> GoldSet:
    gs = GoldSet(chunks_index=chunks_index)
    gs.check_integrity()
    return gs