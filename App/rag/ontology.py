"""Phase 5: skill normalization against the ESCO ontology.

``SkillNormalizer`` canonicalizes extracted JD skills against the ESCO concept
collection (Phase 3's ``OntologyIndexer``) so paraphrases collapse to the same
concept (Finding 3), and classifies each skill required-vs-preferred with an LLM
call structured like Phase 0's ``classify_requirement_type``.

Important contract: ``NormalizedRequirement.skill_name`` is ALWAYS the ORIGINAL
extracted skill string, never the ESCO canonical label. The canonical label is
used only internally (to inform the LLM's tier judgment so paraphrases are rated
consistently) and for ``alt_labels_for()``. This preserves a 1:1 correspondence
with ``extracted_skills`` for Phase 6's resume-side matching.

FALLBACK BEHAVIOR (per spec): on ANY failure (ontology lookup, LLM call, or
malformed response) we fall back to Phase 0's exact ``classify_requirement_type``
behavior and set ``source="keyword_fallback"``. We import and reuse the real
function from ``pipeline`` rather than reimplementing it. The import is lazy
(function-local) so importing this module does NOT trigger ``pipeline``'s module-
level side effects (spaCy/model/OpenAI-client loading); only the fallback path
pays that cost.
"""

from __future__ import annotations

import time

from rag.config import RagSettings
from rag.embedding import EmbeddingService
from rag.schemas import NormalizedRequirement
from rag.store import VectorStore

# Minimum cosine similarity to accept an ontology_store (ESCO) match.
#
# Justification against Phase 3 stress-test numbers (real, measured):
#   * genuine paraphrases resolved with cosine 0.62-0.82
#   * the deliberately obscure out-of-vocabulary term scored only 0.068
# A floor of 0.3 sits well above the OOV garbage (0.068, -> rejected as noise)
# and comfortably below the weakest genuine paraphrase (0.62), so real matches
# are retained with a ~2x margin while noise is cleanly excluded. It is NOT a
# round number plucked arbitrarily — it is chosen from those two measured
# populations.
ONTOLOGY_MATCH_FLOOR = 0.3

# LLM classifier config, mirroring Phase 0's classify_requirement_type.
_CLASSIFY_MODEL = "openai/gpt-4o-mini"
_CLASSIFY_RETRIES = 2
# Phase 10: content-cache version stamp. Bump when the prompt changes so a new
# prompt never reuses an old cached classification.
_CLASSIFY_PROMPT_VERSION = "v1"


class SkillNormalizer:
    def __init__(self, ontology_store: VectorStore, embedder: EmbeddingService, llm_client, dispatcher=None):
        self.ontology_store = ontology_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.settings = RagSettings()
        # Phase 10: optionally-shared LLM dispatcher (cache + log + budget).
        # When None, a per-instance dispatcher is built lazily (default for
        # tests/standalone use); production wiring injects the shared one.
        self._dispatcher = dispatcher

    def _get_dispatcher(self):
        if self._dispatcher is None:
            from rag.llm_call import LLMDispatcher

            self._dispatcher = LLMDispatcher.build(self.llm_client, self.settings)
        return self._dispatcher

    # -- ontology lookup ----------------------------------------------------

    def _lookup_concept(self, skill: str):
        """Return (canonical_label, alt_labels) for a skill, or (skill, []) if the
        best ESCO match is below the cosine floor."""
        qvec = self.embedder.encode([skill])[0]
        hits = self.ontology_store.search_dense(
            self.settings.esco_collection_name,
            qvec,
            k=1,
            filters=None,
        )
        if not hits:
            return skill, []
        score = float(hits[0]["score"])
        if score < ONTOLOGY_MATCH_FLOOR:
            return skill, []
        payload = hits[0]["payload"]
        canonical = payload.get("preferred_label") or skill
        alt_labels = list(payload.get("alt_labels", []) or [])
        return canonical, alt_labels

    def alt_labels_for(self, skill: str) -> list[str]:
        """ESCO alt_labels for a skill; [] if no confident ontology match."""
        return self._lookup_concept(skill)[1]

    # -- LLM classification -------------------------------------------------

    def _classify_canonical(self, jd_text: str, canonical_skills: list[str]):
        """LLM classify of canonicalized skill names, structured like Phase 0's
        classify_requirement_type (same response_format + retry/backoff). Returns
        {"required": [...], "preferred": [...]} on success, else None."""
        skill_list = "\n".join("- %s" % s for s in canonical_skills)
        prompt = (
            "You are analyzing a job description. Classify each skill below as "
            "either REQUIRED (mandatory, must-have) or PREFERRED (optional, "
            "nice-to-have, plus) based on how the job description text describes "
            "each skill.\n\n"
            "Job description:\n%s\n\n"
            "Skills to classify:\n%s\n\n"
            "OUTPUT FORMAT (STRICT JSON ONLY):\n"
            '{"required": ["skill1", "skill2"], "preferred": ["skill3"]}'
        ) % (jd_text[:4000], skill_list)

        for attempt in range(_CLASSIFY_RETRIES):
            try:
                # Phase 10: dispatch through the shared content-addressed cache
                # so re-normalizing the same JD makes zero LLM calls.
                payload = self._get_dispatcher().call(
                    op="classify",
                    prompt=prompt,
                    model=_CLASSIFY_MODEL,
                    cache_version=_CLASSIFY_PROMPT_VERSION,
                )
                # None => budget exhausted or unparseable body: no point retrying
                # (a blocked budget stays blocked), so fall straight to fallback.
                if payload is None:
                    return None
                result = payload
                if (
                    isinstance(result, dict)
                    and isinstance(result.get("required"), list)
                    and isinstance(result.get("preferred"), list)
                ):
                    return result
                return None
            except Exception as e:
                if attempt == _CLASSIFY_RETRIES - 1:
                    print("%s: LLM skill classification failed (canonical); using fallback. Error: %s" % (type(self).__name__, e))
                    return None
                time.sleep(1 * (attempt + 1))
        return None

    # -- main entry ---------------------------------------------------------

    def normalize(
        self, jd_text: str, extracted_skills: list[str]
    ) -> list[NormalizedRequirement]:
        # 1. Canonicalize every extracted skill. Keeps 1:1 correspondence by index.
        canonical_by_index = []
        alt_by_index = []
        for skill in extracted_skills:
            canonical, alts = self._lookup_concept(skill)
            canonical_by_index.append(canonical)
            alt_by_index.append(alts)

        canonical_skills = list(canonical_by_index)

        # 2. Classify canonical names via LLM.
        classification = self._classify_canonical(jd_text, canonical_skills)

        if classification and (classification.get("required") or classification.get("preferred")):
            source = "llm_classifier"
        else:
            # 3. Fallback to Phase 0's exact classify_requirement_type behavior.
            classification, source = self._fallback_classify(jd_text, extracted_skills)

        required_set = {r.lower() for r in (classification.get("required") or [])}
        preferred_set = {p.lower() for p in (classification.get("preferred") or [])}

        requirements = []
        for i, skill in enumerate(extracted_skills):
            canonical = canonical_by_index[i]
            if canonical.lower() in required_set:
                tier = "required"
            elif canonical.lower() in preferred_set:
                tier = "preferred"
            else:
                # LLM didn't mention this canonical label; default conservatively
                # to preferred rather than dropping the requirement entirely.
                tier = "preferred"
            requirements.append(
                NormalizedRequirement(
                    skill_name=skill,
                    tier=tier,
                    source=source,
                )
            )
        return requirements

    def _fallback_classify(self, jd_text: str, extracted_skills: list[str]):
        """Reuse Phase 0's exact classify_requirement_type (lazy import), then mirror
        parse_jd_requirements' second-tier keyword heuristic if it yields nothing."""
        # Lazy import keeps pipeline.py's heavy module-level side effects out of
        # this module's import path; only the fallback path depends on it.
        from pipeline import classify_requirement_type, split_jd_sections

        classification = classify_requirement_type(jd_text, extracted_skills)
        if classification and (classification.get("required") or classification.get("preferred")):
            return classification, "keyword_fallback"

        # Second fallback: reuse the SAME real fallback path parse_jd_requirements
        # uses — split_jd_sections (which centralizes REQUIRED_CUES/PREFERRED_CUES)
        # — rather than a re-hardcoded independent copy of those cue lists.
        sections = split_jd_sections(jd_text)
        req_text_lower = sections["required_text"].lower()
        pref_text_lower = sections["preferred_text"].lower()
        required = [s for s in extracted_skills if s.lower() in req_text_lower]
        preferred = [
            s for s in extracted_skills
            if s.lower() in pref_text_lower and s.lower() not in {r.lower() for r in required}
        ]
        if not required:
            required = extracted_skills[:10]
        return {"required": required, "preferred": preferred}, "keyword_fallback"
