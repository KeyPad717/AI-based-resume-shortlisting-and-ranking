"""Phase 5: citation-validated evidence verification.

Citation validation is PURE CODE — the LLM proposes verdicts and citations, then
``CitationValidator`` verifies them deterministically (the "LLM proposes, code
verifies" boundary the plan insists on). We do NOT ask another LLM whether a
citation is valid.

Deterministic CE-score fallback thresholds (used only when the whole candidate's
batched LLM verdict call fails): a span with cross_encoder_score > 0.6 is
"demonstrated", > 0.3 is "claimed_only", else "absent".
"""

from __future__ import annotations

import json
import re
import time

from rag.prompts import EVIDENCE_VERDICT_MODEL, build_evidence_prompt
from rag.retrieval import CE_RELEVANCE_FLOOR, SkillEvidenceRetriever
from rag.schemas import CandidateEvidence, EvidenceSpan, NormalizedRequirement, SkillVerdict

# Deterministic cross-encoder fallback threshold (see module docstring).
CE_FALLBACK_DEMONSTRATED = 0.6
CE_FALLBACK_CLAIMED = 0.3

_EVIDENCE_RETRIES = 2


def _norm(text: str) -> str:
    """Normalize whitespace and case for verbatim-substring compare only.
    Does NOT accept paraphrases — this just collapses spacing/case."""
    return " ".join((text or "").split()).lower()


class CitationValidator:
    """Rejects citations that a code check can prove invalid.

    A verdict is downgraded (verdict -> \"claimed_only\", quote -> None) — never
    discarded — if any of:
      * a cited chunk_id is not present in retrieved_spans;
      * a cited chunk's candidate_id does not match the given candidate_id;
      * a quote is provided but is not a verbatim substring of the cited chunk's
        text (normalized for spacing/case, but paraphrases are rejected).

    Returns (validated_verdicts, rejected_count). The count feeds the Phase 9
    citation-validity metric.
    """

    def validate(self, verdicts, retrieved_spans, candidate_id):
        span_by_id = {s.chunk_id: s for s in retrieved_spans}
        rejected = 0
        validated = []
        for v in verdicts:
            sv = SkillVerdict(
                skill_name=v.get("skill_name", ""),
                verdict=v.get("verdict", "absent"),
                cited_chunk_ids=list(v.get("cited_chunk_ids", []) or []),
                quote=v.get("quote"),
            )
            if self._is_invalid(sv, span_by_id, candidate_id):
                rejected += 1
                sv.verdict = "claimed_only"
                sv.quote = None
            validated.append(sv)
        return validated, rejected

    def _is_invalid(self, sv: SkillVerdict, span_by_id, candidate_id) -> bool:
        # 1. Any cited chunk id not in retrieved_spans.
        for cid in sv.cited_chunk_ids:
            if cid not in span_by_id:
                return True
        # 2. A cited chunk belongs to a different candidate.
        for cid in sv.cited_chunk_ids:
            span = span_by_id.get(cid)
            if span is not None and span.candidate_id != candidate_id:
                return True
        # 3. A quote is provided but is not a verbatim substring of a cited chunk.
        if sv.quote:
            quote_norm = _norm(sv.quote)
            for cid in sv.cited_chunk_ids:
                span = span_by_id.get(cid)
                if span is not None and quote_norm in _norm(span.text):
                    return False
            return True
        return False


class EvidenceVerifier:
    """Per-candidate evidence verification.

    Verifies ALL requirements for a candidate in ONE batched LLM call (cost +
    consistency). Requirements with zero retrieved evidence are marked
    \"absent\" locally with NO LLM call. On total LLM failure, falls back to a
    deterministic verdict derived from cross_encoder_score instead of failing
    the whole candidate.
    """

    def __init__(
        self,
        retriever: SkillEvidenceRetriever,
        llm_client,
        validator: CitationValidator,
        alt_labels_provider=None,
    ):
        self._retriever = retriever
        self._llm_client = llm_client
        self._validator = validator
        # Optional callable skill -> list[str] (e.g. a bound SkillNormalizer.
        #alt_labels_for). Additive beyond the spec'd three args so the wiring
        # phase can supply real ESCO alt labels without changing the core API.
        self._alt_labels_provider = alt_labels_provider

    def _alt_labels_for(self, skill: str) -> list[str]:
        if self._alt_labels_provider is None:
            return []
        try:
            return self._alt_labels_provider(skill) or []
        except Exception:
            return []

    def _retrieve_for(self, candidate_id: str, requirement: NormalizedRequirement):
        alt = self._alt_labels_for(requirement.skill_name)
        return self._retriever.for_requirement(requirement, candidate_id, alt_labels=alt)

    def verify(
        self, candidate_id: str, requirements: list[NormalizedRequirement]
    ) -> CandidateEvidence:
        per_requirement = []  # (requirement, spans, absent)
        ranged_spans = []  # every retrieved span for the candidate
        for req in requirements:
            spans = self._retrieve_for(candidate_id, req)
            ranged_spans.extend(spans)
            if not spans:
                per_requirement.append((req, [], True))
            else:
                per_requirement.append((req, spans, False))

        # Zero-span requirements: marked absent directly, no LLM call.
        absent = [
            SkillVerdict(
                skill_name=req.skill_name,
                verdict="absent",
                cited_chunk_ids=[],
                quote=None,
            )
            for req, _spans, is_absent in per_requirement
            if is_absent
        ]

        with_spans = [(req, spans) for req, spans, is_absent in per_requirement if not is_absent]

        if not with_spans:
            return CandidateEvidence(
                candidate_id=candidate_id,
                verdicts=absent,
                rejected_citations=0,
            )

        # One batched LLM call for all skills-with-spans.
        parsed = self._call_llm_batch(candidate_id, with_spans)

        if parsed is None:
            # Total LLM failure -> deterministic cross-encoder fallback.
            verdicts = absent + self._deterministic_verdicts(with_spans)
            return CandidateEvidence(
                candidate_id=candidate_id,
                verdicts=verdicts,
                rejected_citations=0,
            )

        # Code-verify the LLM's proposed verdicts & citations.
        validated, rejected = self._validator.validate(
            parsed, ranged_spans, candidate_id
        )
        # Attach the absent (zero-span) verdicts to the validated ones.
        validated += absent
        return CandidateEvidence(
            candidate_id=candidate_id,
            verdicts=validated,
            rejected_citations=rejected,
        )

    def _call_llm_batch(self, candidate_id, with_spans):
        prompt = build_evidence_prompt(candidate_id, with_spans)
        for attempt in range(_EVIDENCE_RETRIES):
            try:
                response = self._llm_client.chat.completions.create(
                    model=EVIDENCE_VERDICT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
                data = json.loads(raw)
                parsed = data if isinstance(data, list) else None
                if isinstance(data, dict):
                    for value in data.values():
                        if isinstance(value, list) and value:
                            parsed = value
                            break
                if isinstance(parsed, list) and parsed:
                    return parsed
                return None
            except Exception as e:
                if attempt == _EVIDENCE_RETRIES - 1:
                    print(
                        "%s: LLM evidence verdict call failed for candidate %s; "
                        "using deterministic CE-score fallback. Error: %s"
                        % (type(self).__name__, candidate_id, e)
                    )
                    return None
                time.sleep(1 * (attempt + 1))
        return None

    def _deterministic_verdicts(self, with_spans):
        """Derive a verdict for each skill-with-spans from cross_encoder_score."""
        verdicts = []
        for req, spans in with_spans:
            best = max((s.cross_encoder_score or 0.0) for s in spans)
            if best > CE_FALLBACK_DEMONSTRATED:
                verdict = "demonstrated"
            elif best > CE_FALLBACK_CLAIMED:
                verdict = "claimed_only"
            else:
                verdict = "absent"
            verdicts.append(
                SkillVerdict(
                    skill_name=req.skill_name,
                    verdict=verdict,
                    cited_chunk_ids=[],
                    quote=None,
                )
            )
        return verdicts
