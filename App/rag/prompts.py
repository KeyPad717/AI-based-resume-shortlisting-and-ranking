"""Phase 5: prompt construction.

Plain, snapshot-testable functions. Every prompt is versioned per the plan's
emphasis on prompt versioning. ``build_evidence_prompt`` is deterministic: given
the same inputs it must produce the exact same string every time (no timestamps,
no randomness), so it can be snapshot-tested.
"""

from __future__ import annotations

EVIDENCE_VERDICT_PROMPT_VERSION = "v1"

# LLM model for evidence verdict generation (mirrors the pipeline's model).
EVIDENCE_VERDICT_MODEL = "openai/gpt-4o-mini"


def build_evidence_prompt(candidate_id: str, requirements_with_spans) -> str:
    """Build the single batched evidence-verdict prompt for a candidate.

    ``requirements_with_spans`` is a list of ``(requirement, spans)`` tuples,
    where ``requirement`` is a NormalizedRequirement and ``spans`` is a list of
    EvidenceSpan (already restricted to the candidate). Only requirements that
    actually retrieved evidence are included (zero-span requirements are handled
    as "absent" by the caller without an LLM call).

    The LLM is asked to propose a verdict + verbatim quote citation per skill;
    Phase 5's CitationValidator then *verifies* those citations in code (LLM
    proposes, code verifies) — this prompt never asks the LLM to self-verify.
    """
    lines = []
    for i, (requirement, spans) in enumerate(requirements_with_spans, start=1):
        lines.append("%d. SKILL: %s" % (i, requirement.skill_name))
        if not spans:
            lines.append("   (no evidence spans retrieved)")
            continue
        for j, span in enumerate(spans, start=1):
            lines.append(
                "   SPAN %d (chunk_id=%s): %s"
                % (j, span.chunk_id, _squash(span.text))
            )

    body = "\n".join(lines) if lines else "(no skills provided)"

    return (
        "You are an evidence verifier for resume shortlisting.\n"
        "For EACH skill listed below, decide the verdict based ONLY on the "
        "provided evidence spans (which come from that candidate's resume), and "
        "cite the exact chunk_id(s) and a VERBATIM quote from the supporting "
        "chunk's text that demonstrates the skill.\n\n"
        "PROMPT VERSION: %s\n\n" % EVIDENCE_VERDICT_PROMPT_VERSION +

        "Rules:\n"
        '- verdict must be one of: "demonstrated", "claimed_only", "absent"\n'
        '  - "demonstrated": the chunk text actually evidences the skill\n'
        '  - "claimed_only": the skill appears as a claim/keyword only\n'
        '  - "absent": no span supports it\n'
        "- quote must be an EXACT substring of one of the cited chunks' text.\n"
        "- If a skill has no supporting evidence, set verdict=\"absent\" and "
        "cite_chunk_ids=[] and quote=null.\n\n"
        "Candidate ID: %s\n"
        "Skills and their evidence spans:\n%s\n\n"
        'OUTPUT FORMAT (STRICT JSON ONLY): a JSON array of objects like\n'
        '[{"skill_name": "...", "verdict": "claimed_only", '
        '"cited_chunk_ids": ["chunk1"], "quote": "exact substring"}]'
    ) % (candidate_id, body)


def _squash(text: str, limit: int = 600) -> str:
    """Truncate a span's text for prompt inclusion; position-dependent but
    deterministic (same input -> same output every time)."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."
