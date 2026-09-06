"""Phase 9: full real evaluation (CLI).

Run from App/ (repo root parent is the project dir):

    cd App
    python3 -m rag.eval.run_eval

Two modes, chosen by environment state:

  * LLM-less baseline (Phase 9 historical record, writes results/run_eval_v1.json):
    no OPENROUTER_API_KEY, or RAG_EVAL_LLM=0. Every LLM call fails instantly and
    the deterministic CE-score fallback drives all verdicts.

  * LLM-backed rerun (writes results/run_eval_v2.json): OPENROUTER_API_KEY set.
    Real gpt-4o-mini classification (SkillNormalizer / classify_requirement_type)
    and real batched evidence verdicts + citations flow through.

Both modes run the SAME retrieval stack, gold-aligned 11-skill grid, six-row
ablation and metrics; only the LLM wiring and the reported LLC/verdict numbers
differ. The aligned verdict grid always uses rule_based_fallback_extraction for
its skill list so the gold alias mapping stays valid; the real LLM extraction
result is recorded as a diagnostic only.

Imports of pipeline and the heavy model classes happen at call time so this
module stays importable from tests.
"""

from __future__ import annotations

import json
import os
import sys
import time

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))  # App/rag/eval
APP = os.path.join(HERE, "..", "..")
sys.path.insert(0, APP)

load_dotenv(os.path.join(APP, ".env"))

from rag.config import RagSettings
from rag.embedding import EmbeddingService
from rag.eval.ablate import RetrievalAblation
from rag.eval.gold_set import load_gold_set
from rag.eval.metrics import (
    aggregate_alias_verdicts,
    citation_validity_rate,
    false_missing_skill_rate,
    kendall_tau,
    verdict_accuracy,
)
from rag.evidence import CitationValidator, EvidenceVerifier
from rag.index import OntologyIndexer
from rag.ontology import SkillNormalizer
from rag.retrieval import (
    CrossEncoderReranker,
    HybridRetriever,
    RRFFuser,
    SkillEvidenceRetriever,
)
from rag.schemas import NormalizedRequirement
from rag.store import InMemoryStore

REPO = os.path.join(APP, "..")  # repo root (houses JD/ and Resumes/)
JD_PDF = os.path.join(REPO, "JD", "JD.pdf")
RESUMES_DIR = os.path.join(REPO, "Resumes")
RESULTS_V1_FILE = os.path.join(HERE, "results", "run_eval_v1.json")
RESULTS_V2_FILE = os.path.join(HERE, "results", "run_eval_v2.json")

SYNTHETIC_ESCO_WARNING = (
    "No real ESCO dataset (ESCO_SKILLS_CSV unset) - using the SYNTHETIC 15-concept "
    "stand-in from scripts/build_esco_index.py. Ontology-expansion results are "
    "illustrative only and cannot generalize to the real taxonomy."
)

# requirement_text -> skill term for the ablation alt-labels provider.
CONFIG_ALT_LABELS = {
    "C programming": "c",
    "C++ programming": "c++",
    "Go": "go",
    "Docker": "docker",
    "Kubernetes": "kubernetes",
    "Linux": "linux",
    "Machine Learning": "machine learning",
    "Python": "python",
    "Java": "java",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
}

# JD-derived classification reference (direct reading of JD bullets): R01-R05
# "Good programming skills" -> required; R08 "Exposure to NodeJS/JS/TS" ->
# required; R11 "Familiarity with Linux" -> required; R15/R16 "preferred, not
# required"; R22 "is a plus". Used only to score classifier sanity, never as an
# input to the pipeline.
JD_REFERENCE_TIERS = {
    "c": "required",
    "c++": "required",
    "python": "required",
    "java": "required",
    "go": "required",
    "javascript": "required",
    "typescript": "required",
    "linux": "required",
    "docker": "preferred",
    "kubernetes": "preferred",
    "machine learning": "preferred",
}


class FailingLLMClient:
    """Never-reachable LLM: every call raises instantly, reproducing the LLM-less
    environment without hammering the network."""

    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("deterministic eval failure (no LLM available)")

    def __init__(self):
        self.chat = type("_chat", (), {"completions": self._Completions()})


class UsageTrackingLLMClient:
    """Wraps the real OpenRouter client, counting successful completions, tokens
    and wall latency while exposing the same chat.completions.create surface."""

    class _Completions:
        def __init__(self, owner):
            self._owner = owner

        def create(self, **kwargs):
            t0 = time.time()
            try:
                response = self._owner._inner.chat.completions.create(**kwargs)
            except Exception:
                self._owner.errors += 1
                self._owner.llm_wall_s += time.time() - t0
                raise
            self._owner.calls += 1
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._owner.prompt_tokens += int(usage.prompt_tokens or 0)
                self._owner.completion_tokens += int(usage.completion_tokens or 0)
            self._owner.llm_wall_s += time.time() - t0
            return response

    class _Chat:
        def __init__(self, owner):
            self.completions = UsageTrackingLLMClient._Completions(owner)

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_wall_s = 0.0
        self.errors = 0
        self.chat = self._Chat(self)


class RealLLMEvidenceVerifier(EvidenceVerifier):
    """EvidenceVerifier that additionally counts the LLM-proposed citations and
    per-candidate LLM failures so the citation-validity and fallback-usage
    numbers can be computed without touching the core class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.citation_proposed = 0
        self.llm_fallback_candidates = 0

    def _call_llm_batch(self, candidate_id, with_spans):
        parsed = super()._call_llm_batch(candidate_id, with_spans)
        if parsed:
            self.citation_proposed += sum(
                1 for v in parsed if (v.get("cited_chunk_ids") or [])
            )
        else:
            self.llm_fallback_candidates += 1
        return parsed


def _extract_jd_text():
    import pdfplumber
    with pdfplumber.open(JD_PDF) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


def _extract_jd_skills():
    import pipeline
    jd_text = _extract_jd_text()
    struct = pipeline.rule_based_fallback_extraction(jd_text, is_jd=True)
    return jd_text, sorted(struct["skills"])


def _chunk_resumes(gs):
    from rag.chunking import ResumeChunker

    index = {}
    for cand in gs.candidates:
        chunker = ResumeChunker()
        chunks = chunker.chunk(os.path.join(RESUMES_DIR, cand["filename"]))
        assert chunker.candidate_id == cand["candidate_id"], (
            cand["filename"], chunker.candidate_id, cand["candidate_id"])
        index[cand["candidate_id"]] = {
            ch.chunk_id: (ch.raw_text, ch.meta.section_type) for ch in chunks
        }
    return index


def _build_resume_store(embedder, chunks_index):
    store = InMemoryStore(expected_dimension=embedder.get_dimension())
    points = []
    for candidate_id, chunks in chunks_index.items():
        for chunk_id, (raw_text, section_type) in chunks.items():
            points.append({
                "id": chunk_id,
                "vector": embedder.encode([raw_text])[0],
                "payload": {
                    "candidate_id": candidate_id,
                    "chunk_id": chunk_id,
                    "section_type": section_type,
                    "chunker_version": "v1",
                    "embed_model": embedder.model_name,
                    "raw_text": raw_text,
                },
            })
    store.upsert(RagSettings().resume_collection_name, points)
    return store


def _build_esco_store(embedder):
    scripts_dir = os.path.join(REPO, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from build_esco_index import SYNTHETIC_CONCEPTS

    store = InMemoryStore(expected_dimension=embedder.get_dimension())
    OntologyIndexer(store, embedder).index_esco(SYNTHETIC_CONCEPTS)
    return store


def _gold_demonstrated_pairs(gs, chunks_index, slug_to_candidate_id):
    pairs = []
    for L in gs.labels:
        if L["v"] != "demonstrated":
            continue
        cid = slug_to_candidate_id[L["c"]]
        chunk_text = chunks_index[cid][L["chunk"]][0]
        assert L["quote"] in chunk_text, (L["c"], L["chunk"], L["quote"])
        pairs.append({
            "requirement_id": L["r"],
            "requirement": gs.requirement_text(L["r"]),
            "candidate_id": cid,
            "candidate_slug": L["c"],
            "gold_chunk_id": L["chunk"],
        })
    return pairs


def _evidence_signal_ranking(grid, slugs):
    def key(slug):
        row = grid.get(slug, {})
        d = sum(1 for v in row.values() if v == "demonstrated")
        c = sum(1 for v in row.values() if v == "claimed_only")
        return (-d, -c, slugs.index(slug))

    return sorted(slugs, key=key)


def _classify_keyword_window(jd_text, skills):
    """Tier (a): pure keyword section window (Phase 0's split_jd_sections
    heuristic). No LLM involved."""
    from pipeline import split_jd_sections

    sections = split_jd_sections(jd_text)
    req_text_lower = sections["required_text"].lower()
    pref_text_lower = sections["preferred_text"].lower()
    required = [s for s in skills if s.lower() in req_text_lower]
    preferred = [
        s for s in skills
        if s.lower() in pref_text_lower
        and s.lower() not in {r.lower() for r in required}
    ]
    if not required:
        required = skills[:10]
    required_lower = {r.lower() for r in required}
    return {s: ("required" if s.lower() in required_lower else "preferred") for s in skills}


def _classify_requirement_type_tier(jd_text, skills, llm_client):
    """Tier (b): Phase 0's classify_requirement_type using the real client if
    provided, else the keyword-window output. Returns (tier_map, fell_back)."""
    import pipeline

    if not isinstance(llm_client, FailingLLMClient):
        orig = pipeline.client
        pipeline.client = llm_client
        try:
            classification = pipeline.classify_requirement_type(jd_text, skills)
        finally:
            pipeline.client = orig
        if classification and (classification.get("required") or classification.get("preferred")):
            required_lower = {r.lower() for r in classification.get("required") or []}
            tier_map = {
                s: ("required" if s.lower() in required_lower else "preferred")
                for s in skills
            }
            return tier_map, False
        return _classify_keyword_window(jd_text, skills), True
    return _classify_keyword_window(jd_text, skills), True


def run_eval(verbose=True, use_real_llm=None):
    gs = load_gold_set()
    slug_to_candidate_id = {c["slug"]: c["candidate_id"] for c in gs.candidates}

    has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    force_llmless = os.environ.get("RAG_EVAL_LLM", "auto").lower() in (
        "0", "false", "off", "no")
    if use_real_llm is not None:
        real = bool(use_real_llm)
    else:
        real = has_key and not force_llmless
    if real and not has_key:
        real = False
    results_file = RESULTS_V2_FILE if real else RESULTS_V1_FILE

    # 1. JD skills via the real fallback extraction (gold-aligned grid).
    jd_text, skills = _extract_jd_skills()
    assert skills == gs.system_skills, (
        "extraction drift! system skills changed vs gold set:\n  extracted=%s\n  gold=%s"
        % (skills, gs.system_skills)
    )

    # 2. Index resumes with the real embedder.
    embedder = EmbeddingService.get_instance()
    chunks_index = _chunk_resumes(gs)
    t0 = time.time()
    store = _build_resume_store(embedder, chunks_index)
    t_index = time.time() - t0

    # 3. ESCO stand-in + real alt-label provider.
    esco_store = _build_esco_store(embedder)
    llm_client = UsageTrackingLLMClient(_real_pipeline_client()) if real else FailingLLMClient()
    normalizer = SkillNormalizer(esco_store, embedder, llm_client)

    def skill_alt_labels(skill):
        try:
            return normalizer.alt_labels_for(skill) or []
        except Exception:
            return []

    def requirement_alt_labels(requirement_text):
        term = CONFIG_ALT_LABELS.get(requirement_text)
        return skill_alt_labels(term) if term else []

    # Three-tier requirement classification (a/b/c).
    tier_a = _classify_keyword_window(jd_text, skills)
    tier_b, b_fell_back = _classify_requirement_type_tier(jd_text, skills, llm_client)
    llm_extraction_diagnostic = None
    if real:
        import pipeline as _p2
        orig = _p2.client
        _p2.client = llm_client
        try:
            llm_struct = _p2.call_llm_extraction(jd_text, is_jd=True)
        finally:
            _p2.client = orig
        normalized = normalizer.normalize(jd_text, skills)
        tier_c = {r.skill_name: r.tier for r in normalized}
        req_sources = sorted({r.source for r in normalized})
        requirements = normalized
        llm_extraction_diagnostic = {
            "n_skills": len(llm_struct.get("skills", [])),
            "skills": sorted(llm_struct.get("skills", [])),
            "fallback_sentinel": llm_struct.get("semantic_summary", "").startswith(
                "Auto-generated by rule-based fallback"),
        }
    else:
        tier_c = {s: "required" for s in skills}
        req_sources = ["keyword_fallback"]
        requirements = [
            NormalizedRequirement(skill_name=s, tier="required", source="keyword_fallback")
            for s in skills
        ]

    # 4. Evidence retrieval + per-candidate verdicts (real batched LLM in v2,
    # deterministic CE-score fallback in v1).
    reranker = CrossEncoderReranker()
    retriever = SkillEvidenceRetriever(
        HybridRetriever(store, embedder, RRFFuser()), reranker
    )
    verify_class = RealLLMEvidenceVerifier if real else EvidenceVerifier
    verifier = verify_class(
        retriever, llm_client, CitationValidator(),
        alt_labels_provider=skill_alt_labels,
    )

    system_skill_verdicts = {s: {} for s in skills}
    system_skill_best_ce = {s: {} for s in skills}
    verdict_times = []
    rejected_total = 0
    for cand in gs.candidates:
        t0 = time.time()
        evidence = verifier.verify(cand["candidate_id"], requirements)
        verdict_times.append((cand["slug"], round(time.time() - t0, 2)))
        rejected_total += evidence.rejected_citations
        by_skill = {v.skill_name: v.verdict for v in evidence.verdicts}
        for s in skills:
            system_skill_verdicts[s][cand["slug"]] = by_skill.get(s, "absent")
        for req in requirements:
            spans = verifier._retrieve_for(cand["candidate_id"], req)
            best = max((sp.cross_encoder_score or 0.0) for sp in spans) if spans else 0.0
            system_skill_best_ce[req.skill_name][cand["slug"]] = round(best, 4)

    # 5. Metrics vs the 161-label gold set.
    predicted_grid = aggregate_alias_verdicts(system_skill_verdicts, gs.system_skill_alias)
    gold_grid = {
        r["id"]: {c["slug"]: gs.gold_verdict(r["id"], c["slug"]) for c in gs.candidates}
        for r in gs.requirements
    }
    va = verdict_accuracy(gold_grid, predicted_grid)
    fms = false_missing_skill_rate(gold_grid, predicted_grid)

    gold_pairs = _gold_demonstrated_pairs(gs, chunks_index, slug_to_candidate_id)

    # 6. Six-row retrieval ablation (real embedder + real cross-encoder).
    ablation = RetrievalAblation(
        store, embedder,
        reranker=CrossEncoderReranker(),
        alt_labels_provider=requirement_alt_labels,
    )
    configs = [
        "dense_only", "bm25_only", "hybrid", "hybrid_ce",
        "hybrid_ce+ontology_expansion", "hybrid_ce+header_prefix",
    ]
    t0 = time.time()
    abl = ablation.run(gold_pairs, configs=configs)
    t_ablation = time.time() - t0

    # 7. Evidence-signal ranking vs hand ranking.
    pred_rank = _evidence_signal_ranking(predicted_grid, gs.hand_ranking)
    tau = kendall_tau(gs.hand_ranking, pred_rank)

    source_distribution = None
    if real:
        from collections import Counter
        source_distribution = dict(Counter(r.source for r in requirements))

    three_tier = {
        "a_keyword_window": tier_a,
        "b_classify_requirement_type": tier_b,
        "b_fell_back_to_keywords": b_fell_back,
        "c_skillnormalizer_llm": tier_c,
        "jd_reference_tiers": JD_REFERENCE_TIERS,
        "agreement_ab": sum(1 for s in skills if tier_a[s] == tier_b[s]) / len(skills),
        "agreement_ac": sum(1 for s in skills if tier_a[s] == tier_c[s]) / len(skills),
        "agreement_bc": sum(1 for s in skills if tier_b[s] == tier_c[s]) / len(skills),
        "accuracy_a_vs_jd": sum(1 for s in skills if tier_a[s] == JD_REFERENCE_TIERS[s]) / len(skills),
        "accuracy_b_vs_jd": sum(1 for s in skills if tier_b[s] == JD_REFERENCE_TIERS[s]) / len(skills),
        "accuracy_c_vs_jd": sum(1 for s in skills if tier_c[s] == JD_REFERENCE_TIERS[s]) / len(skills),
        "normalized_requirement_sources": req_sources,
        "source_distribution": source_distribution,
        "verdict_grid_tier_note": (
            "NormalizedRequirement.tier is not consulted by the evidence "
            "retrieval/verdict path, so verdict accuracy is identical across "
            "tiers a/b/c (which differ only in required/preferred assignment)."
        ),
    }

    llm_usage = None
    if real:
        llm_usage = {
            "provider": "OpenRouter (openai/gpt-4o-mini)",
            "calls": llm_client.calls,
            "prompt_tokens": llm_client.prompt_tokens,
            "completion_tokens": llm_client.completion_tokens,
            "total_tokens": llm_client.prompt_tokens + llm_client.completion_tokens,
            "llm_wall_s": round(llm_client.llm_wall_s, 3),
            "http_errors": llm_client.errors,
        }

    citation_proposed = verifier.citation_proposed if real else 0
    citation_rejected = rejected_total if real else 0
    cv = citation_validity_rate(citation_proposed, citation_rejected) if real else None

    result = {
        "evaluation_id": "run_eval_v2" if real else "run_eval_v1",
        "date": time.strftime("%Y-%m-%d"),
        "environment": {
            "openrouter_api_key": has_key,
            "llm_additive": "ON" if real else "OFF",
            "embedder": embedder.model_name,
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "esco": SYNTHETIC_ESCO_WARNING,
        },
        "skills": skills,
        "gold_rows_with_alias": len([v for v in gs.system_skill_alias.values() if v]),
        "gold_rows_without_alias": len([v for v in gs.system_skill_alias.values() if not v]),
        "index_build_s": round(t_index, 3),
        "retrieval_ablation": abl,
        "ablation_wall_s": round(t_ablation, 3),
        "verdict_accuracy": va,
        "false_missing_skill_rate": fms,
        "system_skill_verdicts": system_skill_verdicts,
        "system_skill_best_ce": system_skill_best_ce,
        "predicted_grid": predicted_grid,
        "gold_grid": gold_grid,
        "kendall_tau": tau,
        "predicted_ranking": pred_rank,
        "hand_ranking": gs.hand_ranking,
        "candidate_verdict_times_s": verdict_times,
    }
    if real:
        result.update({
            "three_tier_classification": three_tier,
            "llm_extraction_diagnostic": llm_extraction_diagnostic,
            "llm_usage": llm_usage,
            "evidence_fallback_candidates": verifier.llm_fallback_candidates,
            "citation_proposed": citation_proposed,
            "citation_rejected": citation_rejected,
            "citation_validity": cv,
        })
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        header = "Phase 9 rerun (real LLM, gpt-4o-mini)" if real else "Phase 9 evaluation (real models, LLMless)"
        print("=== %s ===" % header)
        print("extracted skills (%d): %s" % (len(skills), skills))
        print("index_wall: %.1fs | ablation_wall: %.1fs" % (t_index, t_ablation))
        print("-- retrieval ablation recall@3 (n=68 gold demonstrated pairs) --")
        for cfg, r in abl.items():
            rv = r.get("recall@3")
            print("  %-30s %s" % (cfg, ("%.3f" % rv) if rv is not None else "(gap)"))
        print("verdict_accuracy:", va["accuracy"], "(%d/%d)" % (va["correct"], va["total"]))
        print("  per-gold-verdict:", va["per_gold_verdict"])
        print("false_missing_skill_rate:", fms["false_missing_skill_rate"],
              "missed=%d gold_demonstrated=%d" % (fms["missed"], fms["gold_demonstrated"]))
        print("kendall_tau:", tau["kendall_tau"], "| predicted:", pred_rank)
        print("  caveat:", tau["caveat"])
        if real:
            print("citation_validity:", cv["citation_validity_rate"],
                  "proposed=%d rejected=%d" % (citation_proposed, citation_rejected))
            print("evidence_llm_fallback_candidates:", verifier.llm_fallback_candidates, "of", len(gs.candidates))
            print("three-tier agreement ab/ac/bc: %.3f/%.3f/%.3f  (vs JD ref: a=%.3f b=%.3f c=%.3f)" % (
                three_tier["agreement_ab"], three_tier["agreement_ac"], three_tier["agreement_bc"],
                three_tier["accuracy_a_vs_jd"], three_tier["accuracy_b_vs_jd"], three_tier["accuracy_c_vs_jd"]))
            print("llm_usage:", llm_usage)
        print("results ->", results_file)
    return result


def _real_pipeline_client():
    import pipeline
    return pipeline.client


if __name__ == "__main__":
    run_eval()