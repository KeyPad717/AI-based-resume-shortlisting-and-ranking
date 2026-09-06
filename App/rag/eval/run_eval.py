"""Phase 9: full real evaluation (CLI).

Run from App/ (repo root parent is the project dir):

    cd App
    python3 -m rag.eval.run_eval

What it does (all REAL code paths, real models loaded):
  1. Extract the JD skill list with pipeline.rule_based_fallback_extraction (the
     real LLM-less fallback; OPENROUTER_API_KEY unset in this environment, so the
     gpt-4o-mini tier cannot run).
  2. Chunk all 7 resumes with the real ResumeChunker and embed them with the real
     all-mpnet-base-v2 embedder into an InMemoryStore.
  3. Build the ESCO collection (SYNTHETIC stand-in, see build_esco_index.py) with
     the same embedder, and normalize-ish the JD skills.
  4. Run the real SkillEvidenceRetriever + CrossEncoderReranker (ms-marco
     MiniLM-L-6-v2) + EvidenceVerifier's DETERMINISTIC CE-score fallback per
     candidate to get the LLM-less system verdict for every extracted skill.
  5. Compute the six-row retrieval ablation recall@3 against the gold
     demonstrated-evidence pairs, verdict accuracy / false-missing, and an
     evidence-signal Kendall tau vs the hand ranking.

Imports of pipeline and the heavy model classes happen at call time so this
module stays importable from tests.
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))  # App/rag/eval
APP = os.path.join(HERE, "..", "..")
sys.path.insert(0, APP)

from rag.config import RagSettings
from rag.embedding import EmbeddingService
from rag.eval.ablate import RetrievalAblation
from rag.eval.gold_set import load_gold_set
from rag.eval.metrics import (
    aggregate_alias_verdicts,
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
RESULTS_FILE = os.path.join(HERE, "results", "run_eval_v1.json")

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


class FailingLLMClient:
    """Never-reachable LLM: every call raises instantly, reproducing the LLM-less
    environment without hammering the network."""

    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("deterministic eval failure (no LLM available)")

    def __init__(self):
        self.chat = type("_chat", (), {"completions": self._Completions()})


def _extract_jd_text():
    import pdfplumber
    with pdfplumber.open(JD_PDF) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)


def _extract_jd_skills():
    # Real fallback extraction (LLM tier unavailable). Lazy import keeps this
    # module importable without pipeline's side effects.
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
    # build_esco_index.py lives at <repo>/scripts/ (sibling of App/). Its module
    # scope inserts App/ onto sys.path for rag.* imports, so importing it here is
    # safe as long as the scripts dir is importable.
    scripts_dir = os.path.join(REPO, "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from build_esco_index import SYNTHETIC_CONCEPTS

    store = InMemoryStore(expected_dimension=embedder.get_dimension())
    OntologyIndexer(store, embedder).index_esco(SYNTHETIC_CONCEPTS)
    return store


def _gold_demonstrated_pairs(gs, chunks_index, slug_to_candidate_id):
    """68 gold demonstrated rows -> ablation pairs, verifying each verbatim quote
    against the stored chunk text (the CitationValidator-style guarantee)."""
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
            # The retrieval store is filtered by the full candidate GUID (the
            # chunker's content hash), NOT the readable slug.
            "candidate_id": cid,
            "candidate_slug": L["c"],
            "gold_chunk_id": L["chunk"],
        })
    return pairs


def _evidence_signal_ranking(grid, slugs):
    """Predicted ranking from the system verdict grid: demonstrated first, claimed
    next, hand-ranking order as tie-break. Explicitly NOT final_score (which also
    needs tiers/section/education weights), reported as such."""
    def key(slug):
        row = grid.get(slug, {})
        d = sum(1 for v in row.values() if v == "demonstrated")
        c = sum(1 for v in row.values() if v == "claimed_only")
        return (-d, -c, slugs.index(slug))

    return sorted(slugs, key=key)


def run_eval(verbose=True):
    gs = load_gold_set()
    slug_to_candidate_id = {c["slug"]: c["candidate_id"] for c in gs.candidates}

    # 1. JD skills via the real fallback extraction.
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

    # 3. ESCO stand-in + real alt-label provider (real ESCO lookup for the
    # ontology-expansion behavior of retrieval and evidence tiers).
    esco_store = _build_esco_store(embedder)
    normalizer = SkillNormalizer(esco_store, embedder, FailingLLMClient())

    def skill_alt_labels(skill):
        try:
            return normalizer.alt_labels_for(skill) or []
        except Exception:
            return []

    def requirement_alt_labels(requirement_text):
        term = CONFIG_ALT_LABELS.get(requirement_text)
        return skill_alt_labels(term) if term else []

    # Requirements: real fallback extraction produced these exact skills; assign
    # the gold-consistent required/preferred tiers via the real keyword fallback,
    # then run the verdict path on them.
    requirements = [
        NormalizedRequirement(skill_name=s, tier="required", source="keyword_fallback")
        for s in skills
    ]

    # 4. Evidence retrieval + deterministic CE verdicts per candidate (LLM off).
    reranker = CrossEncoderReranker()
    retriever = SkillEvidenceRetriever(
        HybridRetriever(store, embedder, RRFFuser()), reranker
    )
    verifier = EvidenceVerifier(
        retriever, FailingLLMClient(), CitationValidator(),
        alt_labels_provider=skill_alt_labels,
    )

    system_skill_verdicts = {s: {} for s in skills}
    system_skill_best_ce = {s: {} for s in skills}
    verdict_times = []
    for cand in gs.candidates:
        t0 = time.time()
        evidence = verifier.verify(cand["candidate_id"], requirements)
        verdict_times.append((cand["slug"], round(time.time() - t0, 2)))
        by_skill = {v.skill_name: v.verdict for v in evidence.verdicts}
        for s in skills:
            system_skill_verdicts[s][cand["slug"]] = by_skill.get(s, "absent")
        # Best raw (post-floor) cross-encoder score per skill, read from the SAME
        # retrieval path the verifier uses (EvidenceVerifier._retrieve_for), so
        # the CE-fallback threshold can be swept offline against the gold set.
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

    result = {
        "evaluation_id": "run_eval_v1",
        "date": time.strftime("%Y-%m-%d"),
        "environment": {
            "openrouter_api_key": bool(os.environ.get("OPENROUTER_API_KEY")),
            "embedder": embedder.model_name,
            "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "esco": SYNTHETIC_ESCO_WARNING,
            "llm_additive": "OFF",
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
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print("=== Phase 9 evaluation (real models, LLMless) ===")
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
        print("results ->", RESULTS_FILE)
    return result


if __name__ == "__main__":
    run_eval()