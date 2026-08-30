import numpy as np
import os
import re
import json
import docx
import spacy
import pdfplumber
import openai
import math
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from sentence_transformers import SentenceTransformer, CrossEncoder, util

print("Loading NLP Models...")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

try:
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("abbreviation_detector")
except (OSError, ValueError) as e:
    print(f"Falling back to regular spaCy ({e})")
    nlp = spacy.load("en_core_web_sm")

if not os.environ.get("OPENROUTER_API_KEY"):
    print("WARNING: OPENROUTER_API_KEY not set. LLM extraction will be skipped; rule-based fallback will be used for all documents.")

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY") or "not-needed",
)

REQUIRED_CUES = ["must have", "required", "requirements", "mandatory", "essential", "should have", "you should have", "we are looking for"]
PREFERRED_CUES = ["good to have", "nice to have", "preferred", "plus", "bonus"]
fallback_abbreviations = {"ml": "machine learning", "ai": "artificial intelligence", "nlp": "natural language processing", "dl": "deep learning", "cv": "computer vision"}

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content: text += content + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text(file_path):
    if file_path.lower().endswith(".pdf"): return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"): return extract_text_from_docx(file_path)
    return ""

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fix_numbers(text):
    text = re.sub(r'\b([0-4])\s+([0-9]{2})\b', r'\1.\2', text)
    text = re.sub(r'\b([0-9])\s+00\b', r'\1.00', text)
    return text

def normalize_text(text):
    doc = nlp(text)
    expanded_text = text.lower()
    if hasattr(doc._, 'abbreviations'):
        for abrv in doc._.abbreviations:
            expanded_text = re.sub(r'\b' + re.escape(str(abrv).lower()) + r'\b', str(abrv._.long_form).lower(), expanded_text)
    for abbr, full in fallback_abbreviations.items():
        expanded_text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, expanded_text)

    doc = nlp(expanded_text)
    tokens = []
    i = 0
    while i < len(doc):
        token = doc[i]
        if i < len(doc) - 2 and doc[i].like_num and doc[i+1].text == "." and doc[i+2].like_num:
            tokens.append(doc[i].text + "." + doc[i+2].text)
            i += 3; continue
        if not token.is_stop and not token.is_punct:
            tokens.append(token.lemma_)
        i += 1
    return " ".join(tokens)

def normalize_skill(skill):
    skill = str(skill).lower().strip()
    synonyms = {'ml': 'machine learning', 'ai': 'artificial intelligence', 'nlp': 'natural language processing', 'js': 'javascript', 'ts': 'typescript', 'py': 'python'}
    return synonyms.get(skill, skill)

def compute_semantic_similarity(jd_text, resume_text):
    jd_vec = embedding_model.encode(jd_text, convert_to_tensor=True)
    res_vec = embedding_model.encode(resume_text, convert_to_tensor=True)
    return float(max(0.0, util.cos_sim(jd_vec, res_vec).item()))

def compute_reranker_score(jd_text, resume_text):
    score = reranker_model.predict([jd_text[:1200], resume_text[:1200]])
    return float(1 / (1 + math.exp(-score / 1.5)))

def compute_evidence_score(matched_skills, resume_text, projects):
    if not matched_skills: return 0.0
    text = (resume_text + " " + " ".join(map(str, projects))).lower()
    count = sum(1 for skill in matched_skills if normalize_skill(skill) in text)
    return float(count / len(matched_skills))

def normalize_degree(deg):
    if not deg: return ""
    deg = str(deg).lower()
    if any(x in deg for x in ["btech", "b.tech", "bachelor", "b.e.", "be"]): return "bachelor"
    if any(x in deg for x in ["mtech", "m.tech", "master", "ms", "m.e.", "me"]): return "master"
    if any(x in deg for x in ["phd", "ph.d", "doctorate"]): return "doctorate"
    return deg

def education_score(jd_edu, resume_edu, resume_text):
    if not jd_edu: return 1.0
    jd_norms = {normalize_degree(e) for e in jd_edu}
    res_norms = {normalize_degree(e) for e in resume_edu}
    if jd_norms.intersection(res_norms): return 1.0
    for req in jd_norms:
        for res in res_norms:
            if req in res or res in req: return 1.0
    return 0.0

def build_extraction_prompt(text, is_jd=False):
    role = "expert resume parser" if not is_jd else "expert job description analyzer"
    subject = "resume" if not is_jd else "job description"
    exp_rule = "Include ONLY professional work experience entries." if not is_jd else "Include minimum required years of experience."
    name_extraction = ""
    if not is_jd: name_extraction = '6. "candidate_name": Extract the full name of the candidate. Exclude institute names, locations, or degree titles.'

    return f"""
You are an {role}. Analyze the {subject} text provided below and extract both semantic descriptions and structured data.
STRICT RULES FOR EXTRACTION:
1. "experience": {exp_rule}
   - Include: Full-time jobs, internships (with role, type, duration_years).
   - EXCLUDE: Education (BTech, MTech, degrees), academic projects, training courses.
2. "skills": Extract ALL technical skills.
3. "education": Extract degree names only (BTech, MTech, etc.).
4. "projects": Include project titles or short descriptions (1 sentence).
5. "semantic_summary": Provide a 2-3 sentence professional summary focusing on candidate's technical profile.
{name_extraction}
OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "candidate_name": "full name",
  "skills": ["list of strings"],
  "experience": [{{"role": "title", "type": "internship/full-time", "duration_years": 0}}],
  "education": ["degrees"],
  "projects": ["descriptions"],
  "semantic_summary": "summary text"
}}
{subject.capitalize()}:
{text}
"""

def rule_based_fallback_extraction(text, is_jd=False):
    tech_keywords = ['python', 'java', 'c++', 'c', 'javascript', 'typescript', 'go', 'ruby', 'react', 'node.js', 'angular', 'sql', 'mysql', 'postgres', 'mongodb', 'aws', 'docker', 'kubernetes', 'linux', 'unix', 'machine learning', 'deep learning', 'nlp', 'computer vision', 'html', 'css', 'spring', 'django', 'flask', 'fastapi']
    text_lower = text.lower()
    found_skills = [skill for skill in tech_keywords if skill in text_lower or f" {skill} " in text_lower]
    
    years = 0
    exp_matches = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+experience', text_lower)
    if exp_matches:
        try:
            years = max([int(m) for m in exp_matches])
        except ValueError: pass
            
    name = ""
    if not is_jd:
        doc = nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text
                break
                
    return {
        "candidate_name": name,
        "skills": sorted(set(found_skills)),
        "experience": [{"role": "unspecified", "type": "full-time", "duration_years": years}] if years > 0 else [],
        "education": [],
        "projects": [],
        "semantic_summary": "Auto-generated by rule-based fallback due to LLM extraction failure."
    }

def call_llm_extraction(text, is_jd=False, retries=2):
    prompt = build_extraction_prompt(text[:4000], is_jd=is_jd)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*|^```\s*|```$', '', raw, flags=re.MULTILINE).strip()
            return json.loads(raw)
        except Exception as e:
            if attempt == retries - 1:
                print(f"LLM Extraction failed, running deterministic fallback. Error: {e}")
                return rule_based_fallback_extraction(text, is_jd)
            time.sleep(1 * (attempt + 1))

def validate_experience(exp):
    if exp > 10: return 5.0
    if exp < 0: return 0.0
    return exp

def compute_experience_years(experience_list, resume_text, is_jd=False):
    total = sum(float(e.get("duration_years", 0)) for e in experience_list if isinstance(e, dict))
    return validate_experience(total)

def validate_extracted_struct(struct, resume_text, is_jd=False):
    if struct is None:
        return {
            "name": "",
            "skills": [],
            "experience_years": 0,
            "experience_breakdown": [],
            "education": [],
            "projects": [],
            "semantic_summary": ""
        }
    exp_entries = struct.get("experience", [])
    return {
        "name": struct.get("candidate_name", "").strip(),
        "skills": [s.strip() for s in struct.get("skills", []) if isinstance(s, str)],
        "experience_years": compute_experience_years(exp_entries, resume_text, is_jd=is_jd),
        "experience_breakdown": exp_entries,
        "education": [e.strip() for e in struct.get("education", []) if isinstance(e, str)],
        "projects": [p.strip() for p in struct.get("projects", []) if isinstance(p, str)],
        "semantic_summary": struct.get("semantic_summary", "")
    }

def split_jd_sections(text):
    lower_text = text.lower()
    req_parts, pref_parts = [], []
    for cue in REQUIRED_CUES:
        idx = lower_text.find(cue)
        if idx != -1: req_parts.append(text[idx: idx + 1000])
    for cue in PREFERRED_CUES:
        idx = lower_text.find(cue)
        if idx != -1: pref_parts.append(text[idx: idx + 700])
    return {
        "required_text": "\n".join(req_parts) if req_parts else text,
        "preferred_text": "\n".join(pref_parts)
    }

def classify_requirement_type(jd_text, skills, retries=2):
    skill_list = "\n".join(f"- {s}" for s in skills)
    prompt = f"""
You are analyzing a job description. Classify each skill below as either REQUIRED (mandatory, must-have) or PREFERRED (optional, nice-to-have, plus) based on how the job description text describes each skill.

Job description:
{jd_text[:4000]}

Skills to classify:
{skill_list}

OUTPUT FORMAT (STRICT JSON ONLY):
{{"required": ["skill1", "skill2"], "preferred": ["skill3"]}}
"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*|^```\s*|```$', '', raw, flags=re.MULTILINE).strip()
            result = json.loads(raw)
            if isinstance(result, dict) and isinstance(result.get("required"), list) and isinstance(result.get("preferred"), list):
                return result
            return None
        except Exception as e:
            if attempt == retries - 1:
                print(f"Requirement classification failed, using keyword fallback. Error: {e}")
                return None
            time.sleep(1 * (attempt + 1))
    return None

def normalize_weights(weights):
    total = sum(weights.values())
    if total <= 0:
        weights = {"required_skills": 0.32, "semantic": 0.16, "reranker": 0.16, "experience": 0.10, "education": 0.06, "projects": 0.07, "evidence": 0.13}
        total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}

def parse_jd_requirements(jd_text, jd_struct, user_weights=None):
    all_skills = jd_struct["skills"]
    classification = classify_requirement_type(jd_text, all_skills)
    if classification and (classification["required"] or classification["preferred"]):
        req_lower = {r.lower() for r in classification.get("required", [])}
        pref_lower = {p.lower() for p in classification.get("preferred", [])}
        required_skills = [s for s in all_skills if s.lower() in req_lower] or all_skills[:10]
        preferred_skills = [s for s in all_skills if s.lower() in pref_lower and s.lower() not in {r.lower() for r in required_skills}]
    else:
        sections = split_jd_sections(jd_text)
        req_text_lower = sections["required_text"].lower()
        pref_text_lower = sections["preferred_text"].lower()
        required_skills = [s for s in all_skills if s.lower() in req_text_lower] or all_skills[:10]
        preferred_skills = [s for s in all_skills if s.lower() in pref_text_lower and s.lower() not in {r.lower() for r in required_skills}]

    if user_weights:
        # Use user weights but ensure they are float 0-1
        weights = {k: max(0, float(v)) / 100.0 for k, v in user_weights.items()}
    else:
        weights = {"required_skills": 0.32, "semantic": 0.16, "reranker": 0.16, "experience": 0.10, "education": 0.06, "projects": 0.07, "evidence": 0.13}
        if any(kw in jd_text.lower() for kw in ["senior", "lead", "years of experience"]):
            weights["experience"] += 0.10; weights["semantic"] -= 0.10
        if jd_struct["experience_years"] == 0:
            weights["required_skills"] += weights["experience"] * 0.5
            weights["projects"] += weights["experience"] * 0.5
            weights["experience"] = 0.0
        if not jd_struct["education"]:
            weights["reranker"] += weights["education"] * 0.5
            weights["semantic"] += weights["education"] * 0.5
            weights["education"] = 0.0

    return {"required_skills": required_skills, "preferred_skills": preferred_skills, "experience_years": jd_struct["experience_years"], "education": jd_struct["education"], "weights": normalize_weights(weights)}

def use_skill_normalizer(jd_text, all_skills, ontology_store, embedder, llm_client):
    """Thin wrapper around SkillNormalizer.normalize() (Phase 5).

    NOTE (deliberate deviation from the roadmap): this is NOT yet wired into
    parse_jd_requirements. Phase 0's classify_requirement_type and this phase's
    parse_jd_requirements are left completely unchanged and still live; the
    switch to the ESCO-normalized path, and deciding how the two coexist, is left
    to Phase 6 (which will also wire the new scoring path). Retiring Phase 0 now
    would leave the app invoking a half-integrated component. This function only
    makes SkillNormalizer available as a new, separate entry point.
    """
    from rag.ontology import SkillNormalizer  # lazy: keep pipeline import cheap

    normalizer = SkillNormalizer(ontology_store, embedder, llm_client)
    return normalizer.normalize(jd_text, all_skills)

# ---------------------------------------------------------------------------
# Phase 6: RAG-scored signals (wired only when RagSettings().rag_enabled).
# The legacy weighted-sum path above stays the source of truth when RAG is off,
# so final_score arithmetic is byte-identical to before. When RAG is on we
# recompute exactly four signals (evidence, semantic, reranker, required_skills)
# and leave experience/education/project untouched, then fall through to the
# SAME weighted-sum combination for final_score.
# ---------------------------------------------------------------------------

# Grounded-evidence verdict -> numeric weight (see roadmap §Evidence weighting).
_VERDICT_VALUE = {"demonstrated": 1.0, "claimed_only": 0.35, "absent": 0.0}

# A preferred-tier requirement contributes 1/3 the weight of a required one.
PREFERRED_TIER_WEIGHT_RATIO = 1.0 / 3.0

# Provisional rescale of the Phase 4 RRF fused_rank (a tiny 1/(60+rank) value)
# into a comparable magnitude for the "semantic" signal. Re-validated in Phase 9
# against the gold set.
def _rag_dense_scale():
    from rag.config import RagSettings as _RS
    return float(_RS().dense_top_k)

# Because the gated required_skills signal now restates the same grounded
# evidence/verdict information, we SHIFT weight out of it (0.32 -> 0.25) into
# the verified evidence signal (0.13 -> 0.20). The combined 0.45 is unchanged,
# so every other signal is left relatively untouched and the total stays 1.0.
# This split is PROVISIONAL and must be re-tuned against the Phase 9 gold set.
_RAG_BASE_WEIGHTS = {
    "required_skills": 0.25,
    "semantic": 0.16,
    "reranker": 0.16,
    "experience": 0.10,
    "education": 0.06,
    "projects": 0.07,
    "evidence": 0.20,
}

# Module-level memo of candidate_ids already embedded+indexed in the current
# process, so a repeated request for the same resume does not re-embed it.
_rag_indexed_candidate_ids = set()

# Lazy singleton for the Phase 6 RAG runtime (stores + embedder + reranker +
# llm client). Built on first use so the rag_disabled path never pays the cost.
_rag_runtime_cache = {}


def get_rag_stores():
    """Return the lazy-singleton RAG runtime: ontology_store, resume_store,
    embedder, reranker, llm_client.

    Constructs an InMemoryStore or QdrantStore based on the RAG_STORE env var
    ("qdrant" selects QdrantStore; anything else is an in-memory store). Nothing
    is built at module import time; the first call here pays the model/collection
    setup cost (and only when the RAG path is actually used).
    """
    if _rag_runtime_cache:
        return _rag_runtime_cache

    from rag.config import RagSettings as _RS
    from rag.store import InMemoryStore, QdrantStore
    from rag.embedding import EmbeddingService
    from rag.retrieval import CrossEncoderReranker

    settings = _RS()
    backend = os.environ.get("RAG_STORE", "inmemory").lower()
    if backend == "qdrant":
        vector_size = EmbeddingService.get_instance().get_dimension()
        resume_store = QdrantStore(url=settings.qdrant_url, vector_size=vector_size)
        ontology_store = QdrantStore(url=settings.qdrant_url, vector_size=vector_size)
    else:
        resume_store = InMemoryStore()
        ontology_store = InMemoryStore()

    _rag_runtime_cache.update({
        "ontology_store": ontology_store,
        "resume_store": resume_store,
        "embedder": EmbeddingService.get_instance(),
        "reranker": CrossEncoderReranker(),
        "llm_client": client,
    })
    return _rag_runtime_cache


def _ensure_indexed(r_path, resume_store, embedder):
    """Ingest a resume file into chunks and index them ONCE per candidate.

    Returns the candidate_id (a content hash of the resume's extracted text from
    the Phase 2 chunker). On a successful ingest we embed+upsert only the first
    time a candidate is seen this process; later calls skip re-embedding. A
    genuinely failed extraction raises so the caller can fall back to legacy.
    """
    from rag.ingest import ResumeIngestor
    from rag.index import ResumeIndexer

    ingestor = ResumeIngestor()
    chunks, report = ingestor.ingest(r_path)
    candidate_id = report.candidate_id
    if not candidate_id or not chunks:
        raise RuntimeError("resume ingestion produced no indexable chunks")
    if candidate_id not in _rag_indexed_candidate_ids:
        ResumeIndexer(resume_store, embedder).index_chunks(chunks)
        _rag_indexed_candidate_ids.add(candidate_id)
    return candidate_id


def _tier_weight(requirement):
    return PREFERRED_TIER_WEIGHT_RATIO if requirement.tier == "preferred" else 1.0


def _rag_weights(jd_text, jd_struct):
    """Re-derived weights for the RAG path.

    Mirrors parse_jd_requirements' dynamic adjustments (senior/lead bump, and the
    experience=0 / education-missing redistributions) but starts from the Phase 6
    re-split base (0.25 required_skills / 0.20 evidence) instead of the legacy
    (0.32 / 0.13). Values are renormalized so the 7 signals still sum to 1.0.
    """
    weights = dict(_RAG_BASE_WEIGHTS)
    if any(kw in jd_text.lower() for kw in ["senior", "lead", "years of experience"]):
        weights["experience"] += 0.10
        weights["semantic"] -= 0.10
    if jd_struct["experience_years"] == 0:
        weights["required_skills"] += weights["experience"] * 0.5
        weights["projects"] += weights["experience"] * 0.5
        weights["experience"] = 0.0
    if not jd_struct["education"]:
        weights["reranker"] += weights["education"] * 0.5
        weights["semantic"] += weights["education"] * 0.5
        weights["education"] = 0.0
    return normalize_weights(weights)


class _CachedEvidenceRetriever:
    """EvidenceVerifier adapter that returns pre-retrieved spans per skill.

    Phase 6 retrieves each requirement's spans ONCE (for the semantic/reranker
    signals) and feeds the exact same spans into the EvidenceVerifier so we never
    run a second retrieval pass. Forwarded alt_labels are ignored because they
    already shaped the cached retrieval.
    """

    def __init__(self, spans_by_skill):
        self._spans = spans_by_skill

    def for_requirement(self, requirement, candidate_id, alt_labels=None):
        return list(self._spans.get(requirement.skill_name, []))


def score_candidate_rag(jd_text, resume_text, r_struct, jd_struct, r_path,
                        ontology_store, resume_store, embedder, reranker, llm_client):
    """Recompute the four RAG-grounded signals for one candidate.

    Only evidence, semantic, reranker and required_skills are produced here;
    experience/education/project are intentionally left in process_resumes so
    they stay unchanged. Returns a dict with those four signals, the re-derived
    RAG weights, matched/missing skills, and the per-skill verdicts.
    """
    from rag.retrieval import (
        CrossEncoderReranker,
        HybridRetriever,
        RRFFuser,
        SkillEvidenceRetriever,
    )
    from rag.ontology import SkillNormalizer
    from rag.evidence import CitationValidator, EvidenceVerifier

    candidate_id = _ensure_indexed(r_path, resume_store, embedder)

    normalizer = SkillNormalizer(ontology_store, embedder, llm_client)
    requirements = normalizer.normalize(jd_text, jd_struct["skills"])

    if not isinstance(reranker, CrossEncoderReranker):
        reranker = CrossEncoderReranker(model=reranker)

    skill_retriever = SkillEvidenceRetriever(
        HybridRetriever(resume_store, embedder, RRFFuser()), reranker
    )

    # Retrieve each requirement's spans once; keep them for the evidence verifier.
    spans_by_skill = {}
    semantic_parts, rerank_parts = [], []
    for req in requirements:
        alt = normalizer.alt_labels_for(req.skill_name)
        spans = skill_retriever.for_requirement(req, candidate_id, alt_labels=alt)
        spans_by_skill[req.skill_name] = spans
        top = spans[:3]
        if top:
            semantic_parts.append(
                float(np.mean([s.fused_rank for s in top])) * _rag_dense_scale()
            )
            rerank_parts.append(
                float(np.mean([(s.cross_encoder_score or 0.0) for s in top]))
            )
        else:
            semantic_parts.append(0.0)
            rerank_parts.append(0.0)

    semantic_score = float(np.mean(semantic_parts)) if semantic_parts else 0.0
    rerank_score = float(np.mean(rerank_parts)) if rerank_parts else 0.0

    verifier = EvidenceVerifier(
        _CachedEvidenceRetriever(spans_by_skill),
        llm_client,
        CitationValidator(),
        alt_labels_provider=normalizer.alt_labels_for,
    )
    candidate_evidence = verifier.verify(candidate_id, requirements)
    by_name = {v.skill_name: v for v in candidate_evidence.verdicts}

    # Evidence = tier-weighted verdict aggregation.
    num_w, den_w = 0.0, 0.0
    matched, missing = [], []
    for req in requirements:
        tw = _tier_weight(req)
        verdict = by_name.get(req.skill_name).verdict if by_name.get(req.skill_name) else "absent"
        num_w += tw * _VERDICT_VALUE[verdict]
        den_w += tw
        (matched if verdict != "absent" else missing).append(req.skill_name)
    evidence_score = float(num_w / den_w) if den_w else 0.0

    # required_skills = ontology-expanded coverage gated on verdict != absent.
    gate_num = sum(
        _tier_weight(req)
        for req in requirements
        if by_name.get(req.skill_name) and by_name[req.skill_name].verdict != "absent"
    )
    required_score = float(gate_num / den_w) if den_w else 0.0

    return {
        "required_skills": required_score,
        "semantic": semantic_score,
        "reranker": rerank_score,
        "evidence": evidence_score,
        "weights": _rag_weights(jd_text, jd_struct),
        "matched": matched,
        "missing": missing,
        "candidate_id": candidate_id,
        "verdicts": [v for v in candidate_evidence.verdicts],
    }


def semantic_skill_coverage(requirement_skills, candidate_skills):
    if not requirement_skills: return 1.0, [], []
    if not candidate_skills: return 0.0, [], requirement_skills
    req_normalized = [normalize_skill(s) for s in requirement_skills]
    cand_normalized = [normalize_skill(s) for s in candidate_skills]
    cand_set = set(cand_normalized)
    semantic_scores, matched_skills = [], []
    for idx, req_skill in enumerate(requirement_skills):
        weight = 2.0 if idx < 8 else 1.0
        norm_req = req_normalized[idx]
        if norm_req in cand_set:
            semantic_scores.append(1.0 * weight); matched_skills.append(req_skill); continue
        req_emb = embedding_model.encode([norm_req], convert_to_tensor=True)
        cand_emb = embedding_model.encode(cand_normalized, convert_to_tensor=True)
        sims = util.cos_sim(req_emb[0], cand_emb)[0].cpu().numpy()
        best_sim = float(np.max(sims)) if len(sims) else 0.0
        floor = 0.35
        scaled_sim = max(0, (best_sim - floor) / (1 - floor))
        semantic_scores.append(scaled_sim * weight)
        if best_sim > 0.68: matched_skills.append(req_skill)
    total_weight = sum(2.0 if i < 8 else 1.0 for i in range(len(requirement_skills)))
    return float(sum(semantic_scores) / total_weight), matched_skills, [s for s in requirement_skills if s not in matched_skills]

def experience_score(jd_years, resume_years):
    if jd_years <= 0: return float(min(0.5 + (resume_years * 0.5), 1.0))
    ratio = resume_years / jd_years
    score = ratio if ratio < 1.0 else 1.0 + (min(ratio - 1.0, 0.2) * 0.5)
    return float(round(min(score, 1.0), 3))

def project_score(jd_text, project_list, model):
    if not project_list: return 0.0
    jd_vec = model.encode(jd_text, convert_to_tensor=True)
    proj_vecs = model.encode(project_list, convert_to_tensor=True)
    similarities = util.cos_sim(jd_vec, proj_vecs)[0].cpu().numpy()
    best, avg_top = float(np.max(similarities)), float(np.mean(sorted(similarities, reverse=True)[:3]))
    relevant = [s for s in similarities if s > 0.4]
    relevance_ratio = len(relevant) / len(similarities)
    return round(float(0.5 * best + 0.3 * avg_top + 0.2 * relevance_ratio), 3)

def generate_explanation(result):
    top_matched = ", ".join(result.get("matched_skills", [])) or "relevant skills"
    top_missing = ", ".join(result.get("missing_skills", []))
    explanation = f"{result['name']} scored {result['final_score']:.3f}. Demonstrates expertise in {top_matched}, aligning with the JD. "
    if result.get("experience_years"): explanation += f"Has {result['experience_years']:.1f} years of professional experience. "
    return explanation + (f"Gaps identified in: {top_missing}." if top_missing else "Closely matches all key required skills.")

def process_resumes(jd_file_path, resume_file_paths, user_weights=None):
    jd_raw = extract_text(jd_file_path)
    jd_text = fix_numbers(clean_text(jd_raw))
    jd_struct = validate_extracted_struct(call_llm_extraction(jd_text, is_jd=True), jd_text, is_jd=True)
    jd_reqs = parse_jd_requirements(jd_text, jd_struct, user_weights=user_weights)

    # Phase 6: when enabled, the per-candidate scores below are recomputed via
    # the RAG-grounded path (evidence/semantic/reranker/required_skills). When
    # disabled (the default) the exact legacy weighted-sum path is used so the
    # final scores remain byte-identical to pre-Phase-6 output.
    from rag.config import RagSettings as _RS
    rag_enabled = bool(_RS().rag_enabled)

    results = []
    for r_path in resume_file_paths:
        try:
            r_raw = extract_text(r_path)
            r_text = fix_numbers(clean_text(r_raw))
            r_struct = validate_extracted_struct(call_llm_extraction(r_text, is_jd=False), r_text, is_jd=False)
            
            # Use filename as fallback candidate name
            filename_name = re.sub(r"\b(mt|bt|cv|resume|updated|final|copy|sde)\b", " ", Path(r_path).stem, flags=re.IGNORECASE)
            filename_name = re.sub(r"[^a-zA-Z\s]", "", filename_name).strip().title()
            
            cand_name = r_struct["name"] if r_struct["name"] else filename_name

            exp_scr = experience_score(jd_reqs["experience_years"], r_struct["experience_years"])
            edu_scr = education_score(jd_reqs["education"], r_struct["education"], r_text)
            proj_scr = project_score(jd_text, r_struct["projects"], embedding_model)

            # Phase 6 RAG path: recompute four signals. ANY failure falls back
            # to the exact legacy path for THIS candidate only.
            rag_ok = False
            rag_out = None
            if rag_enabled:
                try:
                    rt = get_rag_stores()
                    rag_out = score_candidate_rag(
                        jd_text, r_text, r_struct, jd_struct, r_path,
                        rt["ontology_store"], rt["resume_store"],
                        rt["embedder"], rt["reranker"], rt["llm_client"],
                    )
                    rag_ok = True
                except Exception as e:
                    print(f"RAG scoring failed for {cand_name}; falling back to legacy path. Error: {e}")

            if rag_ok:
                req_score = rag_out["required_skills"]
                matched = rag_out["matched"]
                missing = rag_out["missing"]
                sem_scr = rag_out["semantic"]
                rerank_scr = rag_out["reranker"]
                ev_scr = rag_out["evidence"]
                weights = rag_out["weights"]
                scoring_version = "v1-rag"
                skill_verdicts = rag_out["verdicts"]
            else:
                req_score, matched, missing = semantic_skill_coverage(jd_reqs["required_skills"], r_struct["skills"])
                sem_scr = compute_semantic_similarity(jd_text, r_text)
                
                # Optimized CrossEncoder Usage (Threshold-based)
                # Only run expensive CrossEncoder if semantic similarity or req_score shows promise
                if sem_scr > 0.35 or req_score > 0.4:
                    rerank_scr = compute_reranker_score(jd_text, r_text)
                else:
                    rerank_scr = sem_scr * 0.6  # Give a scaled down score without doing expensive compute
                    
                ev_scr = compute_evidence_score(matched, r_text, r_struct["projects"])
                weights = jd_reqs["weights"]
                scoring_version = "v1-legacy"
                skill_verdicts = None

            final = sum([
                weights.get('required_skills', 0) * req_score,
                weights.get('semantic', 0) * sem_scr,
                weights.get('reranker', 0) * rerank_scr,
                weights.get('experience', 0) * exp_scr,
                weights.get('education', 0) * edu_scr,
                weights.get('projects', 0) * proj_scr,
                weights.get('evidence', 0) * ev_scr
            ])

            res = {
                "name": cand_name,
                "final_score": round(float(final), 4),
                "matched_skills": matched,
                "missing_skills": missing,
                "experience_years": r_struct["experience_years"],
                "required_skill_score": round(req_score, 2),
                "semantic_score": round(sem_scr, 2),
                "reranker_score": round(rerank_scr, 2),
                "education_score": round(edu_scr, 2),
                "project_score": round(proj_scr, 2),
                "semantic_summary": r_struct.get("semantic_summary", ""),
                "experience_breakdown": r_struct.get("experience_breakdown", []),
                "projects": r_struct.get("projects", []),
                "education": r_struct.get("education", []),
                "scoring_version": scoring_version,
            }
            if skill_verdicts is not None:
                res["skill_verdicts"] = [
                    {
                        "skill_name": v.skill_name,
                        "verdict": v.verdict,
                        "cited_chunk_ids": v.cited_chunk_ids,
                        "quote": v.quote,
                    }
                    for v in skill_verdicts
                ]
            res["explanation"] = generate_explanation(res)
            results.append(res)
        except Exception as e:
            print(f"Error processing {r_path}: {e}")

    return sorted(results, key=lambda x: x["final_score"], reverse=True)
