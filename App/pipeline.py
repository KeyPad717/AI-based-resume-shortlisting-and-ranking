import numpy as np
import os
import re
import json
import docx
import spacy
import pdfplumber
import openai
import math
from pathlib import Path

from sentence_transformers import SentenceTransformer, CrossEncoder, util

print("Loading NLP Models...")
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

try:
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("abbreviation_detector")
except OSError:
    print("Falling back to regular spaCy")
    nlp = spacy.load("en_core_web_sm")

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "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"),
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

def compute_evidence_score(matched_skills, exp_text, projects):
    if not matched_skills: return 0.0
    text = (str(exp_text) * 2 + " " + " ".join(map(str, projects))).lower()
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
        except: pass
            
    name = ""
    if not is_jd:
        doc = nlp(text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text
                break
                
    return {
        "candidate_name": name,
        "skills": list(set(found_skills)),
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

def validate_experience(exp):
    if exp > 10: return 5.0
    if exp < 0: return 0.0
    return exp

def compute_experience_years(experience_list, resume_text, is_jd=False):
    total = sum(float(e.get("duration_years", 0)) for e in experience_list if isinstance(e, dict))
    return validate_experience(total)

def validate_extracted_struct(struct, resume_text, is_jd=False):
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

def normalize_weights(weights):
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()} if total > 0 else weights

def parse_jd_requirements(jd_text, jd_struct, user_weights=None):
    sections = split_jd_sections(jd_text)
    all_skills = jd_struct["skills"]
    req_text_lower = sections["required_text"].lower()
    pref_text_lower = sections["preferred_text"].lower()
    required_skills = [s for s in all_skills if s.lower() in req_text_lower] or all_skills[:10]
    preferred_skills = [s for s in all_skills if s.lower() in pref_text_lower and s.lower() not in {r.lower() for r in required_skills}]

    if user_weights:
        # Use user weights but ensure they are float 0-1
        weights = {k: float(v) / 100.0 for k, v in user_weights.items()}
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
    return float(round(min(score, 1.2), 3))

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

    results = []
    for r_path in resume_file_paths:
        try:
            r_raw = extract_text(r_path)
            r_text = fix_numbers(clean_text(r_raw))
            r_struct = validate_extracted_struct(call_llm_extraction(r_text, is_jd=False), r_text, is_jd=False)
            
            # Use filename as fallback candidate name
            filename_name = re.sub(r"\\b(mt|bt|cv|resume|updated|final|copy|sde)\\b", " ", Path(r_path).stem, flags=re.IGNORECASE)
            filename_name = re.sub(r"[^a-zA-Z\\s]", "", filename_name).strip().title()
            
            cand_name = r_struct["name"] if r_struct["name"] else filename_name

            req_score, matched, missing = semantic_skill_coverage(jd_reqs["required_skills"], r_struct["skills"])
            exp_scr = experience_score(jd_reqs["experience_years"], r_struct["experience_years"])
            edu_scr = education_score(jd_reqs["education"], r_struct["education"], r_text)
            proj_scr = project_score(jd_text, r_struct["projects"], embedding_model)
            sem_scr = compute_semantic_similarity(jd_text, r_text)
            
            # Optimized CrossEncoder Usage (Threshold-based)
            # Only run expensive CrossEncoder if semantic similarity or req_score shows promise
            if sem_scr > 0.35 or req_score > 0.4:
                rerank_scr = compute_reranker_score(jd_text, r_text)
            else:
                rerank_scr = sem_scr * 0.6  # Give a scaled down score without doing expensive compute
                
            ev_scr = compute_evidence_score(matched, r_struct["experience_breakdown"], r_struct["projects"])

            weights = jd_reqs["weights"]
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
                "education": r_struct.get("education", [])
            }
            res["explanation"] = generate_explanation(res)
            results.append(res)
        except Exception as e:
            print(f"Error processing {r_path}: {e}")

    return sorted(results, key=lambda x: x["final_score"], reverse=True)
