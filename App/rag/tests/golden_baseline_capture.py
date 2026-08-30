import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline import process_resumes
from rag.config import RagSettings

print("rag_enabled =", RagSettings().rag_enabled)
assert RagSettings().rag_enabled is False, "golden baseline must run with rag DISABLED"

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "..")
RESUME_BASE = os.path.join(BASE, "Resumes")
JD_BASE = os.path.join(BASE, "JD")
jd_path = os.path.join(JD_BASE, "JD.pdf")
resume_dir = RESUME_BASE
resumes = sorted(
    os.path.join(resume_dir, f) for f in os.listdir(resume_dir)
    if f.lower().endswith((".pdf", ".docx"))
)

results = process_resumes(jd_path, resumes)

out = {
    "jd_path": jd_path,
    "resumes": [os.path.basename(r) for r in resumes],
    "results": results,
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_baseline.json"), "w") as fh:
    json.dump(out, fh, indent=2)

print("=== GOLDEN BASELINE (pre-change) ===")
for r in results:
    print("%-28s final=%.4f req=%.2f sem=%.2f rer=%.2f edu=%.2f proj=%.2f" % (
        r["name"], r["final_score"], r["required_skill_score"],
        r["semantic_score"], r["reranker_score"],
        r["education_score"], r["project_score"],
    ))
print("Saved golden_baseline.json")
