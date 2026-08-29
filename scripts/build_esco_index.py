"""Build the ESCO skills taxonomy index into the vector store.

KNOWN-LIMITATION NOTICE
----------------------
This script does NOT ship a real ESCO dataset. The official ESCO taxonomy is a
large licensed/attribution file (ESCO v1.1 skills CSVs, typically
'skills_en.csv') that must be downloaded manually and placed somewhere on disk.
To avoid hardcoding an unstable URL, this script:

  * Reads the real dataset from a user-supplied path when one is given, OR
  * Falls back to a small SYNTHETIC stand-in dataset (15 fabricated but
    realistic tech-skill concepts) so the indexing logic can be tested
    end-to-end without the real data.

EXPECTED REAL-DATA FORMAT (skills_en.csv)
-----------------------------------------
A CSV with (at minimum) a header row containing the columns:
  preferredLabel, altLabels, conceptUri, ...
where:
  preferredLabel  -> single preferred English label (e.g. "use an API")
  altLabels       -> semicolon/pipe-separated alternative labels
  conceptUri      -> unique ESCO URI (e.g. "http://data.europa.eu/esco/skill/...")

Provide it via environment variable ESCO_SKILLS_CSV, e.g.:
  ESCO_SKILLS_CSV=/path/to/skills_en.csv python scripts/build_esco_index.py

When ESCO_SKILLS_CSV is not set (or the file is missing), the script uses the
synthetic stand-in and prints a clear warning that this is NOT the real ESCO
index.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "App"))

from rag.config import RagSettings
from rag.embedding import EmbeddingService
from rag.index import OntologyIndexer
from rag.store import InMemoryStore

SYNTHETIC_CONCEPTS = [
    {
        "preferred_label": "Kubernetes",
        "alt_labels": ["K8s", "container orchestration", "kubernetes cluster management"],
        "concept_uri": "http://synthetic.example/skill/kubernetes",
    },
    {
        "preferred_label": "Docker",
        "alt_labels": ["containerization", "container runtime", "docker compose"],
        "concept_uri": "http://synthetic.example/skill/docker",
    },
    {
        "preferred_label": "continuous integration",
        "alt_labels": ["CI", "continuous delivery", "continuous deployment", "CI/CD"],
        "concept_uri": "http://synthetic.example/skill/ci",
    },
    {
        "preferred_label": "Git",
        "alt_labels": ["distributed version control", "source control", "versioning", "git workflows"],
        "concept_uri": "http://synthetic.example/skill/git",
    },
    {
        "preferred_label": "SQL",
        "alt_labels": ["structured query language", "relational databases", "SQL queries"],
        "concept_uri": "http://synthetic.example/skill/sql",
    },
    {
        "preferred_label": "MySQL",
        "alt_labels": ["relational database management", "MariaDB"],
        "concept_uri": "http://synthetic.example/skill/mysql",
    },
    {
        "preferred_label": "Python",
        "alt_labels": ["python programming", "pandas", "numpy"],
        "concept_uri": "http://synthetic.example/skill/python",
    },
    {
        "preferred_label": "Machine Learning",
        "alt_labels": ["ML", "supervised learning", "model training", "scikit-learn"],
        "concept_uri": "http://synthetic.example/skill/ml",
    },
    {
        "preferred_label": "REST APIs",
        "alt_labels": ["restful services", "web services", "API design", "HTTP endpoints"],
        "concept_uri": "http://synthetic.example/skill/rest",
    },
    {
        "preferred_label": "Linux",
        "alt_labels": ["unix", "shell scripting", "bash"],
        "concept_uri": "http://synthetic.example/skill/linux",
    },
    {
        "preferred_label": "Java",
        "alt_labels": ["java programming", "Spring", "Spring Boot"],
        "concept_uri": "http://synthetic.example/skill/java",
    },
    {
        "preferred_label": "JavaScript",
        "alt_labels": ["JS", "TypeScript", "web frontend"],
        "concept_uri": "http://synthetic.example/skill/javascript",
    },
    {
        "preferred_label": "Distributed Systems",
        "alt_labels": ["microservices", "event-driven architecture", "Kafka"],
        "concept_uri": "http://synthetic.example/skill/distributed",
    },
    {
        "preferred_label": "Database Administration",
        "alt_labels": ["database tuning", "indexing", "query optimization"],
        "concept_uri": "http://synthetic.example/skill/dba",
    },
    {
        "preferred_label": "Data Structures and Algorithms",
        "alt_labels": ["DSA", "algorithm design", "competitive programming"],
        "concept_uri": "http://synthetic.example/skill/dsa",
    },
]


def load_real_esco(csv_path):
    concepts = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preferred = row.get("preferredLabel") or row.get("preferred_label") or ""
            concept_uri = row.get("conceptUri") or row.get("concept_uri") or ""
            raw_alts = row.get("altLabels") or row.get("alt_labels") or ""
            alts = [
                a.strip()
                for a in raw_alts.replace(";", "|").split("|")
                if a.strip()
            ]
            if not preferred:
                continue
            concepts.append(
                {
                    "preferred_label": preferred,
                    "alt_labels": alts,
                    "concept_uri": concept_uri,
                }
            )
    return concepts


def main():
    settings = RagSettings()
    csv_path = os.environ.get("ESCO_SKILLS_CSV")

    if csv_path and os.path.exists(csv_path):
        concepts = load_real_esco(csv_path)
        print("Loaded %d real ESCO concepts from %s" % (len(concepts), csv_path))
    else:
        concepts = SYNTHETIC_CONCEPTS
        print(
            "WARNING: no real ESCO dataset found (ESCO_SKILLS_CSV not set or missing). "
            "Using SYNTHETIC stand-in dataset of %d concepts - this is NOT the real ESCO index."
            % len(concepts)
        )

    embed = EmbeddingService.get_instance()
    store = InMemoryStore(expected_dimension=embed.get_dimension())
    indexer = OntologyIndexer(store, embed)
    indexer.index_esco(concepts)

    collection = settings.esco_collection_name
    print("Indexed %d concepts into collection %r" % (store.count(collection), collection))


if __name__ == "__main__":
    main()
