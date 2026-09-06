# Phase 9 — Evaluation & Calibration Results

**Scope:** NetApp MTS Intern JD (`JD/JD.pdf`) against 7 IIIT-B candidates (`Resumes/`).
**Gold set:** `App/rag/eval/data/gold_set_netapp_mts_intern_v1.json` — 23 technical requirements (R01–R23) x 7 candidates = **161 labels**
(demonstrated=68, claimed_only=38, absent=55). Hand ranking recorded with adjacent-pair ambiguity notes.
**Environment:** OPENROUTER_API_KEY **unset** ⇒ every LLM call fails fast; ALL results below are the honest **LLM-less** system
(`NormalizedRequirement.source="keyword_fallback"`), matching what the code produces in this repo's environment.
**Reproduce:**
```
cd App
pytest rag/tests/test_eval_metrics.py        # 17 fast, hermetic eval tests
python3 -m rag.eval.run_eval                 # <- real models; writes rag/eval/results/run_eval_v1.json
python3 -m rag.eval.calibrate                # threshold sweep off saved scores (read-only)
```

---

## 1. Metrics table

| Metric | Value | Notes |
|---|---|---|
| Retrieval **recall@3** (hybrid+CE) | **0.882** (60/68) | gold demonstrated pairs; golden retrieved chunk is in the top-3 |
| Verdict **accuracy** (161 cells) | **0.416** (67/161) | dominated by the 55/55 absent cells |
| Accuracy per gold-verdict | demonstrated **0.162**, claimed_only **0.026**, absent **1.000** | see §3 |
| **False-missing skill rate** | **0.838** (57/68 gold demonstrated missed) | the LLM-less bottleneck |
| **Citation-validity rate** | **None** | no LLM ⇒ zero proposed citations; undefined (reported `None`, not 0.0) |
| **Kendall tau** (evidence-signal vs hand) | **1.0** | *{n=7; illustrative only, not a statistically meaningful ranking benchmark}* |
| Index build wall | 11.5 s | all-mpnet-base-v2, 55 chunks indexed into InMemoryStore |
| Verdict wall (per candidate) | 16.1 s cold / ~3.2 s warm | retrieval + real cross-encoder + deterministic fallback |
| LLM cost (this run) | **$0.00** | fallback-only |
| Resume coverage | Phase 2 numbers CITED — **not remeasured** (0.957–0.980/candidate; roadmap's 53.5–93.9% is a *different* embedding-reachability metric, not reproducible here). |

## 2. Six-row retrieval ablation (recall@3 over 68 gold demonstrated pairs, real models)

| Config | recall@3 | Note |
|---|---|---|
| dense_only | 0.632 | all-mpnet dense |
| bm25_only | 0.618 | real BM25 (InMemoryStore) |
| hybrid | 0.809 | RRF dense+sparse (+skills down-weight) |
| hybrid_ce | 0.882 | +ms-marco cross-encoder rerank (n=3) |
| hybrid_ce+ontology_expansion | 0.882 | multi-query alt labels; **0.000 gain** with the synthetic ESCO stand-in |
| hybrid_ce+header_prefix | **GAP** | `prefix_context()` is **not wired** into chunking/indexing — no chunk is embedded with a header today; measuring this row would fabricate numbers |

Ceiling note: the gold demonstrated chunk is already found by the retrieval layer (88%); the gap is downstream in verdicting (§3).

## 3. Three-tier verdict accuracy — LLM-less mode collapses the tiers

Verdicts come from `EvidenceVerifier._deterministic_verdicts` (CE-score fallback, thresholds 0.6/0.3) because the batched LLM verdict call fails without an API key. Tiers (a)/(b)/(c) — LLM vs keyword vs split-sections — **cannot be separated at runtime here**; every requirement is `keyword_fallback`. Unit tests still exercise all three source paths hermetically (`NormalizedRequirement.source` = `llm_classifier` / `keyword_fallback`), including reuse of the real `split_jd_sections` on fallback (ontology.py:177-201).

Measured contributions to the 0.416 accuracy:
- 13/23 gold rows have **no extracted skill** (only 11 skills are extracted from this JD) ⇒ system predicts `absent` for those 91 cells by construction → absent accuracy 55/55 = 1.000, but those rows are unjudgeable, not correct.
- demonstrated recall 11/68 = 0.162: retrieval finds the right chunk 88% of the time, but best CE score rarely crosses the 0.6 "demonstrated" bar in LLM-less mode.

## 4. Constant re-validation (constraint: change only with a cited gold-set number)

| Constant | File | Old | New | Verdict & cited measurement |
|---|---|---|---|---|
| `CE_RELEVANCE_FLOOR` | rag/retrieval.py:41 | 0.5 | **0.5** (unchanged) | hybrid_ce recall@3 = **0.882**; floor isn't strangling retrieval. |
| Squash temperature | rag/retrieval.py:189 | 1.5 | **1.5** (unchanged) | operating point delivers recall@3 0.882; no gold-set demand to move it. |
| `CE_FALLBACK_DEMONSTRATED` | rag/evidence.py:24 | 0.6 | **0.6** (unchanged) | sweep: 0.5 lifts demonstrated-recall 0.162→0.294 & accuracy 0.416→0.466, but collapses demonstrated/claimed into one tier and is a one-JD/7-cand fit. **Not robust enough to demand a change** (n=7 caveat). |
| `CE_FALLBACK_CLAIMED` | rag/evidence.py:25 | 0.3 | **0.3** (unchanged) | structurally **unreachable** in the fallback path: every post-floor span ≥ 0.5 > 0.3, so the claim threshold never binds. Keeping it is harmless; document only. |
| `ONTOLOGY_MATCH_FLOOR` | rag/ontology.py:44 | 0.3 | **0.3** (unchanged) | synthetic stand-in **spuriously** maps `c++`→Linux (cos ≥ 0.3) → unix/bash/shell alt-labels → all 7 candidates "demonstrated" for C++ vs gold 1; stand-in is not the real taxonomy ⇒ **cannot** re-validate until real ESCO data is supplied; flagged. |
| `_RAG_BASE_WEIGHTS` 0.20/0.25 | pipeline.py:369 | — | unchanged | final_score needs tiered (a)/(b)/(c) requirements; **not exercisable in LLM-less mode**. Partial check: evidence-signal ranking reproduces hand ranking exactly (τ=1.0, n=7). |
| `PREFERRED_TIER_WEIGHT_RATIO` 1/3 | pipeline.py:355 | — | unchanged | same reason as above. |

Testable infra: the sweep (`python3 -m rag.eval.calibrate`) is read-only — it re-derives verdicts from the **saved** per-(skill,candidate) cross-encoder scores in `results/run_eval_v1.json`.

## 5. Findings

1. **Retrieval is strong, verdicting is the bottleneck.** hybrid+CE recall@3 = 0.882 yet LLM-less demonstrated-recall = 0.162. Data reaches the top-3; CE-fallback would-be inferences under-cut it.
2. **Ontology expansion did not help** recall@3 (0.882 → 0.882) with the synthetic 15-concept stand-in — and it *hurt* the verdict path (c++ false "demonstrated" everywhere, §4). Expansion must be re-measured against real ESCO before any ingestion wiring.
3. **13 of 23 gold rows are unjudgeable by the LLM-less system** (no extracted skill ⇒ technique-based "absent"). This is the dominant term of every accuracy figure and would flip once the LLM extraction tier runs (needs OPENROUTER_API_KEY).
4. **header-prefix ablation row is a real gap**, not a number: `prefix_context` is reserved, never called by `chunk()`/`ingest()` (chunking.py:264-266).

## 6. Known limitations (all numbers above)

- **n=7, one JD** — ranking tau is illustrative; pair-level metrics (recall, accuracy) are reportable but single-domain.
- Real ESCO skills dataset absent ⇒ synthetic stand-in; ontology & expansion numbers are stand-in-limited.
- LLM off ⇒ tiers collapse to keyword path and citation-validity is undefined; three-tier accuracy and LLM citation numbers are **not** measured here.
- `run_eval_v1.json` holds the full per-skill best-CE matrix for reproducible off-line calibration.