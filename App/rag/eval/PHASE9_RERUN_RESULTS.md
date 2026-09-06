# Phase 9 Rerun — LLM-Backed Evaluation Results

**Scope:** Same harness and gold set as Phase 9 (`JD/JD.pdf` × 7 IIIT-B candidates, 161 labels:
demonstrated=68, claimed_only=38, absent=55), now run with a **real OPENROUTER_API_KEY**.
**Environment:** `OPENROUTER_API_KEY` set ⇒ real gpt-4o-mini classification (`SkillNormalizer` /
Phase 0 `classify_requirement_type`) and real batched LLM evidence verdicts + citations.
Retrieval, chunking, embedding and the six-row ablation are **unchanged code paths**.
**Companion record:** `PHASE9_RESULTS.md` (LLM-less baseline) and `results/run_eval_v1.json`
are historical and **untouched**; this document records `results/run_eval_v2.json`.
**Reproduce:**
```
cd App
pytest rag/tests/test_eval_metrics.py        # 17 fast, hermetic eval tests (unchanged, 17/17 pass)
python3 -m rag.eval.run_eval                 # writes v2 when OPENROUTER_API_KEY is set (v1 when unset / RAG_EVAL_LLM=0)
```

---

## 0. Genuine bug found & fixed during this run (reported, not silent)

`rag/evidence.py` `EvidenceVerifier._call_llm_batch` rejected the real model's output shape. The
evidence prompt (versioned `v1`) instructs the model to return a **bare JSON array**, but
`response_format={"type":"json_object"}` forces a top-level **object**, so gpt-4o-mini wraps the
list under a named key — observed wrappers include `{"skills": [...]}` and `{"result": [...]}`.
The parser only accepted a bare array or `{"verdicts": [...]}`, so `parsed` was `[]` ⇒ the method
returned `None` ⇒ **every** evidence call silently degraded to the deterministic CE-score fallback
even with a valid API key. Raw example captured during the smoke test:
```json
{"result": [{"skill_name":"c","verdict":"claimed_only",
             "cited_chunk_ids":["7abda333:05"],"quote":"Languages: C, C++, SQL, HTML, CSS, JavaScript"}]}
```
Fix (evidence.py:195-203): accept any non-empty list-valued field of the response object. This was
a production-path bug (Phase 9 could not observe it only because no LLM call was ever attempted).
Without it, this rerun would have been byte-identical to the LLM-less baseline — exactly the false
"LLM-backed" result constraint 4 warns against. Existing tests unaffected: `test_evidence.py` +
`test_eval_metrics.py` = 24 passed. No prompt, no constant, no protected function changed.

Also fixed a reporting-only bug in `run_eval.py`: `source_distribution` counted the unique-source
set instead of the 11 requirements (`{'llm_classifier': 1}` → corrected to `{'llm_classifier': 11}`).

## 1. Metrics table (LLM-backed)

| Metric | Phase 9 (LLM-less) | **Rerun (LLM-backed)** | Notes |
|---|---|---|---|
| Retrieval **recall@3** (hybrid+CE) | 0.882 (60/68) | **0.882** (60/68) | retrieval is LLM-independent; unchanged, as expected |
| Verdict **accuracy** (161 cells) | 0.416 (67/161) | **0.453** (73/161) | +6 correct cells |
| Accuracy per gold-verdict | demo 0.162 / claimed 0.026 / absent 1.000 | **demo 0.118 / claimed 0.263 / absent 1.000** | real LLM fixes 9 overclaims, loses 5 demonstrated |
| **False-missing skill rate** | 0.838 (57/68) | **0.882** (60/68) | LLM is more conservative on demonstrated than the CE>0.6 bar |
| **Citation-validity rate** | None (undefined) | **0.967** (proposed 30, rejected 1, accepted 29) | first real, non-None citation number |
| Evidence LLM fallback triggers | 7/7 (by construction) | **0/7** (0 of 10 LLM calls failed; http_errors=0) | CE-fallback is now a dormant safety net |
| **Kendall tau** (evidence-signal vs hand) | 1.0 | **1.0** | *{n=7; illustrative only, not a statistically meaningful ranking benchmark}* |
| Index build wall | 11.5 s | **10.9 s** | same embedder, same 55 chunks |
| Verdict wall (per candidate) | 16.1 s cold / ~3.2 s warm | **18.2 s cold / ~3.8–6.9 s warm** | real batched LLM call per candidate added |
| **LLM cost (this run)** | $0.00 | **$0.0024** | 10 OpenRouter calls, 8,796 tokens; measured credit delta (see §6) |

## 2. Six-row retrieval ablation (recall@3 over 68 gold demonstrated pairs)

| Config | Phase 9 | **Rerun** | Note |
|---|---|---|---|
| dense_only | 0.632 | **0.632** | identical (identical code path) |
| bm25_only | 0.618 | **0.618** | identical |
| hybrid | 0.809 | **0.809** | identical |
| hybrid_ce | 0.882 | **0.882** | identical |
| hybrid_ce+ontology_expansion | 0.882 | **0.882** | identical — 0.000 gain with synthetic stand-in |
| hybrid_ce+header_prefix | **GAP** | **GAP** | `prefix_context()` still not wired into chunking — row left as a gap, not fabricated |

`run_eval_v1.json`['retrieval_ablation'] == `run_eval_v2.json`['retrieval_ablation'] element-for-element.

## 3. Three-tier comparison — now real

Verdicts come from the real batched LLM (`openai/gpt-4o-mini`); the deterministic CE fallback did
not fire once. Verdict accuracy is **one** number (**0.453**) because the evidence layer keys on
`skill_name` only — `NormalizedRequirement.tier` is consulted nowhere in retrieval/evidence
(verified by code inspection), so tiers a/b/c do not change per-skill verdicts. The three
classifiers therefore differ in what they are *for*: required/preferred classification.

**Classification sanity vs the JD's explicit optionality** (reference: R01–R05 "Good programming
skills" + R08 "Exposure to JS/TS" + R11 "Familiarity with Linux" = required; R15/R16 Docker/K8s
"preferred, not required"; R22 ML "is a plus"):

| Tier | Method | LLM? | required/preferred map | accuracy vs JD reference |
|---|---|---|---|---|
| (a) | `split_jd_sections` keyword window | no | all 11 → required (docker, kubernetes, machine learning mislabelled) | **0.727** (8/11) |
| (b) | Phase 0 `classify_requirement_type` | yes | docker/k8s/ML → preferred; **typescript → preferred** | **0.909** (10/11) |
| (c) | Full `SkillNormalizer` (canonicalize + LLM) | yes | docker/k8s/ML → preferred; typescript → **required** | **1.000** (11/11) |

Pairwise agreement: a≈b 0.636 · a≈c 0.727 · b≈c 0.909. `classify_requirement_type` succeeded
(`b_fell_back_to_keywords = false`); full `SkillNormalizer` classified all 11 requirements with
`source = llm_classifier` (0 keyword_fallback). **Both LLM tiers agree with the JD's own
optionality phrasing and beat the keyword window — Phase 0's classifier demonstrably works.**

## 4. Constant re-evaluation (evaluate with real data; **no constant changed**)

| Constant | Verdict & cited measurement (LLM-backed) |
|---|---|
| `CE_FALLBACK_DEMONSTRATED` / `CE_FALLBACK_CLAIMED` (0.6 / 0.3) | **No longer load-bearing.** The CE fallback triggered **0/7** candidates and 0 of 10 LLM calls failed (http_errors=0). Real LLM verdicts are primary; the fallback is now a rare safety net, so these thresholds did not influence a single verdict in this run. Recommendation: **leave unchanged**; revisit only if fallback-trigger frequency rises in production. |
| Three-tier comparison (Phase 0 classifier) | **Validated.** `classify_requirement_type` = 0.909 vs the JD reference and, critically, flags exactly the JD's "preferred, not required" set (docker, kubernetes, machine learning). It is sane; no change indicated. |
| `ONTOLOGY_MATCH_FLOOR` (0.3) | Still **not re-validable**: the c++→Linux spurious mapping (cos ≥ 0.3) persists, but it is a **synthetic-stand-in data artifact**, not a floor problem — and with real classification the tier output was correct anyway (c++ → required under all three tiers despite the wrong canonical). Recommendation: **keep 0.3** pending a real ESCO CSV; kill the collision by fixing the data, not the floor. |
| `CE_RELEVANCE_FLOOR`, squash τ, `_RAG_BASE_WEIGHTS`, `PREFERRED_TIER_WEIGHT_RATIO` | No gold-set demand arose. Retrieval unchanged (0.882); final-score weights still not exercisable end-to-end (tier-aware final scoring remains a follow-up). Unchanged. |

## 5. What changed vs the Phase 9 baseline (every headline number side by side)

| Number | Phase 9 (LLM-less) | Rerun (LLM-backed) | Attribute |
|---|---|---|---|
| Verdict accuracy | 0.416 (67/161) | **0.453** (73/161) | real LLM verdicts |
| demonstrated recall | 0.162 | 0.118 | LLM more conservative than CE>0.6 bar |
| claimed_only recall | 0.026 | 0.263 | LLM correctly reclassifies overconfident "demonstrated" → claims |
| absent recall | 1.000 | 1.000 | unchanged (unjudgeable rows dominate both) |
| False-missing rate | 0.838 (57/68) | 0.882 (60/68) | 5 gold-demonstrated downgraded to claimed_only |
| Citation-validity rate | None (undefined) | **0.967** (30 proposed, 1 rejected) | previously unmeasurable |
| Three-tier comparison | collapsed (all keyword_fallback) | a 0.727 / b 0.909 / c 1.000 vs JD ref | real LLM classify live |
| Evidence LLM fallbacks | 7/7 (by construction) | **0/7** | CE-fallback dormant |
| Kendall tau | 1.0 | 1.0 | identical predicted ranking (n=7 caveat applies) |
| Ablation recall@3 | 0.632/0.618/0.809/0.882/0.882/GAP | identical | retrieval is LLM-independent — confirmed, not assumed |
| LLM cost | $0.00 | **$0.0024** | 10 calls, 8,796 tokens |
| LLM extraction (diagnostic) | not run | **20 skills** (vs 11 rule-based) | would make ~8 of the 13 no-alias rows judgeable → follow-up (gold-alias change), not applied |

Migration of the 68 gold-demonstrated cells v1→v2: 48 absent→absent (retrieval miss, unchanged),
7 claimed→claimed, 6 demonstrated→demonstrated (kept right), **5 demonstrated→claimed_only
(lost)**, **2 claimed_only→demonstrated (gained)**. Of the 38 gold-claimed_only cells:
28 absent→absent, **9 demonstrated→claimed_only (fixed)**, 1 claimed→claimed. Direction is
consistent: the real LLM is a materially better claimed-vs-demonstrated discriminator than the
CE floor, at the cost of being conservative on weak demonstrated spans.

## 6. Cost

Real OpenRouter credits balance (`GET /api/v1/credits`): pre-run `0.09796601` → post-run
`0.10034711`, delta **$0.002381 ≈ $0.0024** for the full run (10 calls, prompt 6,677 / completion
2,119 / total 8,796 tokens, 31.7 s LLM wall, 0 HTTP errors). Token-derived estimate at gpt-4o-mini
list pricing (0.15/1M prompt, 0.60/1M completion) ≈ $0.0023 — consistent. The pre-flight smoke
tests (classify + extraction + 3 evidence calls) consumed an additional small amount (not
isolated; key is free-tier, and the whole task spent well under one cent in total).
This is the project's first real LLM cost number (Phase 9: $0.00).

## 7. Known limitations

- **n=7, one JD** — tau is illustrative; verdict/citation aggregates are single-domain.
- Real ESCO skills dataset absent ⇒ ontology/expansion numbers remain stand-in-limited.
- The 11-skill aligned grid keeps rule-based extraction so the gold alias map stays valid; the
  real LLM extraction diagnostic (20 skills) shows the 13 no-alias rows could shrink to ~5, but
  switching to it needs a gold-set/alias change → explicitly a follow-up, not this run.
- LLM verdicts are stochastic (temperature ≠ 0): a re-run may move a few marginal cells (the
  smoke's `c` verdict flipped claimed_only→demonstrated across calls); headline directions in §5
  are stable, individual cells are not.
- `run_eval_v2.json` records the full verdict grid, per-skill best-CE matrix, three-tier maps and
  LLM usage for reproducible follow-up analysis.