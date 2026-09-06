"""Phase 9: evaluation metrics over the NetApp gold set.

All functions here are PURE (no models / no I/O except the gold-set loader) so
they are unit-testable with hand-constructed fixtures. The heavy lifting (real
retrieval, real verdicts) lives in ``run_eval.py``; this module turns raw run
output into the reportable numbers plus their mandatory caveats.

Ranking-level results MUST carry ``KENDALL_CAVEAT``; per-pair results over the
161 labels do not (that is where the metric has real signal).
"""

from __future__ import annotations

from .gold_set import KENDALL_CAVEAT

# --------------------------------------------------------------------------- #
# Resume coverage — CITED from Phase 2, deliberately NOT remeasured here.
# --------------------------------------------------------------------------- #
# Phase 2 measured "extraction coverage" = chunked_chars / extracted_chars per
# resume with the v1 chunker (docs/phase-summaries/phase-2-summary.md). The
# roadmap's 53.5-93.9% is the earlier embedding-model reachability figure from
# the retrieval phases; it is referenced, not reproduced.
RESUME_COVERAGE_BASELINE = {
    # slug  -> (chunked/extracted coverage from Phase 2, chunk_count)
    "jainish_parmar": (0.965, 8),
    "aditya_dave": (0.980, 7),
    "gaurav_rajpurohit": (0.974, 8),
    "kartavya_patel": (0.957, 7),
    "keyur_padiya": (0.966, 8),
    "mayank_satapara": (0.977, 9),
    "parva_parmar": (0.959, 8),
}
ROADMAP_EMBEDDING_COVERAGE_RANGE = (0.535, 0.939)


def resume_coverage_report() -> dict:
    """Return the cited Phase 2 coverage numbers (NOT a new measurement)."""
    return {
        "metric": "resume_coverage = chunked_chars / extracted_chars (Phase 2 measurement)",
        "per_candidate": {k: v[0] for k, v in RESUME_COVERAGE_BASELINE.items()},
        "roadmap_embedding_model_coverage": ROADMAP_EMBEDDING_COVERAGE_RANGE,
        "note": "Cited from Phase 2 (docs/phase-summaries/phase-2-summary.md); "
        "not remeasured in Phase 9. The roadmap's 53.5-93.9% range refers to "
        "embedding-model-reachable chunks from the retrieval phases, a different "
        "metric than chunked/extracted coverage.",
    }


# --------------------------------------------------------------------------- #
# Retrieval recall@k over the gold pairs.
# --------------------------------------------------------------------------- #
def retrieval_recall_at_k(gold_pairs: list[dict], top3_by_pair: dict, k: int = 3) -> dict:
    """Fraction of demonstrated gold pairs whose gold_chunk_id is in the top-k.

    ``gold_pairs``: list of dicts from ``GoldSet.gold_pairs()``.
    ``top3_by_pair``: {(requirement_id, candidate_id): [chunk_id, ...]} as
    returned by the retrieval ablation runner.
    """
    hits = total = 0
    for pair in gold_pairs:
        key = (pair["requirement_id"], pair["candidate_id"])
        top = top3_by_pair.get(key) or []
        total += 1
        if pair["gold_chunk_id"] in top[:k]:
            hits += 1
    return {
        "k": k,
        "recall@%d" % k: (hits / total) if total else 0.0,
        "pairs": total,
        "hit": hits,
    }


# --------------------------------------------------------------------------- #
# Citation validity — the automatic quote-in-chunk check.
# --------------------------------------------------------------------------- #
def citation_validity_rate(proposed: int, rejected: int) -> dict:
    """Fraction of LLM-proposed citations accepted by CitationValidator.

    ``proposed`` = total citations the LLM proposed across all pairs;
    ``rejected`` = how many CitationValidator downgraded. If proposals == 0 the
    rate is undefined and reported as None (never 0.0, which would falsely claim
    an unexercised check passes 0%).
    """
    if proposed <= 0:
        return {
            "proposed": 0,
            "accepted": 0,
            "citation_validity_rate": None,
            "note": "No LLM-proposed citations to validate (proposal stage never ran).",
        }
    accepted = proposed - rejected
    return {
        "proposed": proposed,
        "rejected": rejected,
        "accepted": accepted,
        "citation_validity_rate": accepted / proposed,
    }


# --------------------------------------------------------------------------- #
# Verdict accuracy — the three classifier tiers.
# --------------------------------------------------------------------------- #
def aggregate_alias_verdicts(system_by_skill: dict, alias_map: dict[str, list[str]]) -> dict:
    """Collapse per-skill verdicts onto the 23 gold requirement rows.

    ``system_by_skill``: {skill_name: {candidate_slug: verdict}}.
    ``alias_map``: gold_set.system_skill_alias (requirement_id -> [skill, ...]).
    A requirement with no aliased skill is predicted "absent" (the LLM-less
    system has no skill to key a verdict on — this IS the measured behavior).
    With several aliases, the most confident non-absent verdict wins
    (demonstrated > claimed_only > absent).
    """
    grid = {}
    for rid, aliases in alias_map.items():
        per_candidate = {}
        for slug in sorted({s for cand in system_by_skill.values() for s in cand}):
            votes = [system_by_skill.get(a, {}).get(slug, "absent") for a in aliases]
            order = {"demonstrated": 2, "claimed_only": 1, "absent": 0}
            per_candidate[slug] = max(votes, key=lambda v: order[v]) if votes else "absent"
        grid[rid] = per_candidate
    return grid


def _gold_grid(gold_set) -> dict[str, dict[str, str]]:
    return {
        L["r"]: {L["c"]: L["v"]}
        for L in gold_set.labels
    }


def verdict_accuracy(gold: dict, predicted: dict) -> dict:
    """Accuracy of the predicted verdict grid vs the gold grid, over all 161 pairs.

    ``gold``/``predicted``: {requirement_id: {candidate_slug: verdict}}.
    """
    correct = total = 0
    by_verdict = {"demonstrated": [0, 0], "claimed_only": [0, 0], "absent": [0, 0]}
    for rid, cand_map in gold.items():
        for slug, gold_v in cand_map.items():
            pred_v = predicted.get(rid, {}).get(slug, "absent")
            ok = pred_v == gold_v
            total += 1
            correct += ok
            by_verdict[gold_v][0] += ok
            by_verdict[gold_v][1] += 1
    per_verdict = {
        v: (by_verdict[v][0] / by_verdict[v][1] if by_verdict[v][1] else None)
        for v in by_verdict
    }
    return {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "per_gold_verdict": per_verdict,
    }


def false_missing_skill_rate(gold: dict, predicted: dict) -> dict:
    """Fraction of gold-DEMONSTRATED pairs the system does NOT call demonstrated.

    This is the number Findings 3 & 4 predicted would be high: demonstrated
    skills the system misses (mostly skills it never extracted from the JD).
    """
    gold_demo = total = 0
    per_candidate: dict[str, list] = {}
    for rid, cand_map in gold.items():
        for slug, gold_v in cand_map.items():
            if gold_v != "demonstrated":
                continue
            total += 1
            pred_v = predicted.get(rid, {}).get(slug, "absent")
            if pred_v == "demonstrated":
                gold_demo += 1
            else:
                per_candidate.setdefault(slug, []).append((rid, pred_v))
    by_slug = {
        slug: 1.0 - len(per_candidate.get(slug, [])) / max(1, sum(
            1 for r, m in gold.items() for s, v in m.items()
            if v == "demonstrated" and s == slug
        ))
        for slug in {s for r, m in gold.items() for s in m}
    }
    return {
        "false_missing_skill_rate": 1.0 - (gold_demo / total) if total else None,
        "gold_demonstrated": total,
        "predicted_demonstrated": gold_demo,
        "missed": total - gold_demo,
        "by_candidate": {s: (total - len(v), len(v)) for s, v in per_candidate.items()},
    }


# --------------------------------------------------------------------------- #
# Kendall tau — ranking-level, MUST be caveated.
# --------------------------------------------------------------------------- #
def kendall_tau(a: list[str], b: list[str]) -> dict:
    """Kendall tau between two orderings. Returns the value plus the mandatory
    sample-size caveat; NEVER report a ranking metric without the caveat."""
    pos_a = {item: i for i, item in enumerate(a)}
    pos_b = {item: i for i, item in enumerate(b)}
    items = list(pos_a.keys())
    assert set(items) == set(pos_b.keys()), "orderings must be over the same set"
    n = len(items)
    total = n * (n - 1) / 2
    concordant = sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if (pos_a[items[i]] - pos_a[items[j]]) * (pos_b[items[i]] - pos_b[items[j]]) > 0
    )
    tau = (2 * concordant - total) / total if total else 0.0
    return {
        "kendall_tau": tau,
        "n_candidates": n,
        "caveat": KENDALL_CAVEAT,
        "report": "kendall_tau=%.3f %s" % (tau, KENDALL_CAVEAT),
    }


# --------------------------------------------------------------------------- #
# Latency & cost.
# --------------------------------------------------------------------------- #
def latency_report(cold_seconds: float, warm_seconds: float, detail: dict | None = None) -> dict:
    """Cold = first candidate scored in a fresh process; warm = subsequent score."""
    return {
        "cold_seconds": round(cold_seconds, 3),
        "warm_seconds": round(warm_seconds, 3),
        "detail": detail or {},
        "cost_notes": [
            "LLM-less mode (OPENROUTER_API_KEY unset): zero LLM tokens consumed, "
            "all verdict paths are deterministic CE-score fallbacks.",
            "Cost per candidate is therefore $0.00 while the LLM-less fallback is "
            "active; with gpt-4o-mini enabled it is ~1 call per candidate.",
        ],
    }


# --------------------------------------------------------------------------- #
# Bundle helper: assemble the full metric family from a real run.
# --------------------------------------------------------------------------- #
def evaluate_run(gold_set, run_result: dict) -> dict:
    """Turn a dict of raw run outputs into the Phase 9 metric family.

    ``run_result`` keys: retrieval_top3_by_pair, verdicts_by_skill,
    citation_proposed, citation_rejected, predicted_ranking,
    cold_seconds, warm_seconds, per_config_recall (from ablation).
    """
    gold = _gold_grid(gold_set)
    gold_pairs = gold_set.gold_pairs()

    predicted_grid = aggregate_alias_verdicts(
        run_result.get("verdicts_by_skill", {}), gold_set.system_skill_alias
    )
    acc = verdict_accuracy(gold, predicted_grid)
    fms = false_missing_skill_rate(gold, predicted_grid)

    recall = {}
    if "per_config_recall" in run_result:
        recall.update(run_result["per_config_recall"])
    elif "retrieval_top3_by_pair" in run_result:
        recall = retrieval_recall_at_k(gold_pairs, run_result["retrieval_top3_by_pair"])

    tau = None
    if "predicted_ranking" in run_result and run_result["predicted_ranking"]:
        tau = kendall_tau(
            gold_set.hand_ranking, run_result["predicted_ranking"]
        )

    return {
        "sample_size_note": gold_set.meta.get("sample_size_note"),
        "resume_coverage": resume_coverage_report(),
        "retrieval_recall": recall,
        "citation_validity": citation_validity_rate(
            run_result.get("citation_proposed", 0),
            run_result.get("citation_rejected", 0),
        ),
        "verdict_accuracy": acc,
        "false_missing_skill_rate": fms,
        "kendall_tau": tau,
        "latency": latency_report(
            run_result.get("cold_seconds", 0.0),
            run_result.get("warm_seconds", 0.0),
        ),
    }


def summarize(metrics: dict) -> str:
    """Compact one-line-per-block summary used by the eval runner and report."""
    lines = []
    lines.append("resume coverage (Phase 2, cited): %s" % metrics["resume_coverage"]["note"])
    rec = metrics["retrieval_recall"]
    if rec and "recall@3" in rec:
        lines.append("recall@3: %.3f (%d/%d gold spans)" % (rec["recall@3"], rec["hit"], rec["pairs"]))
    cv = metrics["citation_validity"]
    lines.append("citation validity: %s" % (cv["citation_validity_rate"] if cv else None))
    acc = metrics["verdict_accuracy"]
    lines.append("verdict accuracy: %.3f (%d/%d)" % (acc["accuracy"], acc["correct"], acc["total"]))
    fms = metrics["false_missing_skill_rate"]
    lines.append("false-missing-skill rate: %s (%d missed / %d gold demonstrated)"
                 % (fms["false_missing_skill_rate"], fms["missed"], fms["gold_demonstrated"]))
    if metrics.get("kendall_tau"):
        lines.append(metrics["kendall_tau"]["report"])
    return "\n".join(lines)