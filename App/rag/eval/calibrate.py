"""Phase 9: read-only constant re-validation (threshold sweep).

Loads the saved real-eval results (best per-skill cross-encoder scores + gold
set) and sweeps CE_FALLBACK_DEMONSTRATED / CE_FALLBACK_CLAIMED / CE_RELEVANCE_FLOOR
choices AGAINST the 161-label gold grid, WITHOUT reloading models.

Policy (phase-9 constraint): this sweep only recommends; an actual code change is
made only when the gold-set measurement robustly demands it, and is committed in
the source with a comment citing the number.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "..", "..")
sys.path.insert(0, APP)

RESULTS_FILE = os.path.join(HERE, "results", "run_eval_v1.json")
from rag.eval.gold_set import load_gold_set
from rag.eval.metrics import aggregate_alias_verdicts


def _deterministic_verdict(best_ce, demo_thresh, claim_thresh):
    if best_ce > demo_thresh:
        return "demonstrated"
    if best_ce > claim_thresh:
        return "claimed_only"
    return "absent"


def sweep():
    gs = load_gold_set()
    data = json.load(open(RESULTS_FILE))
    ce = data["system_skill_best_ce"]  # {skill: {slug: best_ce}}
    alias = gs.system_skill_alias

    gold_grid = {
        r["id"]: {c["slug"]: gs.gold_verdict(r["id"], c["slug"]) for c in gs.candidates}
        for r in gs.requirements
    }

    # Note: best CE is measured POST CE_RELEVANCE_FLOOR (EvidenceVerifier keeps
    # spans with cross_encoder_score >= floor; empty spans record best_ce=0.0 ->
    # verdict 'absent' regardless of thresholds). So demo thresholds < floor are
    # unreachable here and are excluded from the sweep.
    floor = 0.5
    rows = []
    for demo in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        for claim in (0.20, 0.30, 0.40):
            if claim >= demo:
                continue
            system = {s: {} for s in ce}
            for s, per in ce.items():
                for slug, best in per.items():
                    system[s][slug] = _deterministic_verdict(best, demo, claim)
            pred = aggregate_alias_verdicts(system, alias)
            correct = total = 0
            demo_recall_tp = demo_gold = 0
            for rid in gold_grid:
                for slug in gold_grid[rid]:
                    pv = pred[rid][slug]
                    gv = gold_grid[rid][slug]
                    total += 1
                    correct += int(pv == gv)
                    if gv == "demonstrated":
                        demo_gold += 1
                        demo_recall_tp += int(pv == "demonstrated")
            rows.append({
                "demo_thresh": demo, "claim_thresh": claim,
                "accuracy": round(correct / total, 4),
                "demonstrated_recall": round(demo_recall_tp / demo_gold, 4),
                "demonstrated_tp": demo_recall_tp,
                "correct": correct, "total": total,
            })

    # Current code constants (the operating point).
    rows_sorted = sorted(rows, key=lambda r: (-r["accuracy"], -r["demonstrated_recall"]))
    print("=== CE-fallback threshold sweep vs 161-label gold set ===")
    print("floor(CE_RELEVANCE_FLOOR) fixed at 0.50; best CE is post-floor.")
    print("%-11s %-12s %-9s %-10s %-8s %s" % (
        "demo", "claim", "accuracy", "demoRec", "tp", "config"))
    for r in rows_sorted:
        marker = " <-- current 0.6/0.3" if (r["demo_thresh"] == 0.6 and r["claim_thresh"] == 0.3) else ""
        print("%-11.2f %-12.2f %-9.4f %-10.4f %-8d acc=%.4f%s" % (
            r["demo_thresh"], r["claim_thresh"], r["accuracy"],
            r["demonstrated_recall"], r["demonstrated_tp"], r["accuracy"], marker))
    best = rows_sorted[0]
    cur = next(r for r in rows if r["demo_thresh"] == 0.6 and r["claim_thresh"] == 0.3)
    print()
    print("current(0.6/0.3):  accuracy=%.4f  demonstrated_recall=%.4f" % (
        cur["accuracy"], cur["demonstrated_recall"]))
    print("best sweep       :  accuracy=%.4f  demonstrated_recall=%.4f  (demo=%s, claim=%s)" % (
        best["accuracy"], best["demonstrated_recall"], best["demo_thresh"], best["claim_thresh"]))
    return rows


if __name__ == "__main__":
    sweep()