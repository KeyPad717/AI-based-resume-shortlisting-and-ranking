// Score-colour thresholds for the RAG view.
//
// PROVISIONAL: no real RAG-scored gold set exists yet (calibration is a Phase 9
// task), so these are deliberately kept equal to the legacy 0.75/0.5 split for
// now. They are a SEPARATE constant set from the legacy view's getScoreClass so
// the legacy flow is untouched and the two can diverge independently after real
// data lands. Do not silently retune without that gold set.

export const RAG_SCORE_THRESHOLDS = { high: 0.75, medium: 0.5 }

export function ragScoreClass(score) {
  if (score >= RAG_SCORE_THRESHOLDS.high) return 'rag-score-high'
  if (score >= RAG_SCORE_THRESHOLDS.medium) return 'rag-score-medium'
  return 'rag-score-low'
}
