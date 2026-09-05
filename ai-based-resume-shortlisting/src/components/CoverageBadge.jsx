import './CoverageBadge.css'

// Surfaces per-resume ingestion coverage/warnings next to a score.
//
// NOTE (honest gap): the Phase 7 `GET /api/jobs/{id}/status` response
// (`JobStatusResponse`) only carries `{candidate_id, filename, status, error}`.
// It does NOT expose coverage / warnings yet, even though Phase 2's IngestReport
// produces them internally. Until the backend exposes those fields, this badge
// renders what IS available (ingest status + error) and shows an explicit
// "coverage not yet exposed" placeholder — we do NOT fabricate a percentage.
export default function CoverageBadge({ status, coverage, warnings }) {
  const statusClass =
    status === 'error' ? 'badge-error' : status === 'done' ? 'badge-done' : 'badge-pending'

  return (
    <div className={`coverage-badge ${statusClass}`} title="Resume ingestion coverage">
      <span className="coverage-badge-status">
        {status === 'error' ? '⚠ ingest error' : status === 'done' ? '✓ indexed' : '⏳ pending'}
      </span>

      {status === 'error' && <span className="coverage-badge-error">— failed</span>}

      {coverage != null ? (
        <span className="coverage-badge-value">{Math.round(coverage * 100)}% indexed</span>
      ) : (
        <span className="coverage-badge-gap">coverage not yet exposed by API</span>
      )}

      {Array.isArray(warnings) && warnings.length > 0 && (
        <span className="coverage-badge-warnings">⚠ {warnings.length} warning(s)</span>
      )}
    </div>
  )
}
