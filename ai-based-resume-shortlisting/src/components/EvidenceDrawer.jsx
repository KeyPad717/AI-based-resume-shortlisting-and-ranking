import { useEffect, useState } from 'react'
import { getEvidence } from '../api'
import { makeHighlight } from '../evidence/evidenceKey'
import CitationQuote from './CitationQuote'
import './EvidenceDrawer.css'

// Cache of offset-based highlights, DNS-keyed by chunk_id + content-hash so a
// re-ingested (changed) resume invalidates stale offsets. `candidateId` is the
// resume's content hash (Phase 2 chunker), so a changed resume yields a new id.
const offsetHighlightCache = new Map()

export default function EvidenceDrawer({
  jobId,
  candidateId,
  skillName,
  verdict,
  onClose,
  api = getEvidence,
}) {
  const [data, setData] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [offsetHighlights, setOffsetHighlights] = useState([])

  useEffect(() => {
    let cancelled = false
    api(jobId, candidateId, skillName)
      .then((res) => {
        if (cancelled) return
        setError('')
        setData(res || [])

        // Rebuild offset-cache entries keyed by chunk_id + this resume's
        // content-hash (candidateId). A re-ingested, changed resume carries a
        // different content-hash, so its previous cached offsets key to a
        // different entry and are ignored — i.e. stale offsets are dropped.
        const highlights = []
        for (const detail of res || []) {
          for (const span of detail.retrieved_spans || []) {
            // SpanOut exposes chunk_id + text but no char offsets yet (Phase 7
            // gap). When offsets are added, pass real start/end to keep the
            // cache keyed. Until then, guard with the same key over whole text.
            const hl = makeHighlight(
              span.chunk_id,
              candidateId,
              span.text.length,
              0,
              span.text.length
            )
            highlights.push({ chunkId: span.chunk_id, key: hl.key })
            offsetHighlightCache.set(hl.key, { chunkId: span.chunk_id, candidateId })
          }
        }
        setOffsetHighlights(highlights)
      })
      .catch((err) => {
        if (!cancelled) setError(err.message || 'Failed to load evidence')
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [jobId, candidateId, skillName, api])

  const detail = data.find((d) => d.skill_name === skillName && d.candidate_id === candidateId)
  const spanCount = detail ? (detail.retrieved_spans || []).length : 0

  return (
    <div className="evidence-drawer" role="dialog" aria-label={`Evidence for ${skillName}`}>
      <div className="evidence-drawer-header">
        <h4>Evidence: {skillName}</h4>
        <button type="button" className="evidence-drawer-close" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>

      <div className="evidence-drawer-verdict">
        <span className="evidence-drawer-verdict-label">Verdict</span>
        <span className={`evidence-drawer-verdict-value verdict-${verdict || 'unknown'}`}>
          {verdict || 'unknown'}
        </span>
        <span className="evidence-drawer-verdict-note">
          off {spanCount} retrieved span{spanCount === 1 ? '' : 's'}
        </span>
      </div>

      {loading && <p className="evidence-drawer-status">Loading evidence…</p>}
      {error && !loading && <p className="evidence-drawer-status evidence-drawer-error">{error}</p>}

      {!loading && !error && (
        <div className="evidence-drawer-body">
          {detail ? (
            <>
              <CitationQuote
                quote={detail.quote || null}
                section={detail.section_type}
                chunkId={detail.cited_chunk_ids?.[0]}
                offset={offsetHighlights[0] ? { start: 0 } : null}
              />
              {(detail.retrieved_spans || []).length === 0 && (
                // Verdict is present but no supporting span was retrieved —
                // surface this explicitly instead of implying a citation.
                <p className="evidence-drawer-no-span">
                  No supporting passage found for this verdict.
                </p>
              )}
              {(detail.retrieved_spans || []).length > 0 && (
                <ul className="evidence-drawer-spans">
                  {(detail.retrieved_spans || []).map((span) => (
                    <li key={span.chunk_id} className="evidence-drawer-span">
                      <span className="evidence-drawer-span-text">{span.text}</span>
                      <span className="evidence-drawer-span-meta">
                        chunk {span.chunk_id}
                        {span.fused_rank != null && (
                          <span className="evidence-drawer-span-rank">
                            fused rank {Number(span.fused_rank).toFixed(3)}
                          </span>
                        )}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </>
          ) : (
            <p className="evidence-drawer-no-span">No evidence detail returned.</p>
          )}
        </div>
      )}
    </div>
  )
}
