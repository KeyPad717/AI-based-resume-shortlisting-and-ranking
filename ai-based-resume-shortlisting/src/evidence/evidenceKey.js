// Evidence cache-keying + offset-staleness guard.
//
// Phase 2's chunker derives a candidate's content hash as its `candidate_id`
// (a stable fingerprint of the resume's extracted text). A re-ingested resume
// yields the SAME candidate_id only when the content is unchanged; any edit
// produces a different candidate_id.
//
// Any character-offset based highlight we cache must therefore be DNS-keyed by
// `chunk_id + candidate_id`. If a resume is re-ingested (content changed), the
// candidate_id changes, the key changes, and stale char offsets are dropped
// instead of being rendered against mismatched text.

// Build the cache key for a set of offset-based highlights.
export function evidenceCacheKey(chunkId, contentHash) {
  return `${chunkId}:${contentHash}`
}

// Normalize a possibly-unbounded slice into [start, end] clamped to the text
// length. Returns null when the slice is invalid or empty.
export function clampSlice(textLength, start, end) {
  if (
    !Number.isFinite(start) ||
    !Number.isFinite(end) ||
    start < 0 ||
    end <= start ||
    start >= textLength
  ) {
    return null
  }
  return [start, Math.min(end, textLength)]
}

// Compute the highlightable text slice for a chunk, keyed so the caller can map
// it to a cache entry. Returns { key, start, end } (start/end may be null if no
// valid offset pair exists — the caller should fall back to whole-text).
export function makeHighlight(chunkId, contentHash, textLength, start, end) {
  const slice = clampSlice(textLength, start, end)
  return {
    key: evidenceCacheKey(chunkId, contentHash),
    start: slice ? slice[0] : null,
    end: slice ? slice[1] : null,
  }
}

// Determine whether a stored/cached highlight entry is still valid for the
// given chunk + content-hash. A mismatched key means the resume content (or the
// chunk) changed and any offsets we stored are stale.
export function isHighlightStale(entryKey, chunkId, contentHash) {
  return entryKey !== evidenceCacheKey(chunkId, contentHash)
}
