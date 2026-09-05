import { describe, it, expect } from 'vitest'
import {
  evidenceCacheKey,
  makeHighlight,
  clampSlice,
  isHighlightStale,
} from './evidenceKey'

describe('evidence cache-key staleness guard', () => {
  it('keys highlights by chunk_id + content-hash (candidate_id)', () => {
    expect(evidenceCacheKey('c1', 'hashA')).toBe('c1:hashA')
    expect(evidenceCacheKey('c1', 'hashB')).not.toBe(evidenceCacheKey('c1', 'hashA'))
    expect(evidenceCacheKey('c1', 'hashA')).toBe(evidenceCacheKey('c1', 'hashA'))
  })

  it('a changed content-hash invalidates a previously cached offset entry', () => {
    const before = evidenceCacheKey('chunk-9', 'abc123')
    expect(isHighlightStale(before, 'chunk-9', 'xyz789')).toBe(true)
    expect(isHighlightStale(before, 'chunk-9', 'abc123')).toBe(false)
  })

  it('a changed chunk id also invalidates the entry', () => {
    const key = evidenceCacheKey('chunk-9', 'abc123')
    expect(isHighlightStale(key, 'chunk-other', 'abc123')).toBe(true)
  })

  it('clampSlice rejects malformed/empty slices', () => {
    expect(clampSlice(10, 0, 0)).toBeNull()
    expect(clampSlice(10, -1, 5)).toBeNull()
    expect(clampSlice(10, 11, 12)).toBeNull()
    expect(clampSlice(10, NaN, 5)).toBeNull()
    expect(clampSlice(10, 4, 12)).toEqual([4, 10])
  })

  it('makeHighlight returns the keyed slice and null offsets when no valid pair', () => {
    const a = makeHighlight('c1', 'hashA', 100, 10, 40)
    expect(a).toMatchObject({ key: 'c1:hashA', start: 10, end: 40 })
    const bad = makeHighlight('c1', 'hashA', 100, 5, 5)
    expect(bad.start).toBeNull()
    expect(bad.end).toBeNull()
  })
})
