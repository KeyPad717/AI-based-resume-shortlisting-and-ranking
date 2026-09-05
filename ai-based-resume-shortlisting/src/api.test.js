import { describe, it, expect, vi, afterEach } from 'vitest'
import { scoreJob, createJob, getEvidence } from './api'

function mockFetchOnce(status, body) {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: vi.fn().mockResolvedValue(body),
  })
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('api layer', () => {
  it('scoreJob returns List[ScoreResult] on success and posts weights as JSON', async () => {
    mockFetchOnce(200, [{ name: 'A', final_score: 0.8 }])
    const res = await scoreJob('j1', { evidence: 5 })
    expect(res).toHaveLength(1)
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/jobs/j1/score',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ weights: { evidence: 5 } }),
      })
    )
  })

  it('409 is surfaced as a distinct conflict error, not a generic message', async () => {
    mockFetchOnce(409, { detail: 'No indexed resumes yet for this job; check /status' })
    try {
      await scoreJob('j1', null)
      throw new Error('should have thrown')
    } catch (err) {
      expect(err.status).toBe(409)
      expect(err.message).toMatch(/No indexed resumes yet/)
    }
  })

  it('createJob posts jd_file multipart', async () => {
    mockFetchOnce(200, { job_id: 'j1', normalized_requirements: [] })
    const file = new File(['x'], 'jd.pdf', { type: 'application/pdf' })
    await createJob({ jdFile: file, weights: null })
    const [, opts] = globalThis.fetch.mock.calls[0]
    expect(opts.method).toBe('POST')
    expect(opts.body).toBeInstanceOf(FormData)
    expect(opts.body.get('jd_file')).toBe(file)
  })

  it('getEvidence builds the candidates evidence query', async () => {
    mockFetchOnce(200, [])
    await getEvidence('j1', 'hash-1', 'Python')
    expect(globalThis.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/jobs/j1/candidates/hash-1/evidence?skill=Python'
    )
  })
})
