import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import RagJobFlow from './RagJobFlow'
import { ragScoreClass } from './evidence/scoring'

vi.mock('./api', () => ({
  createJob: vi.fn(),
  uploadResumes: vi.fn(),
  getJobStatus: vi.fn(),
  scoreJob: vi.fn(),
  getEvidence: vi.fn(),
}))

import { createJob, uploadResumes, getJobStatus, scoreJob, getEvidence } from './api'

const RESULTS = [
  {
    name: 'cand-1',
    final_score: 0.81,
    semantic_summary: 'strong ML background',
    skill_verdicts: [
      { skill_name: 'Python', verdict: 'demonstrated', cited_chunk_ids: ['c1'] },
      { skill_name: 'AWS', verdict: 'absent', cited_chunk_ids: [] },
    ],
  },
]

function fileSeq() {
  const jd = new File(['jd'], 'jd.pdf', { type: 'application/pdf' })
  const r1 = new File(['r'], 'r.pdf', { type: 'application/pdf' })
  return { jd, r1 }
}

beforeEach(() => {
  vi.clearAllMocks()
  createJob.mockResolvedValue({ job_id: 'j1', normalized_requirements: [] })
  uploadResumes.mockResolvedValue({
    job_id: 'j1',
    resumes: [{ candidate_id: 'cand-1', filename: 'r.pdf', status: 'done' }],
  })
  getJobStatus.mockResolvedValue({
    job_id: 'j1',
    resumes: [{ candidate_id: 'cand-1', filename: 'r.pdf', status: 'done' }],
  })
  scoreJob.mockResolvedValue(RESULTS)
  getEvidence.mockResolvedValue([
    {
      candidate_id: 'cand-1',
      skill_name: 'Python',
      verdict: 'demonstrated',
      cited_chunk_ids: ['c1'],
      quote: 'proficient in Python',
      retrieved_spans: [{ chunk_id: 'c1', text: 'proficient in Python', fused_rank: 0.9 }],
    },
  ])
})

describe('RagJobFlow', () => {
  it('runs create -> ingest -> score and surfaces chips + coverage badge', async () => {
    const { jd, r1 } = fileSeq()
    render(<RagJobFlow />)

    fireEvent.change(screen.getByLabelText(/Choose JD file/i), { target: { files: [jd] } })
    const resumes = screen.getByLabelText(/Choose resume files/i)
    fireEvent.change(resumes, { target: { files: [r1] } })
    fireEvent.click(screen.getByRole('button', { name: /Create Job & Ingest/i }))

    expect(await screen.findByText(/Python/)).toBeInTheDocument()
    expect(screen.getByText('Demonstrated')).toBeInTheDocument()
    expect(await screen.findByText(/indexed/)).toBeInTheDocument()
    expect(createJob).toHaveBeenCalledTimes(1)
    expect(scoreJob).toHaveBeenCalledTimes(1)
  })

  it('opens the evidence drawer when a chip is clicked', async () => {
    const { jd, r1 } = fileSeq()
    render(<RagJobFlow />)
    fireEvent.change(screen.getByLabelText(/Choose JD file/i), { target: { files: [jd] } })
    const resumes = screen.getByLabelText(/Choose resume files/i)
    fireEvent.change(resumes, { target: { files: [r1] } })
    fireEvent.click(screen.getByRole('button', { name: /Create Job & Ingest/i }))

    const chip = await screen.findByText('Demonstrated')
    fireEvent.click(chip)
    await waitFor(() => expect(getEvidence).toHaveBeenCalledWith('j1', 'cand-1', 'Python'))
    const passages = await screen.findAllByText(/proficient in Python/)
    expect(passages.length).toBeGreaterThan(0)
    expect(await screen.findByText(/Evidence: Python/)).toBeInTheDocument()
  })
})

describe('ragScoreClass (provisional RAG thresholds)', () => {
  it('returns distinct classes without mutating the legacy split', () => {
    expect(ragScoreClass(0.85)).toBe('rag-score-high')
    expect(ragScoreClass(0.6)).toBe('rag-score-medium')
    expect(ragScoreClass(0.2)).toBe('rag-score-low')
  })
})
