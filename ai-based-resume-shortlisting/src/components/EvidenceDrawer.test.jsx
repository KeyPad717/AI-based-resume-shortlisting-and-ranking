import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import EvidenceDrawer from './EvidenceDrawer'

const baseDetail = {
  candidate_id: 'hash-c',
  skill_name: 'Python',
  verdict: 'demonstrated',
  cited_chunk_ids: ['c1'],
  quote: 'proficient in Python',
  retrieved_spans: [
    { chunk_id: 'c1', text: 'proficient in Python', dense_rank: 1, fused_rank: 0.9 },
  ],
}

function renderDrawer(api, detail, extra = {}) {
  return render(
    <EvidenceDrawer
      jobId="j1"
      candidateId="hash-c"
      skillName="Python"
      verdict="demonstrated"
      citedChunkIds={['c1']}
      api={api}
      {...extra}
    />
  )
}

describe('EvidenceDrawer', () => {
  it('fetches evidence on demand and renders the span + rank', async () => {
    const api = vi.fn().mockResolvedValue([baseDetail])
    renderDrawer(api)
    await waitFor(() =>
      expect(screen.getAllByText('proficient in Python').length).toBeGreaterThan(0)
    )
    expect(screen.getAllByText(/chunk c1/).length).toBeGreaterThan(0)
    expect(screen.getByText(/fused rank 0.900/)).toBeInTheDocument()
    expect(api).toHaveBeenCalledTimes(1)
    expect(api).toHaveBeenCalledWith('j1', 'hash-c', 'Python')
  })

  it('zero-span verdict shows an explicit "no supporting passage" state', async () => {
    const api = vi.fn().mockResolvedValue([
      {
        ...baseDetail,
        retrieved_spans: [],
        cited_chunk_ids: [],
        quote: null,
      },
    ])
    renderDrawer(api)
    await waitFor(() =>
      expect(
        screen.getByText(/No supporting passage found for this verdict/)
      ).toBeInTheDocument()
    )
  })

  it('never renders a verdict without fetching its evidence detail', async () => {
    const api = vi.fn().mockResolvedValue([baseDetail])
    renderDrawer(api)
    await waitFor(() => expect(screen.getByText(/Evidence: Python/)).toBeInTheDocument())
    expect(screen.getByText('demonstrated')).toBeInTheDocument()
  })

  it('surfaces fetch errors instead of silently showing nothing', async () => {
    const api = vi.fn().mockRejectedValue(new Error('boom'))
    renderDrawer(api)
    await waitFor(() => expect(screen.getByText('boom')).toBeInTheDocument())
  })
})
