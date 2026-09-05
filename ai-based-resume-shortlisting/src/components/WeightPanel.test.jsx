import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useState } from 'react'
import { render, screen, fireEvent, act } from '@testing-library/react'
import WeightPanel from './WeightPanel'

const DEFAULT = {
  required_skills: 30,
  semantic: 15,
  reranker: 15,
  experience: 15,
  education: 10,
  projects: 10,
  evidence: 5,
}

function Harness({ onRescore, debounceMs }) {
  const [values, setValues] = useState(DEFAULT)
  return (
    <WeightPanel
      values={values}
      onChange={setValues}
      onRescore={onRescore}
      jobId="j1"
      debounceMs={debounceMs}
    />
  )
}

beforeEach(() => vi.useFakeTimers())
afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('WeightPanel debounce bound', () => {
  it('collapses rapid consecutive changes into a single rescore after the debounce window', async () => {
    const onRescore = vi.fn().mockResolvedValue(null)
    render(<Harness onRescore={onRescore} debounceMs={400} />)

    const slider = screen.getAllByRole('slider')[0]
    for (const v of [20, 25, 30]) {
      fireEvent.change(slider, { target: { value: String(v) } })
    }

    expect(onRescore).not.toHaveBeenCalled()
    await act(async () => {
      await vi.advanceTimersByTimeAsync(410)
    })
    expect(onRescore).toHaveBeenCalledTimes(1)
  })

  it('keeps per-component rescoring independent across a later change', async () => {
    const onRescore = vi.fn().mockResolvedValue(null)
    render(<Harness onRescore={onRescore} debounceMs={100} />)

    const sliders = screen.getAllByRole('slider')
    fireEvent.change(sliders[0], { target: { value: '40' } })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })
    fireEvent.change(sliders[1], { target: { value: '33' } })
    await act(async () => {
      await vi.advanceTimersByTimeAsync(200)
    })
    expect(onRescore).toHaveBeenCalledTimes(2)
  })
})

describe('WeightPanel 409 conflict handling', () => {
  it('reports "waiting" conflict, not a generic error, when rescore returns 409', async () => {
    const onRescore = vi.fn().mockResolvedValue({ conflict: true })
    render(<Harness onRescore={onRescore} debounceMs={100} />)

    const slider = screen.getAllByRole('slider')[0]
    fireEvent.change(slider, { target: { value: '25' } })
    await act(() => vi.advanceTimersByTimeAsync(200))

    expect(
      screen.getByText(/no resumes indexed for this job yet/)
    ).toBeInTheDocument()
    expect(screen.queryByText(/Rescoring failed/)).not.toBeInTheDocument()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  })

  it('does not show conflict when a rescore succeeds', async () => {
    const onRescore = vi.fn().mockResolvedValue([{ name: 'A', final_score: 0.9 }])
    render(<Harness onRescore={onRescore} debounceMs={100} />)
    const slider = screen.getAllByRole('slider')[0]
    fireEvent.change(slider, { target: { value: '25' } })
    await act(() => vi.advanceTimersByTimeAsync(200))

    expect(onRescore).toHaveBeenCalledTimes(1)
    expect(screen.queryByText(/no resumes indexed for this job yet/)).not.toBeInTheDocument()
  })
})
