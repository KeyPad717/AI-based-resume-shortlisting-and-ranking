import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import EvidenceChip from './EvidenceChip'
import { verdictMeta } from '../evidence/verdicts'

describe('verdict mapping (explicit, no silent fallback)', () => {
  const cases = [
    ['demonstrated', 'Demonstrated'],
    ['claimed_only', 'Claimed only'],
    ['absent', 'Absent'],
  ]
  it.each(cases)('maps %s -> %s', (verdict, label) => {
    expect(verdictMeta(verdict).label).toBe(label)
  })

  it('unknown verdict is handled explicitly, not coerced', () => {
    expect(verdictMeta('unexpected_server_value').label).toBe('Unknown')
    expect(verdictMeta(undefined).label).toBe('Unknown')
  })

  it('renders skill + verdict label for a demonstrated verdict', () => {
    render(<EvidenceChip skillName="Python" verdict="demonstrated" />)
    expect(screen.getByText('Python')).toBeInTheDocument()
    expect(screen.getByText('Demonstrated')).toBeInTheDocument()
  })

  it('renders the distinct Unknown state for an unrecognized verdict', () => {
    render(<EvidenceChip skillName="Go" verdict="weird" />)
    expect(screen.getByText('Go')).toBeInTheDocument()
    expect(screen.getByText('Unknown')).toBeInTheDocument()
  })

  it('is clickable only when a handler is provided', () => {
    const { rerender } = render(<EvidenceChip skillName="K" verdict="absent" />)
    expect(screen.getByRole('button')).toHaveAttribute('tabindex', '-1')
    rerender(<EvidenceChip skillName="K" verdict="absent" onClick={() => {}} />)
    expect(screen.getByRole('button').onclick).toBeDefined()
  })
})
