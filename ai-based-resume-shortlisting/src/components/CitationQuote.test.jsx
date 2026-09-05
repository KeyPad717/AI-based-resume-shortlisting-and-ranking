import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import CitationQuote from './CitationQuote'

const LONG = 'P'.repeat(300)

describe('CitationQuote clamping', () => {
  it('clamps a long quote and offers expand', () => {
    render(<CitationQuote quote={LONG} />)
    const text = screen.getByText(/P{200}…/)
    expect(text).toBeInTheDocument()
    const toggle = screen.getByRole('button', { name: /Show full passage/i })
    expect(toggle).toBeInTheDocument()
  })

  it('expands to the full passage on click', () => {
    render(<CitationQuote quote={LONG} />)
    fireEvent.click(screen.getByRole('button', { name: /Show full passage/i }))
    expect(screen.getByText(LONG)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Collapse/i })).toBeInTheDocument()
  })

  it('does not offer a toggle for a short quote', () => {
    render(<CitationQuote quote="short" />)
    expect(screen.getByText('short')).toBeInTheDocument()
    expect(screen.queryByRole('button')).not.toBeInTheDocument()
  })

  it('renders section and chunk reference when available', () => {
    render(<CitationQuote quote={'x'.repeat(10)} section="EXPERIENCE" chunkId="c-1" />)
    expect(screen.getByText('EXPERIENCE')).toBeInTheDocument()
    expect(screen.getByText('chunk c-1')).toBeInTheDocument()
  })

  it('renders nothing for an empty quote', () => {
    const { container } = render(<CitationQuote quote={null} />)
    expect(container.firstChild).toBeNull()
  })

  it('calls no network (pure presentational)', () => {
    const spy = vi.fn()
    render(<CitationQuote quote="x" onClick={spy} />)
    expect(spy).not.toHaveBeenCalled()
  })
})
