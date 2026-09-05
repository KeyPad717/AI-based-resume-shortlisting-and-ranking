import { useState } from 'react'
import './CitationQuote.css'

const CLAMP_CHARS = 200

export default function CitationQuote({ quote, section, chunkId, offset }) {
  const [expanded, setExpanded] = useState(false)
  if (!quote || typeof quote !== 'string') return null

  const long = quote.length > CLAMP_CHARS
  const shown = long && !expanded ? `${quote.slice(0, CLAMP_CHARS)}…` : quote

  return (
    <figure className="citation-quote">
      <blockquote className="citation-quote-text">{shown}</blockquote>
      <figcaption className="citation-quote-meta">
        {section && <span className="citation-quote-section">{section}</span>}
        {chunkId && <span className="citation-quote-chunk">chunk {chunkId}</span>}
        {offset && offset.start != null && (
          <span className="citation-quote-offset">¶ {offset.start + 1}</span>
        )}
        {long && (
          <button
            type="button"
            className="citation-quote-toggle"
            onClick={() => setExpanded((v) => !v)}
          >
            {expanded ? 'Collapse' : 'Show full passage'}
          </button>
        )}
      </figcaption>
    </figure>
  )
}
