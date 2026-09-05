import { verdictMeta } from '../evidence/verdicts'
import './EvidenceChip.css'

export default function EvidenceChip({ skillName, verdict, onClick, disabled }) {
  const meta = verdictMeta(verdict)
  const clickable = typeof onClick === 'function' && !disabled
  return (
    <span
      role="button"
      tabIndex={clickable ? 0 : -1}
      title={`${skillName}: ${meta.label}`}
      className={`evidence-chip ${meta.className}${clickable ? ' clickable' : ''}`}
      onClick={clickable ? () => onClick() : undefined}
      onKeyDown={
        clickable
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onClick()
              }
            }
          : undefined
      }
    >
      <span className="evidence-chip-skill">{skillName}</span>
      <span className="evidence-chip-verdict">{meta.label}</span>
    </span>
  )
}
