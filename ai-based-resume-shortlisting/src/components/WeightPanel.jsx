import { useEffect, useRef, useState } from 'react'
import { WEIGHT_KEYS } from '../evidence/weights'
import './WeightPanel.css'

// Debounced rescore bound. Kept small so repeated slider input collapses into a
// single request while still feeling live. `debounceMs` is injectable for tests.
export default function WeightPanel({
  values,
  onChange,
  onRescore,
  disabled,
  debounceMs = 400,
  score = null,
  rescore = null,
}) {
  const timer = useRef(null)
  const [pending, setPending] = useState(false)
  const [conflict, setConflict] = useState(false)

  const update = (key, raw) => {
    const v = Math.max(0, Math.min(100, parseInt(raw, 10) || 0))
    const next = { ...values, [key]: v }
    onChange(next)
  }

  const triggerRescore = (weights) => {
    setPending(true)
    setConflict(false)
    onRescore(weights)
      .then((res) => {
        if (res && res.conflict) setConflict(true)
      })
      .catch(() => {})
      .finally(() => setPending(false))
  }

  const schedule = (next) => {
    setConflict(false)
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => triggerRescore(next), debounceMs)
  }

  useEffect(() => {
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
  }, [])

  const total = Object.values(values).reduce((a, b) => a + b, 0)

  return (
    <div className="weight-panel">
      <h4>⚖️ Weighting (RAG rescore)</h4>
      <div className="weight-panel-grid">
        {WEIGHT_KEYS.map(([key, label]) => (
          <label key={key} className="weight-panel-item">
            <span className="weight-panel-label">{label}</span>
            <input
              type="range"
              min="0"
              max="100"
              value={values[key]}
              disabled={disabled}
              onChange={(e) => {
                const next = { ...values, [key]: Number(e.target.value) }
                onChange(next)
                schedule(next)
              }}
            />
            <input
              type="number"
              className="weight-panel-input"
              min="0"
              max="100"
              value={values[key]}
              disabled={disabled}
              onChange={(e) => {
                update(key, e.target.value)
                const val = Math.max(0, Math.min(100, parseInt(e.target.value, 10) || 0))
                schedule({ ...values, [key]: val })
              }}
            />
          </label>
        ))}
      </div>
      <div className={`weight-panel-total ${total === 100 ? 'valid' : 'invalid'}`}>
        Total: {total}%{total !== 100 ? ' — must be 100 to score' : ''}
      </div>
      {pending && <div className="weight-panel-status">⟳ Rescoring…</div>}
      {conflict && (
        <div className="weight-panel-status weight-panel-conflict">
          Waiting — no resumes indexed for this job yet (409). Check /status.
        </div>
      )}
      {rescore && <div className="weight-panel-status">✓ Rescored</div>}
      {score != null && <div className="weight-panel-status">Current final score</div>}
    </div>
  )
}
