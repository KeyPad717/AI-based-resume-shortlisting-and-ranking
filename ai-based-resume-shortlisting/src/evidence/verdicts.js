// Shared verdict metadata used by EvidenceChip / EvidenceDrawer.
// Kept out of the component file so it can be imported by tests and other
// components without triggering react-refresh export rules.

// Explicit four-way verdict map (no silent fallback).
// `unknown` covers any verdict string the backend does not yet emit (forward
// compatible) and is rendered distinctly rather than being coerced.
export const VERDICT_META = {
  demonstrated: { label: 'Demonstrated', className: 'verdict-demonstrated' },
  claimed_only: { label: 'Claimed only', className: 'verdict-claimed' },
  absent: { label: 'Absent', className: 'verdict-absent' },
  unknown: { label: 'Unknown', className: 'verdict-unknown' },
}

export function verdictMeta(verdict) {
  return VERDICT_META[verdict] || VERDICT_META.unknown
}
