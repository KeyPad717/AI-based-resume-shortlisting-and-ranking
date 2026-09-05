// HTTP client for the Phase 7 job-based FastAPI surface.
//
// The legacy `POST /api/score` flow in App.jsx keeps its own inline fetch to
// `http://localhost:8000/api/score` (untouched). This module is additive and is
// used ONLY by the new RAG ("Evidence") view via `/api/jobs*` endpoints.

export const API_BASE = 'http://localhost:8000'

async function toApiError(res) {
  let detail = `HTTP ${res.status}`
  try {
    const data = await res.json()
    if (data && typeof data.detail === 'string') detail = data.detail
  } catch {
    /* non-JSON body */
  }
  const err = new Error(detail)
  err.status = res.status
  return err
}

export async function createJob({ jdFile, weights }) {
  const fd = new FormData()
  fd.append('jd_file', jdFile)
  if (weights) fd.append('weights', JSON.stringify(weights))
  const res = await fetch(`${API_BASE}/api/jobs`, { method: 'POST', body: fd })
  if (!res.ok) throw await toApiError(res)
  return res.json()
}

export async function uploadResumes(jobId, resumeFiles) {
  const fd = new FormData()
  resumeFiles.forEach((f) => fd.append('resume_files', f))
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/resumes`, {
    method: 'POST',
    body: fd,
  })
  if (!res.ok) throw await toApiError(res)
  return res.json()
}

export async function getJobStatus(jobId) {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/status`)
  if (!res.ok) throw await toApiError(res)
  return res.json()
}

// POST /api/jobs/{id}/score  ->  List[ScoreResult]; 409 when no resume is done.
export async function scoreJob(jobId, weights) {
  const res = await fetch(`${API_BASE}/api/jobs/${jobId}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ weights: weights || null }),
  })
  if (!res.ok) throw await toApiError(res)
  return res.json()
}

// GET /api/jobs/{id}/candidates/{candidate_id}/evidence?skill=...
// -> List[EvidenceDetailResponse]
export async function getEvidence(jobId, candidateId, skillName) {
  const candidate = encodeURIComponent(candidateId)
  const params = new URLSearchParams()
  if (skillName) params.set('skill', skillName)
  const res = await fetch(
    `${API_BASE}/api/jobs/${jobId}/candidates/${candidate}/evidence?${params.toString()}`
  )
  if (!res.ok) throw await toApiError(res)
  return res.json()
}
