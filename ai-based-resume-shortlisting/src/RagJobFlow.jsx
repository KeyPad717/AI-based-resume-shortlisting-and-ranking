import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createJob,
  uploadResumes,
  getJobStatus,
  scoreJob,
  getEvidence,
} from './api'
import EvidenceChip from './components/EvidenceChip'
import EvidenceDrawer from './components/EvidenceDrawer'
import CoverageBadge from './components/CoverageBadge'
import WeightPanel from './components/WeightPanel'
import { ragScoreClass } from './evidence/scoring'
import { DEFAULT_WEIGHTS } from './evidence/weights'
import './RagJobFlow.css'

export default function RagJobFlow() {
  const [jdFile, setJdFile] = useState(null)
  const [resumeFiles, setResumeFiles] = useState([])
  const [jobId, setJobId] = useState(null)
  const [statuses, setStatuses] = useState([])
  const [stage, setStage] = useState('setup') // setup | ingesting | scored
  const [results, setResults] = useState([])
  const [error, setError] = useState('')
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS)
  const [openEvidence, setOpenEvidence] = useState(null)
  const [initialScoreConflict, setInitialScoreConflict] = useState(false)
  const pollTimer = useRef(null)

  const stopPolling = () => {
    if (pollTimer.current) {
      clearTimeout(pollTimer.current)
      pollTimer.current = null
    }
  }

  const allSettled = (list) =>
    list.length > 0 && list.every((s) => s.status === 'done' || s.status === 'error')

  const pollStatus = useCallback(async () => {
    try {
      const data = await getJobStatus(jobId)
      setStatuses(data.resumes || [])
      if (allSettled(data.resumes || [])) {
        stopPolling()
        setStage('scored')
        await runScore(weights)
      } else {
        pollTimer.current = setTimeout(pollStatus, 1500)
      }
    } catch (err) {
      setError(err.message || 'Failed to poll job status')
      stopPolling()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  const runScore = async (w) => {
    setInitialScoreConflict(false)
    try {
      const res = await scoreJob(jobId, w)
      setResults(res)
      return res
    } catch (err) {
      if (err.status === 409) {
        setInitialScoreConflict(true)
        return { conflict: true }
      }
      setError(err.message || 'Scoring failed')
      return null
    }
  }

  const handleStart = async () => {
    if (!jdFile) return setError('Please choose a JD file (PDF/DOCX)')
    if (resumeFiles.length === 0) return setError('Please choose at least one resume')
    setError('')
    try {
      const created = await createJob({ jdFile, weights })
      setJobId(created.job_id)
      const uploaded = await uploadResumes(created.job_id, resumeFiles)
      setStatuses(uploaded.resumes || [])
      setStage('ingesting')
    } catch (err) {
      setError(err.message || 'Failed to create job')
    }
  }

  useEffect(() => {
    if (stage === 'ingesting' && jobId) {
      pollStatus()
    }
    return stopPolling
  }, [stage, jobId, pollStatus])

  const handleRescore = useCallback(
    (w) => runScore(w),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [jobId]
  )

  return (
    <div className="rag-flow">
      {error && <div className="error-message">⚠️ {error}</div>}

      {stage === 'setup' && (
        <div className="upload-section">
          <div className="upload-box">
            <h3>Job Description</h3>
            <input type="file" accept=".pdf,.docx" className="file-input" id="rag-jd"
              onChange={(e) => setJdFile(e.target.files[0])} />
            <label htmlFor="rag-jd" className="file-input-label">
              {jdFile ? `✓ ${jdFile.name}` : '📄 Choose JD file (PDF/DOCX)'}
            </label>
          </div>
          <div className="upload-box">
            <h3>Resume Collection</h3>
            <input type="file" accept=".pdf,.docx" multiple className="file-input" id="rag-resumes"
              onChange={(e) => setResumeFiles(Array.from(e.target.files))} />
            <label htmlFor="rag-resumes" className="file-input-label">
              {resumeFiles.length > 0 ? `✓ ${resumeFiles.length} files selected` : '📁 Choose resume files (PDF/DOCX)'}
            </label>
          </div>
        </div>
      )}

      {stage === 'setup' && (
        <div className="run-section">
          <button onClick={handleStart} className="run-button">
            🚀 Create Job & Ingest
          </button>
        </div>
      )}

      {stage === 'ingesting' && (
        <div className="results-section">
          <h2>⏳ Ingesting & Indexing Resumes</h2>
          <ul className="rag-status-list">
            {statuses.map((s) => (
              <li key={s.candidate_id} className={`rag-status-item rag-status-${s.status}`}>
                <span className="rag-status-filename">{s.filename}</span>
                <CoverageBadge status={s.status} />
                {s.error && <span className="rag-status-error">— {s.error}</span>}
              </li>
            ))}
          </ul>
        </div>
      )}

      {stage === 'scored' && (
        <>
          <div className="results-section">
            <div className="rag-results-header">
              <h2>📊 RAG-Scored Candidates</h2>
              <span className="rag-job-id">job {jobId}</span>
            </div>

            {initialScoreConflict && (
              <div className="success-message rag-conflict-hint">
                No indexed resumes scored yet — re-check /status. Rescoring is disabled until done.
              </div>
            )}

            {results.length === 0 && !initialScoreConflict && (
              <p className="rag-empty">No results. Score by adjusting weights below.</p>
            )}

            <div className="rag-cards">
              {results.map((result, index) => {
                const status = statuses.find((x) => x.candidate_id === result.name)
                return (
                  <div key={result.name} className="rag-card">
                    <div className="rag-card-top">
                      <span className={ragScoreClass(result.final_score)}>
                        {index + 1}. {(result.final_score * 100).toFixed(1)}%
                      </span>
                      <strong className="rag-card-name">{result.name}</strong>
                      <div className="rag-card-badge">
                        {status && <CoverageBadge status={status.status} />}
                      </div>
                    </div>
                    <p className="rag-card-summary">{result.semantic_summary}</p>
                    {Array.isArray(result.skill_verdicts) && result.skill_verdicts.length > 0 && (
                      <div className="rag-chip-row">
                        {result.skill_verdicts.map((v) => (
                          <EvidenceChip
                            key={v.skill_name}
                            skillName={v.skill_name}
                            verdict={v.verdict}
                            onClick={() =>
                              setOpenEvidence({
                                jobId,
                                candidateId: result.name,
                                skillName: v.skill_name,
                                verdict: v.verdict,
                                citedChunkIds: v.cited_chunk_ids || [],
                              })
                            }
                          />
                        ))}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>

            {results.length > 0 && (
              <WeightPanel
                values={weights}
                onChange={setWeights}
                onRescore={handleRescore}
                jobId={jobId}
              />
            )}
          </div>
        </>
      )}

      {openEvidence && (
        <EvidenceDrawer
          jobId={openEvidence.jobId}
          candidateId={openEvidence.candidateId}
          skillName={openEvidence.skillName}
          verdict={openEvidence.verdict}
          citedChunkIds={openEvidence.citedChunkIds}
          onClose={() => setOpenEvidence(null)}
          api={getEvidence}
        />
      )}
    </div>
  )
}
