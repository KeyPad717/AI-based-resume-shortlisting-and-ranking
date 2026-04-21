import { useState } from 'react'
import './App.css'

function App() {
  const [jdFile, setJdFile] = useState(null)
  const [resumeFiles, setResumeFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  
  // Dynamic weightage configuration
  // Dynamic weightage configuration aligned with backend
  const [weightage, setWeightage] = useState({
    required_skills: 30,
    semantic: 15,
    reranker: 15,
    experience: 15,
    education: 10,
    projects: 10,
    evidence: 5
  })

  const handleJdUpload = (e) => {
    const file = e.target.files[0]
    if (file && (file.type === 'application/pdf' || file.name.endsWith('.docx'))) {
      setJdFile(file)
      setError('')
    } else {
      setError('Please upload a valid PDF or DOCX file for JD')
    }
  }

  const handleResumeUpload = (e) => {
    const files = Array.from(e.target.files)
    const validFiles = files.filter(file => 
      file.type === 'application/pdf' || file.name.endsWith('.docx')
    )
    
    if (validFiles.length > 0) {
      setResumeFiles(validFiles)
      setError('')
    } else {
      setError('Please upload valid PDF or DOCX files for resumes')
    }
  }

  const handleWeightageChange = (component, value) => {
    const newValue = Math.max(0, Math.min(100, parseInt(value) || 0))
    setWeightage(prev => ({
      ...prev,
      [component]: newValue
    }))
  }

  const validateWeightage = () => {
    const total = Object.values(weightage).reduce((acc, val) => acc + val, 0)
    if (total !== 100) {
      setError(`Weightage must sum to 100%. Current total: ${total}%`)
      return false
    }
    setError('')
    return true
  }

  const handleRunAnalysis = async () => {
    if (!jdFile) {
      setError('Please upload a Job Description file')
      return
    }
    
    if (resumeFiles.length === 0) {
      setError('Please upload resume files')
      return
    }

    if (!validateWeightage()) return

    setLoading(true)
    setError('')
    setSuccess('')
    setResults([])

    try {
      const formData = new FormData()
      formData.append('jd_file', jdFile)
      resumeFiles.forEach(file => {
        formData.append('resume_files', file)
      })
      
      // Send weights as JSON string
      formData.append('weights', JSON.stringify(weightage))
      
      const response = await fetch('http://localhost:8000/api/score', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to analyze resumes')
      }

      const data = await response.json()
      setResults(data.data)
      setSuccess(`Successfully analyzed ${data.data.length} resumes with Industry-Level NLP`)

    } catch (err) {
      setError(err.message || 'An error occurred during analysis')
    } finally {
      setLoading(false)
    }
  }

  const getScoreClass = (score) => {
    if (score >= 0.75) return 'score-high'
    if (score >= 0.5) return 'score-medium'
    return 'score-low'
  }

  const getRankClass = (rank) => {
    if (rank === 1) return 'rank-badge rank-1'
    if (rank === 2) return 'rank-badge rank-2'
    if (rank === 3) return 'rank-badge rank-3'
    return 'rank-badge'
  }

  return (
    <div className="app">
      <header className="header">
        <h1>AI-Based Resume Shortlisting</h1>
        <p>Advanced Industry-Level NLP Pipeline for Academic Recruitment</p>
      </header>

      <main className="main">
        {error && <div className="error-message">⚠️ {error}</div>}
        {success && <div className="success-message">✅ {success}</div>}

        <div className="upload-section">
          <div className="upload-box">
            <h3>Job Description</h3>
            <div className="file-input-wrapper">
              <input
                type="file"
                accept=".pdf,.docx"
                onChange={handleJdUpload}
                className="file-input"
                id="jd-upload"
              />
              <label htmlFor="jd-upload" className="file-input-label">
                {jdFile ? `✓ ${jdFile.name}` : '📄 Choose JD file (PDF/DOCX)'}
              </label>
            </div>
            {jdFile && (
              <div className="selected-files">
                <p>📋 Document Ready</p>
                <ul>
                  <li>File: {jdFile.name}</li>
                  <li>Size: {(jdFile.size / 1024).toFixed(1)} KB</li>
                </ul>
              </div>
            )}
          </div>

          <div className="upload-box">
            <h3>Resume Collection</h3>
            <div className="file-input-wrapper">
              <input
                type="file"
                accept=".pdf,.docx"
                multiple
                onChange={handleResumeUpload}
                className="file-input"
                id="resumes-upload"
              />
              <label htmlFor="resumes-upload" className="file-input-label">
                {resumeFiles.length > 0 
                  ? `✓ ${resumeFiles.length} files selected` 
                  : '📁 Choose resume files (PDF/DOCX)'}
              </label>
            </div>
            {resumeFiles.length > 0 && (
              <div className="selected-files">
                <p>📊 Upload Summary</p>
                <ul>
                  <li>Total Files: {resumeFiles.length}</li>
                  <li>Total Size: {(resumeFiles.reduce((acc, file) => acc + file.size, 0) / 1024).toFixed(1)} KB</li>
                  <li>Status: Ready for analysis</li>
                </ul>
              </div>
            )}
          </div>
        </div>

        <div className="weightage-section">
          <h3>⚖️ Evaluation Weightage (%)</h3>
          <div className="weightage-summary">
            <p>Define how candidates are scored. Total must equal 100%.</p>
          </div>
          <div className="weightage-config">
            <div className="weightage-item">
              <label>Skill Match</label>
              <input
                type="number"
                value={weightage.required_skills}
                onChange={(e) => handleWeightageChange('required_skills', e.target.value)}
                className="weightage-input"
              />
            </div>
            <div className="weightage-item">
              <label>Semantic Fit</label>
              <input
                type="number"
                value={weightage.semantic}
                onChange={(e) => handleWeightageChange('semantic', e.target.value)}
                className="weightage-input"
              />
            </div>
            <div className="weightage-item">
              <label>Reranker</label>
              <input
                type="number"
                value={weightage.reranker}
                onChange={(e) => handleWeightageChange('reranker', e.target.value)}
                className="weightage-input"
              />
            </div>
            <div className="weightage-item">
              <label>Experience</label>
              <input
                type="number"
                value={weightage.experience}
                onChange={(e) => handleWeightageChange('experience', e.target.value)}
                className="weightage-input"
              />
            </div>
            <div className="weightage-item">
              <label>Education</label>
              <input
                type="number"
                value={weightage.education}
                onChange={(e) => handleWeightageChange('education', e.target.value)}
                className="weightage-input"
              />
            </div>
            <div className="weightage-item">
              <label>Projects</label>
              <input
                type="number"
                value={weightage.projects}
                onChange={(e) => handleWeightageChange('projects', e.target.value)}
                className="weightage-input"
              />
            </div>
            <div className="weightage-item">
              <label>Evidence</label>
              <input
                type="number"
                value={weightage.evidence}
                onChange={(e) => handleWeightageChange('evidence', e.target.value)}
                className="weightage-input"
              />
            </div>
          </div>
          <div className={`weightage-total ${Object.values(weightage).reduce((a, b) => a + b, 0) === 100 ? 'valid' : 'invalid'}`}>
            Total: {Object.values(weightage).reduce((a, b) => a + b, 0)}%
          </div>
        </div>

        <div className="run-section">
          <button
            onClick={handleRunAnalysis}
            disabled={loading}
            className="run-button"
          >
            {loading ? '🔄 Extracting & Ranking...' : '🚀 Run Advanced Analysis'}
          </button>
        </div>

        {results.length > 0 && (
          <div className="results-section">
            <h2>📊 Academic Ranking & AI Insights</h2>
            <div className="table-container">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Candidate</th>
                    <th>Final Score</th>
                    <th>AI Explanation</th>
                    <th>Skill Match</th>
                    <th>Semantic Fit</th>
                    <th>Reranker</th>
                    <th>Experience</th>
                    <th>Education</th>
                    <th>Projects</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((result, index) => (
                    <tr key={index}>
                      <td>
                        <span className={getRankClass(index + 1)}>
                          #{index + 1}
                        </span>
                      </td>
                      <td>
                        <strong>{result.name}</strong>
                        <div className="summary-text">{result.semantic_summary}</div>
                      </td>
                      <td className={getScoreClass(result.final_score)}>
                        {(result.final_score * 100).toFixed(1)}%
                      </td>
                      <td className="explanation-cell">
                        {result.explanation}
                      </td>
                      <td className={getScoreClass(result.required_skill_score)}>
                        {(result.required_skill_score * 100).toFixed(1)}%
                      </td>
                      <td className={getScoreClass(result.semantic_score)}>
                        {(result.semantic_score * 100).toFixed(1)}%
                      </td>
                      <td className={getScoreClass(result.reranker_score)}>
                        {(result.reranker_score * 100).toFixed(1)}%
                      </td>
                      <td>{result.experience_years} yrs</td>
                      <td className={getScoreClass(result.education_score)}>
                        {(result.education_score * 100).toFixed(1)}%
                      </td>
                      <td className={getScoreClass(result.project_score)}>
                        {(result.project_score * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

      </main>
    </div>
  )
}

export default App
