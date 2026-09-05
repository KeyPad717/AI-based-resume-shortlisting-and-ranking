// Shared weight definitions used by WeightPanel / RagJobFlow.
// Kept out of the component file for react-refresh friendliness.

export const WEIGHT_KEYS = [
  ['required_skills', 'Skill Match'],
  ['semantic', 'Semantic Fit'],
  ['reranker', 'Reranker'],
  ['experience', 'Experience'],
  ['education', 'Education'],
  ['projects', 'Projects'],
  ['evidence', 'Evidence'],
]

export const DEFAULT_WEIGHTS = {
  required_skills: 30,
  semantic: 15,
  reranker: 15,
  experience: 15,
  education: 10,
  projects: 10,
  evidence: 5,
}
