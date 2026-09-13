"""Test-session isolation for Phase 10 withness.

The Phase 10 LLM dispatcher persists responses to an on-disk content-addressed
cache and appends JSONL call logs for the process lifetime. Both write to real
paths under the working directory, which would (a) leak between tests and (b)
pollute the repo during `pytest`. Disable both for the whole test session and
send them to a session-scoped temp dir just in case a test explicitly opts in.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="ats_rag_test_")

# Hermetic tests: a live OPENROUTER key (App/.env, loaded by pipeline's
# load_dotenv at import) would make call_llm_extraction hit the network and
# diverge from the fallback-based golden baseline. python-dotenv's default
# no-override means this empty value wins over App/.env for the test session.
os.environ.setdefault("OPENROUTER_API_KEY", "")

os.environ.setdefault("LLM_CACHE_ENABLED", "false")
os.environ.setdefault("LLM_LOG_ENABLED", "false")
os.environ.setdefault("LLM_CACHE_DIR", os.path.join(_TMP, "cache"))
os.environ.setdefault("LLM_LOG_PATH", os.path.join(_TMP, "logs", "llm_calls.jsonl"))
os.environ.setdefault("LLM_MAX_CALLS_PER_JOB", "1000")