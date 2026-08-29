import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rag.chunking import (
    _is_bullet_marker,
    _split_ranges_full,
    detect_sections,
    ResumeChunker,
    prefix_context,
)
from rag.ingest import ResumeIngestor

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(TESTS_DIR)
APP_DIR = os.path.dirname(RAG_DIR)
PROJECT_DIR = os.path.dirname(APP_DIR)
RESUME_DIR = os.path.join(PROJECT_DIR, "Resumes")
GOLDEN_DIR = os.path.join(TESTS_DIR, "golden_chunks")

MAX_TOKENS = 160
MIN_COVERAGE = 0.95


def iter_resumes():
    for fname in sorted(os.listdir(RESUME_DIR)):
        if fname.endswith(".pdf"):
            yield fname, os.path.join(RESUME_DIR, fname)


def golden_path(resume_file):
    return os.path.join(GOLDEN_DIR, os.path.splitext(resume_file)[0] + ".json")


def chunk_json(chunks):
    return json.loads("[" + ",".join(ch.model_dump_json() for ch in chunks) + "]")


def test_golden_files_match():
    failures = []
    for fname, path in iter_resumes():
        with open(golden_path(fname)) as f:
            expected = json.load(f)
        chunker = ResumeChunker()
        chunks = chunker.chunk(path)
        actual = chunk_json(chunks)
        if actual != expected:
            failures.append(fname)
            print(f"  GOLDEN MISMATCH: {fname}")
    assert not failures, f"golden mismatch for: {failures}"
    print("GOLDEN OK: chunk output matches saved golden files for all resumes")


def test_coverage_and_invariants():
    low_coverage = []
    all_ok = True
    print("--- coverage per resume (chunked_chars / extracted_chars) ---")
    for fname, path in iter_resumes():
        chunker = ResumeChunker()
        chunks = chunker.chunk(path)
        full_text = chunker.full_text

        chunked = sum(ch.char_end - ch.char_start for ch in chunks)
        coverage = chunked / len(full_text) if full_text else 0.0

        over_limit = [ch for ch in chunks if len(ch.raw_text.split()) > MAX_TOKENS]
        empty_type = [ch for ch in chunks if not ch.meta.section_type]
        bad_off = [ch for ch in chunks if full_text[ch.char_start:ch.char_end] != ch.raw_text]

        problems = []
        if coverage < MIN_COVERAGE:
            low_coverage.append((fname, coverage))
        if over_limit:
            problems.append(f"{len(over_limit)} chunks > {MAX_TOKENS} tokens")
        if empty_type:
            problems.append("empty section_type found")
        if bad_off:
            problems.append(f"{len(bad_off)} chunks with wrong char offsets")

        flag = "  <-- PROBLEM" if problems else ""
        print(f"  {fname:42s} coverage={coverage:.3f} chunks={len(chunks):3d}{flag}")
        if problems:
            all_ok = False
            for p in problems:
                print(f"      {p}")

    assert all_ok, "some invariants failed (see flagged lines above)"
    for fname, cov in low_coverage:
        print(f"  LOW COVERAGE: {fname} = {cov:.3f} < {MIN_COVERAGE}")
    assert not low_coverage, f"coverage below {MIN_COVERAGE} for: {[f for f, _ in low_coverage]}"
    print("INVARIANTS OK: coverage >= %.2f, no chunk > %d tokens, non-empty section_type, exact char offsets for all resumes" % (MIN_COVERAGE, MAX_TOKENS))


def test_prefix_context():
    assert prefix_context("build a dashboard", "projects") == "PROJECTS \u203a build a dashboard"
    assert prefix_context("led x", "experience", org="Acme Corp") == "EXPERIENCE \u203a Acme Corp \u203a led x"
    assert prefix_context("led x", "experience", org="Acme Corp", dates="2023-2024") == "EXPERIENCE \u203a Acme Corp (2023-2024) \u203a led x"
    assert prefix_context("x", "education", dates="2020") == "EDUCATION \u203a 2020 \u203a x"
    print("PREFIX_CONTEXT OK: section-only, section+org, section+org+dates, section+dates")


def test_date_range_vs_numbered_bullet():
    path = os.path.join(TESTS_DIR, "fixtures", "fixture_date_range_bullets.pdf")
    chunker = ResumeChunker()
    words = chunker._extract_words(path)
    full_text, offsets, ends = chunker._build_full_text(words)
    sections = detect_sections(words)
    line_first = chunker._line_first_word_indices(words)

    projects = [sec for sec in sections if sec[0] == "projects"]
    assert len(projects) == 1, "expected a single projects section"
    label, s, e = projects[0]
    region_text = full_text[offsets[s]:ends[e - 1]]
    base = offsets[s]

    bullet_starts = [
        offsets[i] - base
        for i in range(s, e)
        if i in line_first and _is_bullet_marker(words[i]["text"])
    ]
    ranges = _split_ranges_full(region_text, base, bullet_starts)
    assert len(ranges) == 3, "expected exactly 3 numbered-bullet parts, got %d" % len(ranges)

    parts_texts = [full_text[a:b] for a, b in ranges]
    assert "Jan 2023 - Mar 2024" in parts_texts[0], (
        "mid-line date hyphen must survive intact inside part 1, got %r" % parts_texts[0]
    )
    assert "Jan 2023 - Mar 2024" not in parts_texts[1], (
        "date hyphen must not leak into part 2: %r" % parts_texts[1]
    )

    chunks = chunker.chunk(path)
    project_chunks = [ch for ch in chunks if ch.meta.section_type == "projects"]
    assert project_chunks, "expected at least one projects chunk"
    assert "Jan 2023 - Mar 2024" in project_chunks[0].raw_text, (
        "date range must appear intact within a single chunk, got %r" % project_chunks[0].raw_text
    )
    print("DATE_RANGE_BULLET OK: 3 bullet parts; 'Jan 2023 - Mar 2024' intact inside part 1 and one chunk")


def test_two_column_no_false_positive():
    offenders = []
    for fname, path in iter_resumes():
        chunker = ResumeChunker()
        chunker.chunk(path)
        if chunker.layout_notes:
            offenders.append((fname, chunker.layout_notes))
    assert not offenders, "two-column false positives on real resumes: %r" % offenders
    print("TWO_COLUMN_NO_FALSE_POSITIVE OK: layout_notes==[] for all 7 real resumes")


if __name__ == "__main__":
    test_golden_files_match()
    test_coverage_and_invariants()
    test_prefix_context()
    test_date_range_vs_numbered_bullet()
    test_two_column_no_false_positive()
    print("ALL CHUNKING TESTS PASSED")
