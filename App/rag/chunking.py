import re

import pdfplumber

from rag.hashing import content_id
from rag.schemas import Chunk, ChunkMeta


SECTION_VOCAB = [
    ("work experience", "experience"),
    ("relevant coursework", "coursework"),
    ("technical skills", "skills"),
    ("academic achievements", "achievements"),
    ("leadership", "leadership"),
    ("extracurricular", "extracurricular"),
    ("experience", "experience"),
    ("projects", "projects"),
    ("education", "education"),
    ("skills", "skills"),
    ("summary", "skills"),
    ("coursework", "coursework"),
    ("certificates", "certificates"),
    ("achievements", "achievements"),
    ("achievement", "achievements"),
    ("publications", "publications"),
    ("awards", "awards"),
    # certifications/certification are distinct credentials from
    # certificates/certificate, so map them onto the same credential label.
    ("certifications", "certificates"),
    ("certification", "certificates"),
]

BULLET_CHARS = "\u2022\u2023\u25e6"
LINE_TOL = 4.0
_TRIMS = " \t\r\n" + BULLET_CHARS

# Leading-line list markers, in addition to the \u2022-family BULLET_CHARS.
# Detected as a LINE-STARTING token (not anywhere in a line) so that hyphens in
# date ranges ("Jan 2023 - Mar 2024") or digits mid-sentence are never mistaken
# for bullets (see _line_first_word_indices + _is_bullet_marker).
_BULLET_LEAD_TOKENS = ("-", "*", "\u25aa", "\u25cb")
_NUMBER_BULLET_RE = re.compile(r"^(?:[0-9]{1,2}[\.\)]|\([0-9]{1,2}\))$")


def _is_bullet_marker(token):
    return token in _BULLET_LEAD_TOKENS or bool(_NUMBER_BULLET_RE.match(token))


class ExtractionError(Exception):
    pass


def _token_count(text):
    return len(text.split())


def _trim_offsets(text, base):
    lead = len(text) - len(text.lstrip(_TRIMS))
    trail = len(text) - len(text.rstrip(_TRIMS))
    return (base + lead, base + len(text) - trail)


def _token_ranges(text, base):
    ranges = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        st = i
        while i < n and text[i] not in " \t\r\n":
            i += 1
        if i > st:
            ranges.append((base + st, base + i))
    return ranges


# --------------------------------------------------------------------------- #
# Section detection
# --------------------------------------------------------------------------- #
def _cluster_lines(words):
    lines = []
    for w in words:
        col = w.get("col", 0)
        placed = False
        for L in lines:
            if L["col"] == col and abs(w["doctop"] - L["top"]) <= LINE_TOL:
                L["words"].append(w)
                n = len(L["words"])
                L["top"] = L["top"] * (n - 1) / n + w["doctop"] / n
                placed = True
                break
        if not placed:
            lines.append({"top": w["doctop"], "col": col, "words": [w]})
    for L in lines:
        L["words"].sort(key=lambda w: w["x0"])
    lines.sort(key=lambda L: L["top"])
    return lines


# Conservative two-column detection. A page is treated as two-column only when
# its word start-positions (x0) split into two bands separated by a large,
# largely-empty gap with a non-trivial number of tokens on BOTH sides. Single
# column resumes (indented bullets, right-aligned dates, sub-items) keep their
# largest internal x0 gap small (~20pt), far below the threshold, so they are
# never misdetected.
COLUMN_GAP_THRESHOLD = 100.0
COLUMN_MIN_SIDE_TOKENS = 15


def _detect_columns(page_words):
    if len(page_words) < 2:
        return "single", None
    x0s = sorted(w["x0"] for w in page_words)
    best_gap = 0.0
    best_a = best_b = None
    for a, b in zip(x0s, x0s[1:]):
        gap = b - a
        if gap > best_gap:
            best_gap = gap
            best_a, best_b = a, b
    if best_gap < COLUMN_GAP_THRESHOLD:
        return "single", None
    left = sum(1 for x in x0s if x < best_a)
    right = sum(1 for x in x0s if x > best_b)
    if left < COLUMN_MIN_SIDE_TOKENS or right < COLUMN_MIN_SIDE_TOKENS:
        return "single", None
    return "two", (best_a + best_b) / 2.0


def _median_body_height(words):
    hs = sorted(w["height"] for w in words if w["height"] < 20)
    if not hs:
        return 10.0
    return float(hs[len(hs) // 2])


def _header_label(ws, med):
    if len(ws) > 6:
        return None
    text = " ".join(w["text"] for w in ws)
    norm = re.sub(r"\s+", " ", text).lower().strip(" " + BULLET_CHARS + "-")
    if text.strip().startswith(BULLET_CHARS):
        return None
    if not re.search(r"[a-z]", norm):
        return None

    for keyword, label in SECTION_VOCAB:
        if re.search(r"\b" + re.escape(keyword) + r"\b", norm):
            return label

    if len(ws) <= 4:
        x0 = [w["x0"] for w in ws]
        if max(x0) < 60 and min(x0) < 30 and max(w["height"] for w in ws) >= med + 1.0:
            if re.match(r"^[A-Za-z][A-Za-z\s/\-&]*$", norm):
                return "other"

    return None


def detect_sections(words):
    """Return non-empty (section_type, start_idx, end_idx) word-index segments."""
    total = len(words)
    if total == 0:
        return []

    med = _median_body_height(words)
    lines = _cluster_lines(words)
    pos = {id(w): i for i, w in enumerate(words)}

    headers = []
    for L in lines:
        ws = L["words"]
        label = _header_label(ws, med)
        if label is not None:
            headers.append((label, pos[id(ws[0])], pos[id(ws[-1])]))

    if not headers:
        return [("other", 0, total)]

    segments = []
    first_start = headers[0][1]
    if first_start > 0:
        segments.append(("other", 0, first_start))

    for k, (label, first, last) in enumerate(headers):
        start = last + 1
        end = headers[k + 1][1] if k + 1 < len(headers) else total
        if start < end:
            segments.append((label, start, end))

    return segments


# --------------------------------------------------------------------------- #
# Text-level utilities (string domain)
# --------------------------------------------------------------------------- #
def split_bullets(section_text, bullet_starts=None):
    if bullet_starts:
        starts = [s for s in bullet_starts if 0 < s < len(section_text)]
        boundaries = [0] + sorted(starts) + [len(section_text)]
        parts = [
            section_text[a:b].strip(_TRIMS)
            for a, b in zip(boundaries, boundaries[1:])
        ]
        parts = [p for p in parts if p]
        if len(parts) > 1:
            return parts
    parts = re.split("[" + BULLET_CHARS + "]", section_text)
    parts = [p.strip(_TRIMS) for p in parts if p.strip(_TRIMS)]
    if len(parts) > 1:
        return parts
    parts = re.split(r"[,\n]", section_text)
    parts = [p.strip(_TRIMS) for p in parts if p.strip(_TRIMS)]
    if parts:
        return parts
    cleaned = section_text.strip(_TRIMS)
    return [cleaned] if cleaned else []


def merge_to_target(bullets, min_tokens=40, max_tokens=120):
    if not bullets:
        return []

    expanded = []
    for b in bullets:
        if _token_count(b) <= max_tokens:
            expanded.append(b)
            continue
        toks = b.split()
        cur = []
        cnt = 0
        for t in toks:
            if cnt + 1 <= max_tokens:
                cur.append(t)
                cnt += 1
            else:
                expanded.append(" ".join(cur))
                cur = [t]
                cnt = 1
        if cur:
            expanded.append(" ".join(cur))

    merged = []
    cur = []
    cnt = 0
    for b in expanded:
        t = _token_count(b)
        if not cur:
            cur = [b]
            cnt = t
        elif cnt + t <= max_tokens:
            cur.append(b)
            cnt += t
        else:
            merged.append(" ".join(cur).strip(_TRIMS))
            cur = [b]
            cnt = t
    if cur:
        merged.append(" ".join(cur).strip(_TRIMS))
    return merged


# Intentionally not yet called from chunk() or ingest() — reserved for the
# embedding/retrieval string-construction step in a later phase.
def prefix_context(chunk_text, section_type, org=None, dates=None):
    prefix = section_type.upper()
    middle = org or ""
    if dates:
        if middle:
            middle += " (" + dates + ")"
        else:
            middle = dates
    if middle:
        prefix += " \u203a " + middle
    prefix += " \u203a "
    return prefix + chunk_text


# --------------------------------------------------------------------------- #
# Range-domain splitting/merging (offset-preserving)
# --------------------------------------------------------------------------- #
def _split_ranges_full(text, base, bullet_starts=None):
    if bullet_starts:
        starts = sorted(s for s in bullet_starts if 0 < s < len(text))
        boundaries = [0] + starts + [len(text)]
        ranges = []
        for a, b in zip(boundaries, boundaries[1:]):
            seg = text[a:b]
            if seg.strip(_TRIMS):
                ranges.append(_trim_offsets(seg, base + a))
        if len(ranges) > 1:
            return ranges

    ranges = []
    last = 0
    for m in re.finditer("[" + BULLET_CHARS + "]", text):
        seg = text[last:m.start()]
        if seg.strip(_TRIMS):
            ranges.append(_trim_offsets(seg, base + last))
        last = m.end()
    tail = text[last:]
    if tail.strip(_TRIMS):
        ranges.append(_trim_offsets(tail, base + last))

    if len(ranges) > 1:
        return ranges

    ranges = []
    last = 0
    for m in re.finditer(r"[,\n]", text):
        seg = text[last:m.start()]
        if seg.strip(_TRIMS):
            ranges.append(_trim_offsets(seg, base + last))
        last = m.end()
    tail = text[last:]
    if tail.strip(_TRIMS):
        ranges.append(_trim_offsets(tail, base + last))

    if ranges:
        return ranges

    return [_trim_offsets(text, base)]


def _split_sentences_ranges(text, base, window=3, overlap=1):
    segs = []
    last = 0
    for m in re.finditer(r"(?<=[.!?])\s+|\n+", text):
        if m.start() > last and text[last:m.start()].strip(_TRIMS):
            segs.append((last, m.start()))
        last = m.end()
    if last < len(text) and text[last:].strip(_TRIMS):
        segs.append((last, len(text)))
    if not segs:
        return [_trim_offsets(text, base)]

    step = max(1, window - overlap)
    result = []
    for i in range(0, len(segs), step):
        chunk = segs[i:i + window]
        s = chunk[0][0]
        e = chunk[-1][1]
        if text[s:e].strip(_TRIMS):
            result.append(_trim_offsets(text[s:e], base + s))
    return result


def _split_long_range(full_text, s, e, max_tokens):
    tok = _token_ranges(full_text[s:e], s)
    groups = []
    cur_s = cur_e = None
    cnt = 0
    for st, en in tok:
        if cur_s is None:
            cur_s, cur_e, cnt = st, en, 1
        elif cnt + 1 <= max_tokens:
            cur_e, cnt = en, cnt + 1
        else:
            groups.append((cur_s, cur_e))
            cur_s, cur_e, cnt = st, en, 1
    if cur_s is not None:
        groups.append((cur_s, cur_e))
    return groups


def _cap_range(full_text, s, e, max_tokens):
    if _token_count(full_text[s:e]) <= max_tokens:
        return [(s, e)]
    return _split_long_range(full_text, s, e, max_tokens)


def _merge_ranges_full(full_text, ranges, min_tokens=40, max_tokens=120):
    expanded = []
    for s, e in ranges:
        if _token_count(full_text[s:e]) <= max_tokens:
            expanded.append((s, e))
        else:
            expanded.extend(_split_long_range(full_text, s, e, max_tokens))

    merged = []
    cur_s = cur_e = None
    cur_tokens = 0
    for s, e in expanded:
        t = _token_count(full_text[s:e])
        if cur_s is None:
            cur_s, cur_e, cur_tokens = s, e, t
        elif cur_tokens + t <= max_tokens:
            cur_e, cur_tokens = e, cur_tokens + t
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e, cur_tokens = s, e, t
    if cur_s is not None:
        merged.append((cur_s, cur_e))
    return merged


# --------------------------------------------------------------------------- #
# Chunker
# --------------------------------------------------------------------------- #
class ResumeChunker:
    chunker_version = "v1"
    embed_model = "sentence-transformers/all-mpnet-base-v2"
    min_tokens = 40
    max_tokens = 120

    def __init__(self):
        self.full_text = ""
        self.candidate_id = ""
        self.chunks = []
        self.layout_notes = []

    def _extract_words(self, file_path):
        self.layout_notes = []
        pages_words = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                pages_words.append(page.extract_words())

        words = []
        for pageno, ws in enumerate(pages_words):
            mode, boundary = _detect_columns(ws)
            if mode == "two":
                self.layout_notes.append(
                    "two-column layout detected on page %d" % pageno
                )
                for w in ws:
                    w["col"] = 0 if w["x0"] < boundary else 1
                left = [w for w in ws if w["col"] == 0]
                right = [w for w in ws if w["col"] == 1]
                left.sort(key=lambda w: (w["doctop"], w["x0"]))
                right.sort(key=lambda w: (w["doctop"], w["x0"]))
                words.extend(left)
                words.extend(right)
            else:
                words.extend(ws)

        if not words:
            raise ExtractionError(
                "No text extracted: the PDF appears to be scanned/image-based."
            )

        if not self.layout_notes:
            # Single-column PDFs keep the original global reading order so the
            # existing golden chunks are unaffected.
            words.sort(key=lambda w: (w["doctop"], w["x0"]))
        return words

    def _line_ids(self, words):
        line_id = {}
        for li, L in enumerate(_cluster_lines(words)):
            for w in L["words"]:
                line_id[id(w)] = li
        return line_id

    def _line_first_word_indices(self, words):
        pos = {id(w): i for i, w in enumerate(words)}
        first = set()
        for L in _cluster_lines(words):
            L["words"].sort(key=lambda w: w["x0"])
            first.add(pos[id(L["words"][0])])
        return first

    def _build_full_text(self, words):
        line_id = self._line_ids(words)
        self._line_id = line_id
        full = ""
        offsets = []
        pos = 0
        prev = None
        for w in words:
            if prev is not None:
                sep = "\n" if line_id[id(prev)] != line_id[id(w)] else " "
                full += sep
                pos += len(sep)
            offsets.append(pos)
            full += w["text"]
            pos += len(w["text"])
            prev = w
        ends = [offsets[i] + len(words[i]["text"]) for i in range(len(words))]
        return full, offsets, ends

    def _split_section(self, label, text, base, bullet_starts=None):
        if label == "other":
            return _split_sentences_ranges(text, base, window=3, overlap=1)
        return _split_ranges_full(text, base, bullet_starts)

    def chunk(self, file_path):
        words = self._extract_words(file_path)
        self.full_text, offsets, ends = self._build_full_text(words)
        self.candidate_id = content_id(self.full_text)
        sections = detect_sections(words)

        id_prefix = self.candidate_id[:8]
        chunks = []
        line_first = self._line_first_word_indices(words)
        for label, start, end in sections:
            if start >= end:
                continue
            region_text = self.full_text[offsets[start]:ends[end - 1]]
            base = offsets[start]
            bullet_starts = []
            for i in range(start, end):
                if i in line_first and _is_bullet_marker(words[i]["text"]):
                    rel = offsets[i] - base
                    if 0 <= rel < len(region_text):
                        bullet_starts.append(rel)
            ranges = self._split_section(label, region_text, base, bullet_starts)
            for s, e in _merge_ranges_full(
                self.full_text, ranges, self.min_tokens, self.max_tokens
            ):
                for s2, e2 in _cap_range(self.full_text, s, e, self.max_tokens):
                    text = self.full_text[s2:e2]
                    s3, e3 = _trim_offsets(text, s2)
                    t3 = self.full_text[s3:e3]
                    chunk_id = f"{id_prefix}:{len(chunks):02d}"
                    meta = ChunkMeta(
                        candidate_id=self.candidate_id,
                        chunk_id=chunk_id,
                        section_type=label,
                        chunker_version=self.chunker_version,
                        embed_model=self.embed_model,
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            candidate_id=self.candidate_id,
                            raw_text=t3,
                            char_start=s3,
                            char_end=e3,
                            meta=meta,
                        )
                    )

        self.chunks = chunks
        return chunks
