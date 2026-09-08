"""
Locator module to locate Schedule of Activities regions inside clinical protocol PDFs.
"""

import re
import subprocess
import fitz # PyMuPDF

SOA_HEADING_PATTERNS = [
    r"schedule of activities",
    r"schedule of assessments",
    r"schedule of procedures",
    r"schedule of events",
    r"schedule of measures",
    r"study flow ?chart",
    r"table of events",
    r"time and events? schedule",
    r"overview of study assessments",
]
SOA_HEADING_RE = re.compile("|".join(SOA_HEADING_PATTERNS), re.IGNORECASE)

ROW_HINT_RE = re.compile(
    r"informed consent|vital signs|physical exam|adverse event|"
    r"concomitant medication|pregnancy test|\becg\b|electrocardiogram|"
    r"pharmacokinetic|\bpk\b|randomi[sz]ation|laboratory|urinalysis|"
    r"screening|baseline|follow-?up|hematology",
    re.IGNORECASE,
)

CONTINUATION_RE = re.compile(r"continued|concluded", re.IGNORECASE)
NEXT_SECTION_RE = re.compile(r"^\s*appendix\s+[ivxlcdm0-9]+|^\s*table\s+\d+\.", re.IGNORECASE)

# Matches a NEW table/appendix heading occurring anywhere on a page, not just
# at the head. Used to catch a second, distinct schedule that starts partway
# down a page we already believe is a continuation of the current SoA (e.g.
# footnotes for Schedule A ending mid-page, then "APPENDIX II: Schedule of
# Blood Collections" beginning below them on the SAME physical page).
EMBEDDED_TABLE_START_RE = re.compile(
    r"^\s*(appendix\s+[ivxlcdm]+\s*[:.]|table\s+\d+[.:]\s)", re.IGNORECASE
)

# Lines that are purely a list marker ("1.", "a.", "3)") with no other
# content on them. PyMuPDF's line-splitter frequently isolates these onto
# their own short line for indented enumerated lists, which would otherwise
# be miscounted as grid-cell-like "short lines" by _is_table_page.
LIST_MARKER_ONLY_RE = re.compile(r"^\s*(\d{1,2}|[a-z])[.\)]\s*$", re.IGNORECASE)
PAGE_NUMBER_ONLY_RE = re.compile(r"^\s*\d{1,4}\s*$")

# A line that IS an enumerated list item (marker + content together, e.g.
# "3. SUI" or "a. BSCS"). A page dense with these is narrative prose
# describing assessments, not a grid -- even though the assessment names
# themselves (SUI, BSCS, ECG...) also legitimately appear as row labels in
# a real SoA grid, which is what makes this page type easy to false-positive
# on if you only look at short-line ratio and keyword hits.
NUMBERED_LIST_ITEM_RE = re.compile(r"^\s*\d{1,2}\.\s*\S")
LETTERED_LIST_ITEM_RE = re.compile(r"^\s*[a-z]\.\s*\S", re.IGNORECASE)


def _layout_lines(pdf_path: str, page_no_1indexed: int) -> list[str]:
    """
    Returns the page's text via `pdftotext -layout`, which reconstructs
    word order more reliably than PyMuPDF's get_text() on some older PDFs
    (glyph positioning can otherwise fragment a heading across lines).
    Used only for the embedded-heading check, to avoid a subprocess call
    on every page.
    """
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-f", str(page_no_1indexed), "-l", str(page_no_1indexed), pdf_path, "-"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return result.stdout.splitlines()
    except Exception:
        return []


def _find_embedded_heading(pdf_path: str, page_no_1indexed: int, skip_lines: int = 3) -> str | None:
    """
    Scans a page (via the more reliable pdftotext-layout text) for a second
    table/appendix heading appearing after the first few lines. Returns the
    matched heading text, or None. `skip_lines` avoids matching the page's
    OWN leading heading (already handled separately by _is_valid_heading /
    NEXT_SECTION_RE at the head of the page).
    """
    lines = _layout_lines(pdf_path, page_no_1indexed)
    for line in lines[skip_lines:]:
        stripped = line.strip()
        if not stripped:
            continue
        if EMBEDDED_TABLE_START_RE.match(stripped):
            return stripped
    return None
FOOTNOTE_CONTINUATION_RE = re.compile(r"notes on the schedule|footnotes? to (the )?(flow ?chart|schedule|table)", re.IGNORECASE)
PROSE_REFERENCE_RE = re.compile(r"provides an overview|will be conducted|is shown|as shown|described in|presented in|outlined below|see\s+(table|schedule|appendix)", re.IGNORECASE)


def _is_valid_heading(page_text: str) -> bool:
    """
    Returns True if an early line on the page is an actual SoA heading rather than body prose mentioning the table. 
    By checking line-by-line, it filters out full sentences containing verbs (like "Table 1 provides..."), 
    which previously caused the parser to start a page too early. 
    This function ensures extraction only triggers on actual short, verbless title labels.
    """
    
    lines = [l for l in page_text.splitlines()[:12] if l.strip()]
    joined_text = " ".join(lines)
    for line in lines:
        match = SOA_HEADING_RE.search(line)
        if not match:
            continue
        idx = joined_text.find(line.strip()[:20]) if line.strip() else -1
        window = joined_text[max(0, idx - 40): idx + len(line) + 20] if idx >= 0 else line
        if not PROSE_REFERENCE_RE.search(window):
            return True
    return False


def _is_table_page(page_text: str) -> bool:
    """
    Heuristic for whether a page continues an SoA grid: requires both row
    keywords and a high ratio of short lines. LIST_MARKER_ONLY_RE and
    PAGE_NUMBER_ONLY_RE exclude bare list markers/page numbers from that
    ratio, and NUMBERED_LIST_ITEM_RE/LETTERED_LIST_ITEM_RE veto pages
    dense with enumerated prose (e.g. "3. SUI", "4. AEs") -- text alone
    can't otherwise distinguish a list item from a real grid row label
    with the same short text (see README: known issues).
    """

    hint_count = len(ROW_HINT_RE.findall(page_text))
    lines = [l for l in page_text.splitlines() if l.strip()]
    if not lines:
        return False

    filtered_short = [
        l for l in lines
        if len(l.strip()) <= 6
        and not LIST_MARKER_ONLY_RE.match(l.strip())
        and not PAGE_NUMBER_ONLY_RE.match(l.strip())
    ]
    ratio = len(filtered_short) / len(lines)

    list_item_lines = sum(
        1 for l in lines if NUMBERED_LIST_ITEM_RE.match(l) or LETTERED_LIST_ITEM_RE.match(l)
    )
    list_item_ratio = list_item_lines / len(lines)

    return hint_count >= 3 and ratio > 0.2 and list_item_ratio <= 0.10


def _matched_heading_line(page_text: str) -> str | None:
    """
    Returns the actual line matched against SOA_HEADING_RE, rather than
    the page's first non-blank line -- the latter can be running header/
    footer text that prints before the real heading on continuation pages,
    which matters since this label is shown next to the table in the UI.
    """
    lines = [l for l in page_text.splitlines()[:12] if l.strip()]
    joined_text = " ".join(lines)
    for line in lines:
        match = SOA_HEADING_RE.search(line)
        if not match:
            continue
        idx = joined_text.find(line.strip()[:20]) if line.strip() else -1
        window = joined_text[max(0, idx - 40): idx + len(line) + 20] if idx >= 0 else line
        if not PROSE_REFERENCE_RE.search(window):
            return line.strip()
    return None


def _locate_from(doc, pdf_path, start_idx, max_span, heading_text, source, claimed_pages, regions):
    """
    Builds one region starting at start_idx, appends it to `regions`, and --
    if _walk_forward finds a second table heading embedded partway down the
    region's last page -- recurses to build a SEPARATE region starting at
    that same page, so two distinct tables sharing one physical page never
    get merged into a single schema. Both regions record the shared page
    and each other's heading in "shares_page_with" so the extractor can warn
    the model instead of silently blending them.
    """
    end_idx, embedded_heading = _walk_forward(doc, pdf_path, start_idx, max_span)
    for p in range(start_idx, end_idx + 1):
        claimed_pages.add(p)

    region = {
        "start_page": start_idx + 1,
        "end_page": end_idx + 1,
        "heading_text": heading_text,
        "source": source,
    }
    regions.append(region)

    if embedded_heading:
        split_page = end_idx + 1  # 1-indexed page shared by both regions
        sub_region_start = len(regions)  # index where the new region will land
        _locate_from(
            doc, pdf_path, end_idx, max_span,
            embedded_heading, "embedded_split",
            claimed_pages, regions,
        )
        other_region = regions[sub_region_start]
        region["shares_page_with"] = {"page": split_page, "other_heading": other_region["heading_text"]}
        other_region["shares_page_with"] = {"page": split_page, "other_heading": region["heading_text"]}


def find_soa_regions(pdf_path: str, max_span: int = 8) -> list[dict]:
    """
    Returns a list of candidate SoA regions, each:
        {
            "start_page": int (1-indexed),
            "end_page": int (1-indexed, inclusive),
            "heading_text": str,
            "source": "toc" | "body_heading" | "embedded_split",
            "shares_page_with": {"page": int, "other_heading": str}  # optional
        }
    Multiple entries indicate that several distinct SoA tables were detected
    (a main schedule plus a sub-study/PK-sampling/extension schedule, or two
    tables that happen to share one physical page -- see "embedded_split").
    Because this code execution path receives limited testing, check the module docstring for important context.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    regions = []
    claimed_pages = set() # page numbers (0-indexed) already assigned to a region

    # --- Pass 1: TOC / bookmarks ---
    toc = doc.get_toc()
    for _, title, page_no in toc:
        if SOA_HEADING_RE.search(title):
            start_idx = page_no - 1
            if start_idx < 0 or start_idx >= total_pages or start_idx in claimed_pages:
                continue
            _locate_from(doc, pdf_path, start_idx, max_span, title.strip(), "toc", claimed_pages, regions)

    # --- Pass 2: body heading scan (catches docs with no/incomplete TOC,
    #     e.g. protocol12 in our test set, whose SoA has no TOC entry) ---
    for i in range(total_pages):
        if i in claimed_pages:
            continue
        text = doc[i].get_text()
        if _is_valid_heading(text):
            matched = _matched_heading_line(text) or "(heading line not re-identified)"
            _locate_from(doc, pdf_path, i, max_span, matched, "body_heading", claimed_pages, regions)

    doc.close()
    regions.sort(key=lambda r: r["start_page"])
    return regions


def _walk_forward(doc, pdf_path: str, start_idx: int, max_span: int) -> tuple[int, str | None]:
    """
    Starting at start_idx (0-indexed, already known to be part of an SoA),
    walk forward while subsequent pages still look like the same table.
    Stops at the first page that doesn't, or after max_span pages as a
    safety cap (protocols in our test set maxed out at a 4-page span).

    Returns (end_idx, embedded_heading). embedded_heading is non-None when a
    page we're including as a continuation (e.g. because it carries this
    table's footnote overflow) ALSO contains a second table/appendix heading
    further down the same physical page. That page is still included here
    (its top portion genuinely belongs to this region), but the caller uses
    embedded_heading to start a SEPARATE region at the same page rather
    than silently folding the second table's rows into this one.
    """
    end_idx = start_idx
    embedded_heading = None
    total_pages = len(doc)
    for i in range(start_idx + 1, min(start_idx + max_span, total_pages)):
        text = doc[i].get_text()
        head = text[:200]
        if NEXT_SECTION_RE.search(head) and not CONTINUATION_RE.search(head):
            break
        if _is_table_page(text) or CONTINUATION_RE.search(text[:300]) or FOOTNOTE_CONTINUATION_RE.search(text[:300]):
            end_idx = i
            embedded_heading = _find_embedded_heading(pdf_path, i + 1)
            if embedded_heading:
                # This page is the last one for the current region -- don't
                # keep walking into what is visually a different table.
                break
        else:
            break
    return end_idx, embedded_heading

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python locator.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    result = find_soa_regions(path)
    print(json.dumps(result, indent=2))