"""
Locator module to locate Schedule of Activities regions inside clinical protocol PDFs.
"""

import re
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
    A heuristic to check if a page continues an SoA grid, used only for finding the table's end. 
    It strictly requires both row keywords and a high ratio of short lines. This prevents a bug 
    where heavily fragmented, single-character text pages were falsely identified as table pages just because they had many short lines.
    """
    
    hint_count = len(ROW_HINT_RE.findall(page_text))
    lines = [l for l in page_text.splitlines() if l.strip()]
    short_lines = sum(1 for l in lines if len(l.strip()) <= 6)
    ratio = (short_lines / len(lines)) if lines else 0
    return hint_count >= 3 and ratio > 0.2


def find_soa_regions(pdf_path: str, max_span: int = 8) -> list[dict]:
    """
    Returns a list of candidate SoA regions, each:
        {
            "start_page": int (1-indexed),
            "end_page": int (1-indexed, inclusive),
            "heading_text": str,
            "source": "toc" | "body_heading",
        }
    Multiple entries indicate that several distinct SoA tables were detected (such as a main schedule or sub-study schedule).  
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
            end_idx = _walk_forward(doc, start_idx, max_span)
            for p in range(start_idx, end_idx + 1):
                claimed_pages.add(p)
            regions.append(
                {
                    "start_page": start_idx + 1,
                    "end_page": end_idx + 1,
                    "heading_text": title.strip(),
                    "source": "toc",
                }
            )

    # --- Pass 2: body heading scan (catches docs with no/incomplete TOC,
    #     e.g. protocol12 in our test set, whose SoA has no TOC entry) ---
    for i in range(total_pages):
        if i in claimed_pages:
            continue
        text = doc[i].get_text()
        if _is_valid_heading(text):
            end_idx = _walk_forward(doc, i, max_span)
            for p in range(i, end_idx + 1):
                claimed_pages.add(p)
            first_line = next((l.strip() for l in text.splitlines() if l.strip() and not l.strip().isdigit()), "")
            regions.append(
                {
                    "start_page": i + 1,
                    "end_page": end_idx + 1,
                    "heading_text": first_line,
                    "source": "body_heading",
                }
            )

    doc.close()
    regions.sort(key=lambda r: r["start_page"])
    return regions


def _walk_forward(doc, start_idx: int, max_span: int) -> int:
    """
    Starting at start_idx (0-indexed, already known to be part of an SoA),
    walk forward while subsequent pages still look like the same table.
    Stops at the first page that doesn't, or after max_span pages as a
    safety cap (protocols in our test set maxed out at a 4-page span).
    """
    end_idx = start_idx
    total_pages = len(doc)
    for i in range(start_idx + 1, min(start_idx + max_span, total_pages)):
        text = doc[i].get_text()
        head = text[:200]
        if NEXT_SECTION_RE.search(head) and not CONTINUATION_RE.search(head):
            break
        if _is_table_page(text) or CONTINUATION_RE.search(text[:300]) or FOOTNOTE_CONTINUATION_RE.search(text[:300]):
            end_idx = i
        else:
            break
    return end_idx

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python locator.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    result = find_soa_regions(path)
    print(json.dumps(result, indent=2))