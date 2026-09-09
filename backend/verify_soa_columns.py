"""
verify_soa_columns.py

A lightweight, INDEPENDENT cross-check for SoA extraction output.

Problem this addresses:
    The extractor (extractor.py) relies entirely on a vision LLM (Gemini) to
    look at a page IMAGE and report which column each "X" belongs to. There
    is no grounding in the PDF's actual geometry, so the model can
    (and does) occasionally misjudge which column a mark sits under --
    especially on sparse rows with few anchor points nearby.

What this script does instead:
    1. Reads the PDF's real text layer with PyMuPDF (page.get_text("words")),
       which gives the exact (x0, y0, x1, y1) bounding box of every token --
       including every literal "X" character -- independent of any model's
       visual judgement.
    2. Finds the header line containing visit-number labels (e.g. "1", "2",
       ..., "13", "ET", "RT") and uses each label's x-center as a column
       anchor.
    3. For each row of body text, buckets every "X"-like token into the
       nearest column anchor by x-position.
    4. Compares that geometrically-derived set of "which columns have an X"
       against what the JSON output says for the same row (fuzzy-matched by
       row label).
    5. Reports disagreements. It does NOT silently "fix" the JSON -- a
       geometric heuristic has its own failure modes (multi-line row labels,
       footnote superscripts glued to an "X", non-"X" cell values like "P",
       "3X", "Q2W" which this script does not attempt to classify). Treat
       output as a prioritized list of rows to manually re-check against the
       source PDF, not as ground truth to auto-apply.

Usage:
    python verify_soa_columns.py <path_to_protocol.pdf> <path_to_soa.json> \
        [--start-page N] [--end-page N]

    If --start-page/--end-page are omitted, the script uses the JSON's own
    "source_pages" field (written by extractor.py) for each extracted region.

Known limitations (read before trusting output):
    - Only tokens that are exactly "X" or "X" followed by 1-2 letters/digits
      (superscript footnote markers, e.g. "Xa", "Xb") are treated as
      "X marks". Other real cell values ("P", "3X", "Q2W", "(X)", doses,
      arrows) are NOT detected by this script and those rows are skipped
      with a note -- this script does not attempt to be a general cell-value
      parser, only an X/column-alignment sanity check.
    - Row-label matching is fuzzy (difflib) and can mismatch on very short
      or very similar row labels (e.g. "ADAS-Cog" vs "ADAS-Cog subscale").
      Matches below the similarity threshold are skipped and reported as
      "unmatched" rather than guessed.
    - Column anchors are taken from whichever line contains the most
      recognizable visit-style labels on a given page. Continuation pages
      with no repeated header cannot be verified this way and are flagged
      as such -- this is a real gap, not a silent skip.
    - This is a heuristic SECOND opinion, not a replacement for the
      extractor. Where it disagrees with the JSON, that means "look at this
      row by eye," not "the JSON is wrong" or "the script is right."
"""

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass, field

import fitz  # PyMuPDF

X_MARK_RE = re.compile(r"^X[a-zA-Z0-9]{0,2}$")
VISIT_LABEL_RE = re.compile(r"^(ET|RT|\d{1,2})$", re.IGNORECASE)
Y_LINE_TOLERANCE = 4.0  # points; words within this y-center distance are "same line"
ROW_LABEL_MATCH_THRESHOLD = 0.55  # difflib ratio cutoff for fuzzy row-label matching


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def yc(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class ColumnAnchor:
    label: str
    xc: float


@dataclass
class RowCheckResult:
    row_id: str
    row_label: str
    matched_pdf_line: str | None
    json_columns_with_x: set
    pdf_columns_with_x: set
    status: str  # "match" | "mismatch" | "unmatched" | "skipped_non_x_values"
    detail: str = ""


def get_words(page) -> list[Word]:
    raw = page.get_text("words")  # (x0, y0, x1, y1, word, block, line, word_no)
    return [Word(x0=w[0], y0=w[1], x1=w[2], y1=w[3], text=w[4]) for w in raw]


def cluster_into_lines(words: list[Word]) -> list[list[Word]]:
    """Groups words into visual lines by y-center proximity, then sorts each
    line left-to-right. Simple greedy clustering -- adequate for SoA tables,
    which have well-separated row heights (no rotated/overlapping text)."""
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: w.yc)
    lines: list[list[Word]] = [[words_sorted[0]]]
    for w in words_sorted[1:]:
        if abs(w.yc - lines[-1][-1].yc) <= Y_LINE_TOLERANCE:
            lines[-1].append(w)
        else:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w.x0)
    return lines


def find_column_anchors(lines: list[list[Word]]) -> list[ColumnAnchor] | None:
    """
    Scans lines top-to-bottom for the one most likely to be the visit-number
    header row: the line with the highest count of tokens matching
    VISIT_LABEL_RE. Returns None if no line has at least 3 such tokens
    (arbitrary floor -- fewer than that isn't a credible header row and the
    page likely has no repeated header at all, e.g. many continuation pages).
    """
    best_line = None
    best_count = 0
    for line in lines:
        labels = [w for w in line if VISIT_LABEL_RE.match(w.text)]
        if len(labels) > best_count:
            best_count = len(labels)
            best_line = labels
    if best_line is None or best_count < 3:
        return None
    return [ColumnAnchor(label=w.text, xc=w.xc) for w in best_line]


def nearest_anchor(xc: float, anchors: list[ColumnAnchor]) -> ColumnAnchor:
    return min(anchors, key=lambda a: abs(a.xc - xc))


def find_row_label_line(
    lines: list[list[Word]], header_yc: float, row_label: str
) -> list[Word] | None:
    """
    Finds the line below the header whose leading text best fuzzy-matches
    row_label. Only considers words positioned left of where data columns
    start (heuristically: left of the leftmost column anchor) as candidate
    label text, since row labels sit in the leftmost table column.
    """
    candidates = []
    for line in lines:
        if line[0].yc <= header_yc:
            continue
        leading_text = " ".join(w.text for w in line if not X_MARK_RE.match(w.text))
        if not leading_text.strip():
            continue
        ratio = difflib.SequenceMatcher(None, leading_text.lower(), row_label.lower()).ratio()
        # Also credit a strong prefix match (row labels sometimes wrap to a
        # second line that this simple line-clustering won't stitch back
        # together, so we only require the START of the label to match).
        prefix_ratio = difflib.SequenceMatcher(
            None, leading_text[: len(row_label)].lower(), row_label.lower()
        ).ratio()
        score = max(ratio, prefix_ratio)
        if score >= ROW_LABEL_MATCH_THRESHOLD:
            candidates.append((score, line))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    return candidates[0][1]


def find_row_label_anchor_line(
    lines: list[list[Word]], header_yc: float, row_label: str
) -> list[Word] | None:
    """
    Like find_row_label_line, but matches against only the FIRST ~20
    characters of row_label. Long, multi-line row labels (e.g. "CT Scan (if
    not within last year and patient passes all other screens)") get
    visually split across 2-3 physical text lines by cluster_into_lines,
    and a whole-label fuzzy match can latch onto the wrong fragment (the
    middle or last line) instead of the first line -- which is the one
    whose y-position actually defines the row's grid line. Anchoring on a
    short prefix avoids that.
    """
    prefix = row_label[:20].lower()
    candidates = []
    for line in lines:
        if line[0].yc <= header_yc:
            continue
        leading_text = " ".join(w.text for w in line if not X_MARK_RE.match(w.text))
        if not leading_text.strip():
            continue
        score = difflib.SequenceMatcher(None, leading_text[:20].lower(), prefix).ratio()
        if score >= ROW_LABEL_MATCH_THRESHOLD:
            candidates.append((score, line))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    return candidates[0][1]


def gather_page_results(
    page,
    json_columns: list[dict],
    json_rows: list[dict],
) -> tuple[dict, set]:
    """
    For ONE page, returns:
      - a dict {row_id: set(colid, ...)} of X columns found on this page
        (only for rows whose label-line was found here at all)
      - the set of JSON column ids whose visit_label was detected as a
        header anchor on this page (i.e. columns this page can actually
        verify -- rows may span multiple pages with different columns
        each, so callers must UNION this across pages, never overwrite).
    Returns ({}, set()) if this page has no detectable header.
    """
    words = get_words(page)
    lines = cluster_into_lines(words)
    anchors = find_column_anchors(lines)
    if anchors is None:
        return {}, set()

    header_line_yc = None
    for line in lines:
        if any(VISIT_LABEL_RE.match(w.text) and w.text in {a.label for a in anchors} for w in line):
            header_line_yc = line[0].yc
            break
    if header_line_yc is None:
        return {}, set()

    label_to_colid = {c["visit_label"]: c["id"] for c in json_columns if c.get("visit_label")}
    coverable_colids = {label_to_colid[a.label] for a in anchors if a.label in label_to_colid}

    per_row_x: dict = {}
    for row in json_rows:
        # Use a widened line-height search window around the anchor line to
        # tolerate the value sitting on a slightly different sub-line than
        # the matched label fragment for multi-line row labels (see
        # find_row_label_anchor_line's docstring for why the label match
        # itself only anchors on a prefix).
        line = find_row_label_anchor_line(lines, header_line_yc, row["label"])
        if line is None:
            continue
        row_yc = line[0].yc
        window_lines = [
            ln for ln in lines
            if abs(ln[0].yc - row_yc) <= Y_LINE_TOLERANCE * 3  # ~3 line-heights
        ]
        pdf_cols_with_x = set()
        for ln in window_lines:
            for w in ln:
                if X_MARK_RE.match(w.text):
                    anchor = nearest_anchor(w.xc, anchors)
                    colid = label_to_colid.get(anchor.label)
                    if colid:
                        pdf_cols_with_x.add(colid)
        per_row_x[row["id"]] = pdf_cols_with_x

    return per_row_x, coverable_colids


def verify(pdf_path: str, soa_json_path: str) -> None:
    with open(soa_json_path) as f:
        soa_data = json.load(f)

    doc = fitz.open(pdf_path)

    for region_idx, region in enumerate(soa_data):
        parsed = region.get("parsed", {})
        columns = parsed.get("columns", [])
        rows = parsed.get("rows", [])
        source_pages = region.get("source_pages", [])

        if not columns or not rows:
            print(f"\n=== Region {region_idx}: no parsed columns/rows -- skipping ===")
            continue

        heading = parsed.get("heading", "(no heading)")
        print(f"\n=== Region {region_idx}: {heading} (pages {source_pages}) ===")

        pages_with_no_header = []
        found_on_any_page: set = set()  # row_ids whose label was located somewhere
        pdf_x_by_row: dict = {}         # row_id -> set(colid) UNIONED across all pages
        coverable_colids: set = set()   # column ids verifiable on ANY page in this region

        for page_no in source_pages:
            page = doc[page_no - 1]
            per_row_x, page_coverable = gather_page_results(page, columns, rows)
            if not per_row_x and not page_coverable:
                pages_with_no_header.append(page_no)
                continue
            coverable_colids |= page_coverable
            for row_id, cols in per_row_x.items():
                found_on_any_page.add(row_id)
                pdf_x_by_row.setdefault(row_id, set())
                pdf_x_by_row[row_id] |= cols

        all_results: list[RowCheckResult] = []
        for row in rows:
            row_id = row["id"]
            row_label = row["label"]
            if row_id not in found_on_any_page:
                continue  # reported separately below as "not found on any page"

            json_values = {cid: cell["value"] for cid, cell in row["cells"].items()}
            non_x_values = {v for v in json_values.values() if v not in ("", "X")}
            if non_x_values:
                all_results.append(RowCheckResult(
                    row_id=row_id, row_label=row_label, matched_pdf_line=None,
                    json_columns_with_x=set(), pdf_columns_with_x=set(),
                    status="skipped_non_x_values",
                    detail=f"Row contains non-X values {sorted(non_x_values)} -- "
                           f"this script only cross-checks plain 'X' placement.",
                ))
                continue

            # Only compare columns this region could actually verify somewhere
            # (a column whose header never appeared on any page in range,
            # e.g. page 52 in protocol1 which is a cover page with no table,
            # can't be checked and must not count as a false "extra").
            json_cols_with_x = {cid for cid, v in json_values.items() if v == "X" and cid in coverable_colids}
            pdf_cols_with_x = pdf_x_by_row.get(row_id, set()) & coverable_colids

            if json_cols_with_x == pdf_cols_with_x:
                status, detail = "match", ""
            else:
                status = "mismatch"
                missing_in_json = pdf_cols_with_x - json_cols_with_x
                extra_in_json = json_cols_with_x - pdf_cols_with_x
                parts = []
                if missing_in_json:
                    parts.append(f"PDF shows X in {sorted(missing_in_json)} but JSON does not")
                if extra_in_json:
                    parts.append(f"JSON shows X in {sorted(extra_in_json)} but PDF geometry does not")
                detail = "; ".join(parts)

            all_results.append(RowCheckResult(
                row_id=row_id, row_label=row_label, matched_pdf_line=None,
                json_columns_with_x=json_cols_with_x, pdf_columns_with_x=pdf_cols_with_x,
                status=status, detail=detail,
            ))

        unmatched_rows = [r for r in rows if r["id"] not in found_on_any_page]

        mismatches = [r for r in all_results if r.status == "mismatch"]
        matches = [r for r in all_results if r.status == "match"]
        skipped = [r for r in all_results if r.status == "skipped_non_x_values"]

        print(f"  Rows matched & compared: {len(all_results)} "
              f"({len(matches)} agree, {len(mismatches)} disagree, "
              f"{len(skipped)} skipped -- non-X values)")
        if unmatched_rows:
            print(f"  Rows NOT found on any page via geometry "
                  f"(check manually -- could be a real gap): "
                  f"{[r['label'] for r in unmatched_rows]}")
        if pages_with_no_header:
            print(f"  Pages with no detectable header row (unverifiable by this script): "
                  f"{pages_with_no_header}")

        if mismatches:
            print("\n  --- DISAGREEMENTS (verify these by eye against the source PDF) ---")
            for r in mismatches:
                print(f"  Row '{r.row_label}' ({r.row_id}):")
                print(f"      JSON says X in   : {sorted(r.json_columns_with_x)}")
                print(f"      Geometry says X in: {sorted(r.pdf_columns_with_x)}")
                print(f"      -> {r.detail}")

        if skipped:
            print("\n  --- SKIPPED (non-X values present; not cross-checked by this script) ---")
            for r in skipped:
                print(f"  Row '{r.row_label}' ({r.row_id}): {r.detail}")

    doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_path")
    parser.add_argument("soa_json_path")
    args = parser.parse_args()
    verify(args.pdf_path, args.soa_json_path)
