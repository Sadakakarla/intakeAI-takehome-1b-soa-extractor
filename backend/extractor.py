"""
Extractor module takes and processes a located SoA region (from locator.py) and produces a structured,
machine-readable representation (JSON) of the table.
"""

import base64
import json
import os
import re
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types
import fitz  # PyMuPDF

RENDER_DPI = 200

# GROQ_VISION_MODEL = "qwen/qwen3.6-27b"
GEMINI_MODEL = "gemini-3.6-flash"
MAX_IMAGES_PER_REQUEST = 2

SCHEMA_DESCRIPTION = """
Return ONLY a single JSON object (no markdown fences, no commentary) with
this exact shape:

{
  "h": "<the table's title/heading as printed>",
  "cols": [
    {
      "id": "c1",
      "per": "<MANDATORY: Overarching study period, e.g. 'Screening', 'Treatment', 'Follow-up'. Carry forward from left if implied/merged across columns.>",
      "vis": "<visit name/number as printed, e.g. '1', '2', 'ET'>",
      "day": "<day/week value as printed, e.g. 'Week 4', '-2'>",
      "win": "<allowable visit window as printed, e.g. '\\u00b1 3 days', or null>"
    }
  ],
  
  "grp": [
    {
      "lbl": "<row category header row as printed, e.g. 'Safety Assessments', 'Clinical Evaluations'>", 
      "rows": ["r1", "r2"]
    }
    // CRITICAL: Grouping rows. Category headers go here in 'grp' with their child row IDs (e.g., "r1", "r2"), NOT in 'rows'.
  ],
  
  "rows": [
    {
      "id": "r1",
      "lbl": "<assessment/activity name, verbatim. Only actual tests/procedures, NOT category section headers.>",
      "cells": [
        {"c": "c1", "v": "<EXACT verbatim cell content: 'X', 'P', '3X', 'Q2W', '(X)', a dose, etc.>", "fm": ["a","c"]}
        // ONLY include a cell here if it has a value or a marker.
        // Any column NOT listed for this row is BLANK -- do not list blank cells.
        // "fm" = footnote marker(s) attached to this cell, exactly as
        // printed (letter/number/symbol). Omit "fm" entirely if none.
      ]
    }
    // one entry per row, TOP TO BOTTOM, in printed order
  ],
  "amb": [
    "<anything you could not confidently resolve -- an illegible cell, a
       column header that seems to be missing/blank when neighbors aren't,
       overlapping text, etc. Do NOT silently guess -- describe it here.>"
  ]
}

Do NOT include footnote text anywhere -- only the marker character(s) on
the cells that carry them, in "fm". We extract footnote text separately.

CRITICAL RULES:
- Capture cell values EXACTLY as printed. Do not normalize "3X"/"Q2W"/"(X)"
  to true/false or to a plain "X". Preserve doses, dashes, dots, arrows.
- A cell may carry more than one footnote marker -- list all of them.
- Carry forward column study periods ('per') horizontally if they span multiple columns.
- If given more than one page, they are the SAME table continuing (headers
  may repeat, full or abbreviated -- that confirms column identity, it is
  not a second table).
- Distinguish row CATEGORY headers ("Safety Assessments" -- structure, not
  an assessment) from actual assessment rows. Category headers go in "grp".
- If a page you were given is NOT actually part of this table, say so in
  "amb" and do not fabricate content for it.
- If a page you are given contains ONLY footnote text, definitions, or glossary notes 
  (and no table grid columns/rows), do NOT output them as rows. Leave rows empty for 
  that page or describe it in "amb".
- Be faithful, not clever: represent genuine ambiguity in "amb" rather than
  quietly picking an interpretation.
"""


@dataclass
class SoAExtractionResult:
    protocol_file: str
    source_pages: list
    raw_model_output: str
    parsed: dict = field(default_factory=dict)
    parse_error: str = None


def render_region_to_images(pdf_path: str, start_page: int, end_page: int, trailing_pages: int = 1) -> list[dict]:
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    max_page = min(end_page + trailing_pages, total_pages)

    rendered_images = []
    scale = RENDER_DPI / 72
    matrix = fitz.Matrix(scale, scale)

    for page_no in range(start_page, max_page + 1):
        page_obj = doc[page_no - 1]
        pixmap = page_obj.get_pixmap(matrix=matrix)
        image_bytes = pixmap.tobytes("png")
        rendered_images.append(
            {
                "page": page_no,
                "b64_png": base64.b64encode(image_bytes).decode("utf-8"),
                "is_trailing_lookahead": page_no > end_page,
            }
        )
    doc.close()
    return rendered_images


def chunk_images(images: list[dict], chunk_size: int = MAX_IMAGES_PER_REQUEST) -> list[list[dict]]:
    """
    Breaks regional images into sequential, non-overlapping API chunks. 
    This is safe to do because manual review (protocol9, pp. 26-27) 
    showed that multi-page tables extend vertically (adding rows with identical columns) instead of horizontally.
    """
    
    return [images[i:i + chunk_size] for i in range(0, len(images), chunk_size)]


def expand_model_output(compact_data: dict) -> dict:
    """
    Expands the compact JSON into the full internal data structure for the pipeline. 
    It lengthens key names, pads all row dictionaries with missing columns, 
    and leaves footnotes empty to be handled later by extract_soa.
    """
    
    expanded = {
        "heading": compact_data.get("h"),
        "columns": [],
        "row_groups": [],
        "rows": [],
        "ambiguities": compact_data.get("amb", []),
        "footnotes": [],
    }
    for col in compact_data.get("cols", []):
        expanded["columns"].append(
            {
                "id": col.get("id"),
                "period": col.get("per"),
                "visit_label": col.get("vis"),
                "study_day_or_week": col.get("day"),
                "visit_window": col.get("win"),
            }
        )
    for group in compact_data.get("grp", []):
        expanded["row_groups"].append({"label": group.get("lbl"), "row_ids": group.get("rows", [])})

    column_ids = [c["id"] for c in expanded["columns"]]
    for row in compact_data.get("rows", []):
        cell_map = {cid: {"value": "", "footnote_markers": []} for cid in column_ids}
        for cell in row.get("cells", []):
            cid = cell.get("c")
            if cid not in cell_map:
                cell_map[cid] = {"value": "", "footnote_markers": []}
            cell_map[cid]["value"] = cell.get("v", "")
            cell_map[cid]["footnote_markers"] = cell.get("fm", [])
        expanded["rows"].append({"id": row.get("id"), "label": row.get("lbl"), "cells": cell_map})

    return expanded


def merge_chunk_results(parsed_chunks: list[dict]) -> dict:
    if not parsed_chunks:
        return {}
    if len(parsed_chunks) == 1:
        return parsed_chunks[0]

    merged = {
        "heading": next((c.get("heading") for c in parsed_chunks if c.get("heading")), None),
        "columns": next((c.get("columns") for c in parsed_chunks if c.get("columns")), []),
        "row_groups": [],
        "rows": [],
        "footnotes": [],
        "ambiguities": [],
    }

    for idx, chunk in enumerate(parsed_chunks):
        tag = f"chunk{idx}_"

        for row in chunk.get("rows", []):
            cloned_row = dict(row)
            cloned_row["id"] = tag + row["id"]
            merged["rows"].append(cloned_row)

        for group in chunk.get("row_groups", []):
            cloned_group = dict(group)
            cloned_group["row_ids"] = [tag + rid for rid in group.get("row_ids", [])]
            merged["row_groups"].append(cloned_group)

        merged["ambiguities"].extend(chunk.get("ambiguities", []))

        # Flag discrepancies if continuation chunks alter the primary column structure
        if idx > 0 and chunk.get("columns") and chunk["columns"] != merged["columns"]:
            merged["ambiguities"].append(
                f"Chunk {idx}'s columns differed from chunk 0's. This extractor assumes "
                f"identical columns across continuation pages and does not reconcile mismatches automatically. "
                f"Needs manual check."
            )

    return merged


MARKER_PATTERN = r"(?:\*{1,4}|[a-zA-Z]\.?|\d{1,2}\.|\(\d{1,2}\)|\u2020|\u2021|\u00a7|#|Detox|SCID)"
FOOTNOTE_START_RE = re.compile(r"^\s*(" + MARKER_PATTERN + r"|[A-Z][a-zA-Z\s]+)\s*[:=\-\u2013\u2014]\s*(.+)$")
FOOTNOTE_START_LOOSE_RE = re.compile(r"^\s*(\*{1,3}|[a-zA-Z0-9]+)[\.\)]?\s+(.+)$")
BOILERPLATE_RE = re.compile(r"copyright \u00a9|clinical study protocol|^document page \d+|^version no\.|^version \d+\s*[\u2013-]|^page \d+$", re.IGNORECASE)
HEADING_EXCLUSION_RE = re.compile(r"^\s*appendix\s+[ivxlcdm0-9]+\s*[:.]", re.IGNORECASE)
STOP_RE = re.compile(r"^\s*\d{1,2}(\.\d+)*\s+[A-Z][A-Z ]{3,}|^\s*appendix\s+[ivxlcdm0-9]+", re.IGNORECASE)


def _clean_superscripts(lines: list[str]) -> list[str]:
    """
    Fixes a PyMuPDF parsing bug where superscript footnote letters get isolated on the line before their "X" marker, which breaks footnote-to-cell linkage. 
    This function uses a lookahead scan to detect an isolated single letter followed by an "X" line 
    and merges them back together into a correctly formatted string (e.g., "Xb - ...").
    """
    
    cleaned = []
    idx = 0
    single_letter = re.compile(r"^[a-z]$")
    x_match = re.compile(r"^X\s*[-\u2013\u2014:=]\s*(.+)$")
    while idx < len(lines):
        line = lines[idx].strip()
        if idx + 1 < len(lines) and single_letter.match(line):
            next_line = lines[idx + 1].strip()
            match_res = x_match.match(next_line)
            if match_res:
                cleaned.append(f"X{line} - {match_res.group(1)}")
                idx += 2
                continue
        cleaned.append(lines[idx])
        idx += 1
    return cleaned


def extract_footnote_text_from_pdf(pdf_path: str, start_page: int, end_page: int, trailing_pages: int = 1) -> dict:
    """
    Retrieves footnote text straight from the PDF's text layer, avoiding the vision model for simple prose. 
    By scanning sequentially and appending normal lines to the most recently matched footnote, it captures long footnotes that break across pages. 
    Finally, it returns a dictionary of markers to text, stripping out the leading "X" value so the markers match exactly what the vision model reports.
    """
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    final_page = min(end_page + trailing_pages, total_pages)

    lines = []
    for p in range(start_page - 1, final_page):
        lines.extend(doc[p].get_text().splitlines())
    doc.close()

    lines = _clean_superscripts(lines)
    footnotes = {}
    active_marker = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or BOILERPLATE_RE.search(line):
            continue
        if HEADING_EXCLUSION_RE.match(line):
            active_marker = None
            continue
        
        match = FOOTNOTE_START_RE.match(line) or FOOTNOTE_START_LOOSE_RE.match(line)
        if match:
            raw_marker, text = match.group(1), match.group(2)
            marker = raw_marker[1:] if (raw_marker.startswith("X") and len(raw_marker) == 2) else raw_marker
            marker = marker.rstrip(".")
            active_marker = marker
            footnotes[marker] = text.strip()
        elif STOP_RE.match(line):
            active_marker = None
        elif active_marker is not None:
            if len(footnotes[active_marker]) < 500:
                footnotes[active_marker] += " " + line

    return footnotes


def _attach_footnotes(merged_data: dict, footnote_text: dict) -> None:
    """
    Scans the merged grid for every marker actually used on a cell, looks
    each one up in footnote_text, and populates merged["footnotes"]. A
    marker seen in the grid but NOT found in the text layer is recorded
    with text=None and flagged in "ambiguities" -- faithful-not-clever.
    """
    used_markers = {}
    for row in merged_data.get("rows", []):
        for col_id, cell in row.get("cells", {}).items():
            for m in cell.get("footnote_markers", []):
                used_markers.setdefault(m, []).append(f"{row['id']}:{col_id}")

    for marker, refs in sorted(used_markers.items()):
        txt = footnote_text.get(marker)
        merged_data["footnotes"].append({"marker": marker, "text": txt, "attached_to": refs})
        if txt is None:
            merged_data["ambiguities"].append(f"Footnote marker '{marker}' appears on cell(s) {refs} but its definition text could not be located via text-layer extraction. Needs manual check against the source.")


def extract_soa(pdf_path: str, start_page: int, end_page: int, api_key: str = None) -> SoAExtractionResult:
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not resolved_key:
        raise RuntimeError("Gemini API key not found in environment variables.")

    images = render_region_to_images(pdf_path, start_page, end_page, trailing_pages=0)
    image_chunks = chunk_images(images, chunk_size=3)

    client = genai.Client(api_key=resolved_key)
    expanded_chunks = []
    raw_outputs = []
    parse_errors = []
    max_retries = 3

    for chunk in image_chunks:
        contents = []
        for img in chunk:
            label = f"Page {img['page']}"
            contents.append(label)
            contents.append(
                types.Part.from_bytes(
                    data=base64.b64decode(img["b64_png"]),
                    mime_type="image/png",
                )
            )
        contents.append(SCHEMA_DESCRIPTION)

        delay = 10
        success = False
        chunk_err = None

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"),
                )
                raw_text = response.text
                raw_outputs.append(raw_text)

                cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
                compact_json = json.loads(cleaned)
                expanded_chunks.append(expand_model_output(compact_json))
                success = True
                break
            except Exception as err:
                chunk_err = err
                err_message = str(err)
                if any(code in err_message for code in ["429", "503", "RESOURCE_EXHAUSTED"]) and attempt < max_retries - 1:
                    print(f"Rate limited or server busy (attempt {attempt + 1}/{max_retries}), waiting {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                break

        if not success:
            parse_errors.append(f"Chunk starting page {chunk[0]['page']} failed: {chunk_err}")

    result = SoAExtractionResult(
        protocol_file=os.path.basename(pdf_path),
        source_pages=[img["page"] for img in images],
        raw_model_output="\n---BOUNDARY---\n".join(raw_outputs),
    )

    if parse_errors:
        result.parse_error = "; ".join(parse_errors)
        return result

    merged_result = merge_chunk_results(expanded_chunks)
    footnotes = extract_footnote_text_from_pdf(pdf_path, start_page, end_page)
    _attach_footnotes(merged_result, footnotes)
    result.parsed = merged_result

    return result


if __name__ == "__main__":
    import sys
    from locator import find_soa_regions

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <path_to_pdf>")
        sys.exit(1)

    target_pdf = sys.argv[1]
    located_regions = find_soa_regions(target_pdf)
    
    if not located_regions:
        print("No SoA region found.")
        sys.exit(1)

    extracted_results = []
    for reg in located_regions:
        print(f"Extracting region: pages {reg['start_page']}-{reg['end_page']} ({reg['heading_text']})")
        extraction_data = extract_soa(target_pdf, reg["start_page"], reg["end_page"])
        
        if extraction_data.parse_error:
            print(f"  WARNING: model output was not valid JSON: {extraction_data.parse_error}")
            
        extracted_results.append(
            {
                "source_pages": extraction_data.source_pages,
                "parsed": extraction_data.parsed,
                "raw_model_output": extraction_data.raw_model_output if extraction_data.parse_error else None,
            }
        )

    output_filename = f"outputs/{os.path.splitext(os.path.basename(target_pdf))[0]}_soa.json"
    os.makedirs("outputs", exist_ok=True)
    
    with open(output_filename, "w") as out_file:
        json.dump(extracted_results, out_file, indent=2)
        
    print(f"Wrote {output_filename}")