# Clinical Schedule of Assessments (SoA) Extraction Pipeline

A robust, hybrid-architecture extraction pipeline designed to parse messy, multi-page clinical trial protocol PDFs into structured, machine-readable Schedule of Assessments (SoA) data matrices.

## 1. Architectural Design & Implementation

Clinical protocol tables routinely break standard coordinate-based PDF parsers (like `pdfplumber` or `camelot`). They feature shattered gridlines, landscape rotations, fused superscript footnote markers, and tables that span multiple pages. 

To solve this, the pipeline abandons strict coordinate math in favor of a decoupled, three-phase architecture:

### Phase 1: Heuristic Location (`locator.py`)
The system finds SoA tables dynamically without hardcoded page numbers.
*   **Two-Pass Scanning:** It first scans the PDF's Table of Contents (bookmarks) using regex patterns (e.g., "Schedule of Activities", "Time and Events"). If the TOC is missing or incomplete, it falls back to scanning the document's body text for heading patterns, filtering out prose references (e.g., "Table 1 provides an overview...") to avoid false positives.
*   **Lookahead Pagination (`_walk_forward`):** Once a heading is found, the locator scans subsequent pages. By calculating a "row hint" density (ratio of short text lines and specific keywords like "ECG" or "vital signs") and looking for continuation keywords, it successfully captures tables that span multiple pages.

### Phase 2: Vision-Based Grid Extraction (`extractor.py`)
Instead of scraping the text layer for the grid, the pipeline renders the located pages as high-resolution PNGs via PyMuPDF and passes them to a Vision Language Model (Google Gemini 3.6 Flash).
*   **Chunking Strategy:** To prevent token-limit exhaustion and context-window degradation on large tables, the system batches images into chunks of 3 pages per API request. 
*   **Sparse Matrix Prompting:** The vision model is strictly prompted to return a lightweight, abbreviated JSON schema. It explicitly drops blank cells (saving massive amounts of output tokens) and preserves exact cell values verbatim (e.g., "3X", "Q2W", "(X)") rather than coercing them into booleans.
*   **Resiliency:** The extraction runs inside an automated exponential backoff loop to elegantly handle free-tier API rate limits (`RESOURCE_EXHAUSTED` or `429` errors).

### Phase 3: Text-Layer Footnote Resolution
Asking a vision model to transcribe dense paragraphs of prose wastes tokens and frequently results in truncated text at page boundaries. 
*   **Decoupled Extraction:** Footnote prose is extracted directly from the PDF's plaintext layer using PyMuPDF. It scans for standard markers (asterisks, letters, daggers) and appends subsequent lines to handle footnotes that spill across page breaks.
*   **Superscript Bug-Fixing:** PyMuPDF often detaches superscript footnote letters from their "X" markers (e.g., parsing "Xb" as two separate lines). A custom lookahead regex function (`_clean_superscripts`) detects and merges these back together so they can be accurately mapped.
*   **Faithful Linkage:** The pipeline iterates through the markers identified by the vision model inside the grid and maps them to the text-layer definitions.

## 2. Output Schema & Rationale

The pipeline outputs a strict JSON schema designed for structural fidelity and programmatic consumption:

*   `h`: The table's title/heading.
*   `cols`: An array defining the column hierarchy (`per`: Study Period, `vis`: Visit Label, `day`: Study Day/Week, `win`: Visit Window). Column periods carry forward horizontally if implied.
*   `grp`: Separates structural category headers (e.g., "Safety Assessments") from actual assessable rows, preserving the visual hierarchy of the table.
*   `rows`: The assessment rows. Cells are represented as a dictionary mapped to column IDs. Each cell contains an exact `value` string and an array of `footnote_markers`.
*   `amb`: An array logging any ambiguities (e.g., unresolvable cells, missing definitions) for human review.

## 3. Tool, API, and Model Selection

*   **PyMuPDF (fitz):** Selected over `PyPDF2` for its superior speed, exact text-layer extraction, and built-in ability to render PDF pages directly to image matrices for the vision pipeline.
*   **Google Gemini 3.6 Flash:** Chosen for its exceptional visual reasoning, strict JSON-mode adherence, and generous free-tier context window.
*   **FastAPI & Vanilla HTML/JS:** Chosen for the user interface. By avoiding heavy frontend frameworks like React or Node.js, the reviewer can run the entire stack locally with a single Python command.

## 4. Manual Verification Results

*Note: The pipeline defaults to a "faithful, not clever" design philosophy. If an ambiguity is found, it logs it in the UI rather than guessing silently.*

*   **Protocol 1:** *TO BE ADDED: [Detail what was right, what was wrong, and how it was wrong]*
*   **Protocol 5:** *TO BE ADDED: [Detail what was right, what was wrong, and how it was wrong]*
*   **Protocol 9:** 
    *   **What was right:** Fully captured the multi-page vertical flow across pages 26–29 without fragmentation. Identified all 11 chronological study days mapped across 4 study phases. Successfully categorized 33 assessment rows under 4 primary clinical groupings. Correctly parsed multi-frequency text cells ("6X", "5X", "8X") without boolean flattening. Successfully identified that page 29 was an auxiliary footnote page, suppressing ghost rows and logging it to ambiguities.
    *   **What was wrong:** Dropped row-level footnote linkage (`footnotes` returned empty) because reference markers were placed on row labels rather than inside individual grid cells. Visual horizontal spanning arrows across treatment phases were read as blank cells rather than spanning ranges.
*   **Protocol 12:** *TO BE ADDED: [Detail what was right, what was wrong, and how it was wrong]*
*   **Protocol 15:** *TO BE ADDED: [Detail what was right, what was wrong, and how it was wrong]*

## 5. Where the Tool Breaks & How It Reacts

*   **Mismatched Columns on Continuation Pages:** If the structure of the table drastically changes across a page break (e.g., a sub-study table uses entirely different columns), the pipeline merges the chunks but flags a warning in the `ambiguities` array indicating a structural mismatch. It does not attempt to automatically mutate the schema.
*   **Complex Glossary-Style Footnotes:** If inline definitions lack explicit punctuation splitters, the regex may fail to capture the definition from the text layer. The system handles this gracefully: it outputs the marker found in the grid with a `null` text value and appends a warning to the `ambiguities` array prompting a manual check.

## 6. What I Would Build Next (Given Two More Weeks)

1.  **Bounding Box UI Highlighting:** Map the extracted JSON elements back to their original PDF coordinate boxes, allowing users to click a cell in the web UI and see the exact region highlighted on the source document.
2.  **Local Vision Model Alternative:** Integrate a smaller, open-weights vision model (like Qwen-VL or Pixtral) to remove the dependency on commercial APIs, completely eliminating rate-limiting bottlenecks and data privacy concerns.
3.  **Enhanced NLP Footnote Engine:** Upgrade the regex-based footnote parser to use an NLP-based named entity recognition approach. This would better untangle dense, non-standard glossary blocks that evade simple regex matching.

## 7. AI Tool Usage

*   *TO BE ADDED: [List the AI tools you used (e.g., Claude Code, Cursor, ChatGPT)]*
*   *TO BE ADDED: [Detail exactly where they helped (e.g., scaffolding the FastAPI backend, writing regex)]*
*   *TO BE ADDED: [Detail where they got in the way (e.g., wasted time on pdfplumber dead-ends)]*

---

## Quick Start & Installation

1. **Environment Setup** (Requires Python 3.10+)
   ```bash
   # Navigate to the project root
   cd intakeAI-takehome-1b-soa-extractor
   
   # Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r backend/requirements.txt