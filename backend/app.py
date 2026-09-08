"""
SoA Extraction Tool - web server.

Endpoints:
    POST /api/extract   - upload a protocol PDF, get back located + extracted
                           SoA table(s) as JSON.
    GET  /               - serves the frontend.

Run:
    export GEMINI_API_KEY='your_key'
    uvicorn app:app --reload --port 8000
Then open http://localhost:8000
"""

import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from locator import find_soa_regions
from extractor import extract_soa

app = FastAPI(title="SoA Extraction Tool")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")

    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(
            500,
            "Server has no GEMINI_API_KEY configured. Set it in your environment "
            "before starting the server: export GEMINI_API_KEY='your_key'",
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        regions = find_soa_regions(tmp_path)
        if not regions:
            return JSONResponse(
                {
                    "filename": file.filename,
                    "regions": [],
                    "message": "No Schedule of Activities-style table was found in this document. "
                    "This can happen if the document uses heading phrasing outside what the "
                    "locator recognizes, or if the table is scanned/image-only with no text layer "
                    "at all near its heading.",
                }
            )

        results = []
        for region in regions:
            try:
                extraction = extract_soa(
                    tmp_path, region["start_page"], region["end_page"],
                    shares_page_with=region.get("shares_page_with"),
                )
                results.append(
                    {
                        "heading": region["heading_text"],
                        "located_pages": f"{region['start_page']}-{region['end_page']}",
                        "source": region["source"],
                        "shares_page_with": region.get("shares_page_with"),
                        "source_pages_sent_to_model": extraction.source_pages,
                        "parsed": extraction.parsed,
                        "parse_error": extraction.parse_error,
                        "raw_model_output": extraction.raw_model_output if extraction.parse_error else None,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "heading": region["heading_text"],
                        "located_pages": f"{region['start_page']}-{region['end_page']}",
                        "source": region["source"],
                        "error": f"Extraction failed for this region: {e}",
                    }
                )

        return JSONResponse({"filename": file.filename, "regions": results})

    finally:
        os.unlink(tmp_path)