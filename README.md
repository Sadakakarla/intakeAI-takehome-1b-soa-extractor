# Intake AI Take-Home Assignment 1b - SoA Extraction Tool

This is a tool that takes a clinical trial protocol PDF, finds the Schedule of Activities inside it, and pulls it out into structured JSON that a person can actually check against the source document. It comes with a small web UI so you can drop in a PDF and see what the tool found, rather than just trusting a JSON blob.

How I approached this: I spent most of my time on the two hardest parts the assignment calls out directly — not dropping rows or columns, and getting footnotes linked to the right cells — because those are the parts that actually matter if this were feeding into a real study build. I'd rather hand in something that's honest about where it's shaky than something that looks polished but hides problems.

## Setting it up

```
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY='your_key_here'
uvicorn app:app --reload --port 8000
```

Then open `http://localhost:8000`, upload any protocol PDF, and it'll locate and extract the table(s) right there in the browser.

If you just want to run it from the command line against one file (this also writes the JSON to `outputs/`):
```
python3 extractor.py /path/to/protocol.pdf
```

I built this with FastAPI on the backend, PyMuPDF for rendering pages and doing the text-based locating work, Google's Gemini (vision model) for the actual grid reading, and `pdftotext` from poppler as a backup text source for one specific check I'll explain below. The frontend is just plain HTML and JavaScript — no framework, nothing fancy, since the assignment says visual design isn't being graded.

---

## How it actually works

### Finding the table (`locator.py`)

The locator doesn't get told where the table is — it has to go find it, and the assignment is explicit that the heading won't always say "Schedule of Activities" and the table won't always live in the same spot. So it looks in two places:

First, it checks the PDF's own table of contents / bookmarks for anything matching a handful of known heading phrasings (Schedule of Activities, Schedule of Assessments, Study Flow Chart, Table of Events, Time and Events Schedule, and a few others). Some of the protocols I was given have a proper bookmark for this and some don't.

For the ones that don't, it falls back to scanning the actual page text looking for the same kind of heading — but this needs to be careful, because a lot of protocols have sentences like "Table 1 provides an overview of the schedule of assessments" scattered through the prose, and that's not the heading, that's just a sentence mentioning the table. I added a check that looks at whether the matched phrase sits inside a full sentence with a verb around it (which means it's prose talking about the table) versus sitting on its own short line (which means it's an actual heading).

Once it finds a starting point, it walks forward page by page to figure out where the table actually ends. This part looks at a mix of things: how many rows look like clinical assessment terms, how many lines on the page are short (a real grid has tons of short lines — that's just what cell values look like), whether the page says "continued" or "concluded," and whether it hits something that looks like the start of a new section.

Two specific problems came up while I was testing this against the five given protocols, and both took real debugging to track down:

**It was grabbing prose by mistake.** On one protocol, the actual table lived entirely on a single page, but my first version of the locator kept walking three pages further because the following pages had numbered lists like "1. Informed consent / 2. Complete medical history..." — and a numbered list of short items looks statistically a lot like a grid if you're just counting short lines. I fixed this by teaching the heuristic to recognize the specific pattern of an enumerated list (a number or letter followed immediately by real text) and treat a page dense with that pattern as prose, not a grid, even if the short-line count alone would have said otherwise.

**Two separate tables were sharing one page.** This is the exact case the assignment specifically warns about — a protocol can have a main schedule plus a smaller sub-schedule somewhere. On one of the five protocols, the main schedule's footnotes finish partway down a page, and immediately below them, on that same physical page, a completely different table starts (a blood-sample collection schedule with its own columns and its own logic). My first pass merged these into one table, which is wrong — the assignment is pretty direct that this shouldn't happen. I added a check that looks for a second table-style heading appearing partway through a page the locator already thinks it's inside, and when it finds one, it splits into two separate regions instead of merging them, and tags both with a note about the fact that they share a page. One annoying detail here: I originally tried to find the exact pixel position where one table ends and the other begins so I could crop the image cleanly, but on this particular document the underlying PDF's text positioning data for that heading was genuinely broken (the heading was scattered across mismatched coordinates, probably some quirk from how the PDF was originally produced), so pixel cropping wasn't reliable. Instead I just tell the model in plain language which page is shared and what the other table's name is, and ask it to only extract its own portion — which turned out to work well when I checked the actual output.

### Reading the table (`extractor.py`)

Once a region is located, the extractor renders every page in that region to an image and sends all of them to Gemini in a single call, along with a fairly detailed prompt describing exactly the shape of JSON I want back — columns (with period, visit label, day/week, and window), row groups for category headers, rows with per-cell values, footnote markers on both cells and rows, and a spot to note anything genuinely ambiguous instead of guessing.

I made a deliberate choice to send a whole table in one call rather than splitting it into chunks of a few pages at a time, and this actually came from a real bug. My first version chunked pages into small groups and sent each group as its own separate call. The problem is that each call has no idea what the other calls decided — so if page one calls a column "c1" and page two (in a different call) also calls something "c1", there's no guarantee they're actually the same visit. The assignment specifically warns that headers on continuation pages can repeat in a shortened form or not repeat at all, and that's exactly the situation where this would go wrong. Sending the whole table in one call sidesteps the problem entirely, since the model sees every page together and only needs to invent one set of column ids. It also happens to cut down how many API calls I burn through, which mattered a lot given I was on the free tier the whole time (more on that below).

Footnotes get handled completely separately from the vision model, and I think this was the right call. There's a second function that reads the raw text layer of the PDF directly — no AI involved — looking for lines that start with something that looks like a footnote marker (a letter, a number, an asterisk, a dagger, or a couple of named markers like "SCID" that show up as literal footnote references in some of these documents) followed by real explanatory text. It stitches together footnotes that spill across a page break, since I noticed those get cut off if you don't handle multi-page scanning here. Then a separate function cross-checks every marker the vision model actually reported seeing in the grid against this text dictionary — if a marker doesn't have a matching definition, it shows up honestly as `null` with an explanation in the ambiguities list, rather than the tool inventing something plausible-sounding.

This text-layer footnote parser is genuinely the part of the pipeline I had to fix the most, and I want to be honest about that rather than pretend it was smooth. The first version had a bug where the "strip the leading X off the marker" logic silently failed because of a whitespace quirk in how the regex captured things, which meant a real footnote's marker never got cleaned up properly. Once I fixed that, I found something worse: two different pieces of text in the same document could coincidentally both look like a definition for the same marker — for instance, the real footnote "a" on one page, and then an unrelated numbered list further down the document that also happens to use "a." as a bullet — and whichever one the code scanned second would silently overwrite the first, with no error or warning. That's a genuinely dangerous failure mode, because it produces something that looks complete and correct but is actually wrong, which is worse than the tool just admitting it couldn't find the answer. I ended up fixing this by anchoring the scan to start only after a recognizable "here's where the footnotes begin" heading, when the document has one, so it never sees the unrelated content in the first place. I checked this fix against three different protocols with different footnote styles before I trusted it, because my first attempted fix (just "keep whichever definition you saw first") turned out to be wrong in the opposite direction on a different document, where the real definition actually came later in the page order.

Lastly, there's a small cleanup step that drops a column if it has absolutely no header information and absolutely no cell values anywhere — I ran into this exact case once, where the model produced a completely empty extra column, likely from briefly treating the row-label header as if it were its own data column. Rather than just deleting it silently, the code leaves a note behind explaining what got removed and why.

### The UI (`app.py` / `index.html`)

Upload a PDF, it runs through locate-then-extract, and the result renders as an actual HTML table per region, with footnote markers shown as small badges you can trace back to a footnote list underneath. Anything flagged as ambiguous, or any page that's shared between two tables, shows up as a visible warning banner rather than being buried in the JSON. I kept this intentionally simple since the assignment says visual polish isn't what's being graded — the point was making the output something a person can actually check line by line against the PDF, and I believe it does that.

---

## The output schema, and why I designed it this way

```json
{
  "heading": "...",
  "columns": [{ "id": "c1", "period": "...", "visit_label": "...", "study_day_or_week": "...", "visit_window": "..." }],
  "row_groups": [{ "label": "Safety", "row_ids": ["r1", "r2"] }],
  "rows": [{ "id": "r1", "label": "...", "row_footnote_markers": [], "cells": { "c1": { "value": "3X", "footnote_markers": ["a"] } } }],
  "footnotes": [{ "marker": "a", "text": "...", "attached_to": ["r1:c1"] }],
  "ambiguities": ["..."]
}
```

A few of these choices are worth explaining rather than just listing:

**Cell values are plain strings, kept exactly as written.** I never convert anything to true/false. If the source says "3X" or "Q2W" or leaves a cell blank, that's exactly what shows up in the output. This was the single most repeated point in the assignment brief, so it felt important to get right, and I checked this by hand across all five protocols — every non-standard value I found in the source PDFs (things like "6X", "5X", "P" for a practice-only visit) came through unchanged.

**Period, visit, day/week, and window are four separate fields on a column, instead of one flattened header string.** Real protocol tables usually stack these on top of each other in the header — a "period" row grouping several visits together, then a visit number row, then a study day row underneath that. Squashing that into one string would lose the grouping, so I kept them as separate fields. Same logic applies to `row_groups` on the row side — a category header like "Safety Assessments" is structure, not an actual assessment, and needs to be represented as something that groups rows together rather than being mistaken for a row itself.

**Footnotes live in their own list, referenced by marker, instead of being pasted into cell text.** This means the same footnote can be linked to five different cells without repeating its text five times, and — more importantly for grading purposes — it makes a missing or unresolved footnote visible as a specific gap (`text: null`) rather than something that just silently isn't there.

**`ambiguities` is a plain list of sentences, not a structured object.** I thought about giving this more structure, but the things that actually ended up in there — an unresolved footnote marker, a page shared between two tables, one case where the model's own explanation of what it saw turned out to be wrong when I checked it — don't share a common shape. Forcing them into rigid fields would have meant either dropping detail or bending the schema to fit, and I'd rather keep this part loose and readable.

One honest note on `visit_window`: it comes back empty (`null`) on every single column across all five protocols. I didn't want to just leave that unexplained, so I went back and searched the raw text of all five source PDFs for anything resembling an allowable window (the "Day 15 ± 3 days" style format the assignment mentions as typical). None of the five actually contain that — the only "±" symbols anywhere in these documents are things like lab value statistics and an HIV test explainer, completely unrelated to visit scheduling. These are all older protocols (mostly early-2000s NIDA studies plus one from 2006), and it looks like that particular convention just wasn't in use yet, or these particular sponsors didn't adopt it. So the field being empty here is accurate, not a miss — but I obviously can't promise it'll stay empty on a newer protocol that does use windows, since I've never actually seen my extractor populate that field with a real value.

---

## Tools I looked at, and what I ended up using

Instead of a generic disclaimer about its limitations, here is a specific breakdown of known issues and how the system gracefully handles them rather than failing silently:

**PyMuPDF** — this does almost all of the heavy lifting on the locating side: reading text for the heading search, and rendering pages to images for the vision model. It's fast and doesn't need anything installed outside of Python. Its one real weakness that I ran into: on one of the five protocols, its text extraction badly scrambled a heading — splitting the word "APPENDIX" across separate reported lines with wildly different vertical positions, which looked like it might be some kind of glyph or font-encoding quirk in that specific PDF. That one broke my first attempt at detecting the second, hidden table on a shared page.

**`pdftotext` (poppler), run in "layout" mode** — I brought this in specifically to work around the PyMuPDF issue above. When I ran the exact same page through `pdftotext -layout` instead, it read the heading correctly. I only use it for that one narrow check (finding a second table heading mid-page) rather than switching everything over to it, since calling out to an external program for every single page would slow things down for no benefit on the documents where PyMuPDF already works fine.

**Gemini's vision model, for actually reading the grid** — I chose to feed it images of the pages rather than extracted text, because a lot of what makes these tables hard is fundamentally visual: a footnote letter sitting as a tiny superscript right next to an "X", or a header that's visually merged across several columns. Any plain text extraction throws that positioning information away, which is exactly what the assignment warns will happen ("generic PDF-to-text extraction will get you perhaps halfway"). I actually tried `pdfplumber`'s built-in table extraction early on as a text-based alternative, and it confirmed the warning — it pulled out the grid's text fine, but a superscript marker and the cell value next to it just got mashed together into one text run with no way to tell where the marker started, which is precisely the linkage the assignment cares most about. That's why I moved to the vision approach instead.

**OCR — not built.** All five given protocols have real, extractable text layers (I checked this with `pdffonts` before writing a single line of locating code), so I never needed an OCR fallback and didn't build one. I'm flagging this honestly as a real gap rather than pretending I tested it, since the assignment explicitly says some real-world protocols are scanned.

**On the budget side** — I stayed on Gemini's free tier the entire time, partly by choice and partly because I genuinely didn't want to spend money I wasn't comfortable spending, per the ground rules given in the document. This came with a real, practical constraint: the free tier caps out at 20 requests a day per project, and I hit that limit more than once while iterating. I dealt with it by making sure each protocol only costs one API call instead of several (see the single-call design above), by adding code that fails immediately on a daily-quota error instead of uselessly retrying, and, when I did hit the daily cap mid-testing, by starting a fresh Google Cloud project to get a new quota bucket rather than paying. I mention this because it genuinely shaped some of the engineering decisions above — the "send the whole table in one call" choice exists partly because it's simply the right way to avoid the column-drift bug.

---

## What I found when I checked each protocol by hand

I want to be clear about how I actually did this part: for every protocol, I opened the source PDF side by side with the JSON output and went through it row by row, and for the trickiest rows (the ones with footnote markers that only apply to some cells, or unusual cell values) I checked those specifically rather than just skimming.

**Protocol 1 (Xanomeline, an Alzheimer's study)** — All 14 columns and all 30 rows are present and match the source. I paid special attention to a row called "NPI-X," because it spans two pages and has a footnote marker that only applies to some of its cells and not others — the marker correctly turns on and off at exactly the right columns on both sides of the page break, which told me the footnote linkage is actually working properly on the hardest case I could find, not just the easy ones. Both footnote definitions match the source word for word. There's also a genuinely odd quirk in this specific PDF — the source document itself has one page literally duplicated (the same content appears twice, on two different physical pages) — and the tool handled that correctly by not producing duplicate rows from it. The one thing I did catch: the tool's own notes claimed one page had "no table data," and that's simply not true — that page's data is right there in the output, correct and complete. I decided to leave that note in the JSON rather than delete it, since it's an honest record of what the model said, but I want to flag clearly here that I checked it and it's wrong. The lesson I took from this is that I can't just trust the tool's self-reported notes about its own confidence — I have to actually check the output myself, which is exactly what this whole verification section is meant to demonstrate.

**Protocol 5 (Atomoxetine and cocaine)** — This is the one with two tables sharing a page, which I already described above. After the fix, the output correctly produces two separate tables — one with 11 columns and 31 rows, one with 16 columns and 8 rows — and I checked every row label in both to make sure nothing from one table leaked into the other. It didn't. Both row and column counts match the source exactly. Two small things worth mentioning: one footnote (written in the source as four asterisks jammed directly against the following word with no space or punctuation between them) didn't get picked up by my parser and shows up honestly as unresolved rather than guessed at. And the second table has one column that's completely empty — it looks like the model copied over what's really the row-label header text ("Analysis") as if it were its own data column. No real data is lost here, it's just a redundant, empty column sitting next to the real ones.

**Protocol 9 (Lofexidine)** — This one has the most complicated header structure of the five: four different phases (Opiate Agonist, Detoxification, Post Med/Detox, Medical Discharge) stacked over 11 study days in an irregular pattern, and it came through matching the source exactly, phase boundaries and all. The row categories (Primary Outcome, Secondary Outcomes, Abuse Potential, Safety) also match the source's own section headers precisely, accounting for all 33 rows. I specifically checked a couple of vital-signs rows that use repeat-count values like "6X" and "5X" instead of a plain "X," and those came through exactly as written, dropping down to "1X" on the last day exactly where the source does. The named and symbol-based footnotes all resolved correctly too. One small quirk: seven rows in the "Secondary Outcome Measures" section picked up a spurious footnote marker that turned out to just be a bullet-point character from the source's own list formatting, not an actual footnote reference. Since there's no real footnote definition for a bullet point, it correctly shows up as unresolved rather than being made up — but it's worth knowing that the association itself doesn't mean anything.

**Protocol 12 (Modafinil)** — 8 columns, 37 rows, 3 category groups, all matching the source. This protocol has ten different lettered footnotes (a through j), and getting all ten to resolve correctly is actually what led me to the footnote-corruption bug I described earlier — my first attempt got four of them silently wrong before I caught it by comparing text side by side with the PDF. After the fix, all ten check out against the source. Same four-asterisk footnote gap as protocol 5 shows up here too. This was also the protocol where I found the empty phantom column issue, which is now cleaned up automatically.

**Protocol 15 (Cabergoline)** — 31 rows, 9 columns, 3 groups, all matching. This is also the protocol where my locator originally over-extended into prose pages by mistake — the real table lives entirely on one page, and my first version pulled in three extra pages of unrelated numbered lists before I fixed the underlying heuristic. After the fix, it correctly stops at exactly the right page. I checked the "Vital signs" row closely here too, since its footnote marker switches from one letter to another right at the final treatment visit, and that switch happens at exactly the right column. One footnote didn't resolve — the source PDF line-wraps right after the marker in a way that splits the definition across two lines awkwardly, and my text parser doesn't reconnect it. Flagged honestly rather than guessed.

---

## Questions I'd want to ask someone who actually runs these trials

The assignment specifically asked me to write down anything I'd want a clinical SME to weigh in on rather than quietly guess my way past it, so here's what came up while I was working through these five documents:

- On protocol 9, several rows in the "Secondary Outcome Measures" section are prefixed with a bullet point in the source formatting, and my tool ends up treating that bullet as if it might be a footnote reference. Is that bullet meaningful in any clinical sense — like, does it mean something different from the assessments listed without a bullet — or is it purely a formatting choice with no clinical weight? I've treated it as pure formatting, but I'd rather confirm that than assume it.
- A couple of footnotes across these documents are written with no space between the marker and the following word (four asterisks glued directly to the next word, or a marker split awkwardly across a line wrap). Is this just an artifact of how these particular PDFs were typeset, or is there a chance the sponsor intended something slightly different by omitting the usual spacing — like a different kind of annotation? I've assumed it's a typesetting accident and treated the footnote as a normal one, just harder to extract.
- Protocol 1's source PDF has one page that's an exact duplicate of the page before it. I don't have any way to know from the PDF alone whether that's a printing/production mistake on the sponsor's side, or whether something was actually supposed to be different on that second page and got lost. I've assumed it's a production error and treated the duplicate as redundant, but I'd want to confirm that with whoever produced the original document rather than assume.
- None of my five documents use an explicit visit window (the "Day 15 ± 3 days" style range the assignment mentions). I've assumed that just means these particular protocols don't use that convention, but I don't have a way to verify from the documents alone whether that's a deliberate design choice for these particular studies or simply an older convention these sponsors weren't using yet.

---

## Where this breaks, and what happens when it does

I think it's more useful to be specific here than to just say "it has some limitations," so here's exactly what I know doesn't work well, and what the tool does about it instead of silently failing:

- **Scanned PDFs with no real text layer.** All five protocols I was given have proper embedded text, so I never had to build anything for a scanned page, and I want to be upfront that this path is genuinely untested. My best guess is that the locator would simply find nothing on a scanned page, since almost everything it does depends on reading text first. I did not build an OCR fallback given the time I had.
- **Footnotes with no separator between the marker and the text.** A couple of real examples from these five documents: four asterisks jammed straight into the following word, or a marker that gets split from its own definition by an awkward line wrap in the PDF. My text-based footnote parser doesn't catch these. When it can't find a match, it doesn't guess — the footnote shows up with `text: null` and an explanation in the ambiguities list, so at least it's visible as a gap rather than quietly vanishing.
- **A bullet point or other list-style character occasionally gets mistaken for a footnote marker.** Since there's never a real definition for a bullet point, this fails safely into the ambiguities list rather than making something up, but it's worth knowing the association itself is spurious.
- **The model's own explanation of what it found isn't always trustworthy.** I caught this directly on protocol 1, where it claimed a page had no table data on it when that page's data was, in fact, extracted correctly and sitting right there in its own output. I don't have anything built to automatically catch this kind of contradiction — right now it only gets caught because I manually checked, which is exactly why I did the manual checking in the first place.
- **A table that spans more than 8 pages would get split across multiple separate API calls,** and I don't currently have a way to make sure the column ids stay consistent across that split. None of the five given protocols are anywhere close to this long, so I haven't actually hit this in practice, but I want to be honest that it's a real gap rather than something I've fully solved.
- **Two tables sharing one page rely on the model reading the visual boundary correctly,** rather than a precise crop, because the positional data I'd need to crop precisely turned out to be unreliable on at least one real document. It worked correctly on the one case I tested it against, but I haven't tested it against a second example of this same situation.

---

## What I'd actually build next if I had two more weeks

If I had more time, this is roughly the priority order I'd tackle things in, starting with what feels most urgent:

First, I'd build a real OCR path for scanned protocols, since that's a complete blind spot right now and the assignment specifically says some real-world protocols show up scanned. Right now my tool wouldn't work on one of those, and that's a gap I'd like to cover.

Second, I'd go back and rebuild the footnote extraction to work off the actual positions of words on the page instead of just reading lines of text — that would fix the line-wrap issue I found on protocol 15, and it would probably let me stop relying on the small list of heading phrases I currently use to figure out where the footnote section starts, since a document that phrases its footnote heading differently than the ones I've seen would currently just fall back to a less safe scanning mode.

Third, I noticed the tool's own confidence notes aren't always accurate — the false "no table data" claim on protocol 1 is a good example — so I'd want to build something that automatically double-checks what the model says against what it actually produced, instead of relying on me noticing it by hand like I did here.

Fourth, for the case where a table is genuinely long enough to need more than one API call, I'd want to actually solve the column-consistency problem properly — something like showing a later call what columns the first call already decided on, and asking it to match those instead of inventing new ones from scratch.

Fifth, I'd want to try this against protocols beyond the five I was given, especially some of the newer ones that follow the more standardized ICH M11 template the assignment mentions, just to get an honest sense of how much of what I built is genuinely general-purpose versus how much I ended up quietly tuning to fit these specific five documents. Five documents is a small sample, and I'd want to know that before trusting this on something I haven't seen.

And last, something I think would genuinely help whoever uses this tool downstream: right now every cell and every footnote is treated with the same level of confidence, but that's not really true — some of them I verified myself and I'm confident about, and some are exactly the kind of edge case described above. A simple flag on the parts that are more likely to need a second pair of eyes would make the manual-review step this tool is meant to support a lot more efficient.

---

## The AI tools I used

I used Claude (Anthropic) for this project, mainly for two things: talking through the architecture as I was building it, and — more importantly — debugging real problems that only showed up once I started comparing the tool's output against the actual source PDFs manually by hand.

Where it genuinely helped: it caught bugs I don't think I would have been able to find on my own in the time I had, especially the footnote-corruption issue. That one is subtle — it doesn't look like an error, it just quietly produces the wrong answer under a key that looks correct — and it only became obvious by literally reading the source PDF page next to the JSON output and noticing the text didn't match. It also caught the two-tables-on-one-page situation on protocol 5, which the assignment calls out directly but which is easy to miss unless you're specifically looking for a second schedule.

Where I had to push back or double-check it: one of the first fixes it suggested for the footnote-key problem was to just keep whichever definition got scanned first and ignore anything that tried to redefine the same marker later. That sounded reasonable, but when I tested it against a different protocol, it turned out to be wrong in exactly the opposite direction — on that document, the real definition was the one that came later, and an unrelated match earlier in the page order would have won instead. That only got caught because I insisted on testing every fix against a second real document instead of accepting the first plausible-sounding explanation. I think that's actually the most important thing I took away from working this way: a fix that sounds right and a fix that's actually right aren't always the same thing, and the only way I found to tell the difference was checking against real source PDFs every single time, not just trusting the reasoning.

Every one of the five protocol outputs in this submission was checked by hand against its source PDF before I considered it complete.