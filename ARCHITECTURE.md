# System Architecture

**Intelligent Land Record Digitization and Validation System**
Smart India Hackathon 2026 · Problem Statement 26018 · Ministry of Rural Development, DoLR

How a land record travels from an uploaded file to a verified entry in the
register, and which of the 25 backend modules touches it on the way.

Every figure here was measured on the running system, not estimated.

| | |
| --- | --- |
| Backend modules | 25 |
| Backend lines | 15,248 |
| Tests | 722 across 31 files |
| Fields extracted | 17 |
| Validation rules | 66 |
| Upload formats accepted | 20 |

---

## 1. The shape of it

**One pipeline, not several.** Every upload converges on the same extractor and
the same rule engine, whatever format it arrived in and whichever engine read
it. Formats differ in exactly one place: the reader chosen at stage 2.

The backend is **Python standard library only** at its core — an HTTP server
and SQLite, no framework and no build step. Every heavy capability sits behind
an optional import and degrades with an explicit message rather than failing.
That is what makes `python3 run.py` work on a demo laptop with nothing
installed, and it is why the offline story is real rather than aspirational.

The frontend is plain HTML, CSS and JavaScript with Leaflet vendored locally.
No build, no CDN, no network required.

---

## 2. The document journey

Fourteen stages in a genuine sequence: a document enters at one and leaves at
fourteen. Stages marked *optional* skip cleanly when their dependency is absent.

### 01 · Intake — `db.py`
SHA-256 hash, checked against every previous upload. A duplicate does not stop
processing; it becomes a warning and demotes an auto-approval to review.

### 02 · Route by format — `ocr_engine.py`
Four readers, chosen by file extension. **This is the only place formats
differ.**

| Reader | Handles |
| --- | --- |
| PDF text layer | Tried first. Full accuracy, no OCR. This is where 99.3% precision comes from. |
| Tesseract OCR | Images, and PDFs with no text layer. 14 Indic language packs. |
| `office_reader.py` | DOCX, XLSX, ODT, ODS, PPTX, CSV, TSV, HTML, RTF — ZIP plus XML, read with the standard library. |
| Manual queue | No engine available, or nothing readable. Recorded as work to be done, never as an empty success. |

### 03 · Restore the image — `cnn_denoiser.py`
Scans only. Quality assessment, deskew above a measured 2° threshold, CNN
denoise falling back to non-local means, then Otsu plus adaptive binarisation.
Sharpness is measured *after* a median blur so scan noise cannot pass as detail.

### 04 · Read the text — `ocr_engine.py`
Tesseract with 14 Indic language packs. A PDF carrying a text layer skips this
entirely and extracts at full accuracy.

### 05 · Detect handwriting — `handwriting.py`
Four geometric features per line. A line judged handwritten has its confidence
suppressed, because Tesseract does not fail loudly on cursive — it returns
confident nonsense.

### 06 · Transcribe handwriting — `trocr_htr.py` *(optional, Latin only)*
TrOCR through ONNX Runtime, kept alongside the Tesseract reading rather than
replacing it. Refuses non-Latin script outright: its checkpoints are IAM
English.

### 07 · Extract 17 fields — `field_extractor.py`
Label-anchored matching over **226 bilingual label aliases**, each validated
against its field's value pattern. Native digits folded to ASCII across seven
scripts.

### 08 · Apply what was learned — `learning.py` *(optional)*
Value aliases, character-confusion repair and confidence recalibration, derived
from verifier corrections. Runs *before* the gazetteer so a learned alias gets
first refusal on a value.

### 09 · Bridge the script — `bhashini.py` *(opt-in)*
Devanagari place names transliterated to the Latin spelling the administrative
master uses, with an exonym table for names no model can derive. Took district
resolution from **0 of 43 to 43 of 43**.

### 10 · Correct vocabulary — `gazetteer.py`
Closed-vocabulary repair against the LGD master, land classes and identifier
shapes, before the rules see the text.

### 11 · Classify the document — `doc_type.py`
Five types plus an explicit unknown, by weighted keyword scoring. Scoping
required fields to the detected type took a real Power of Attorney from 4
spurious errors and trust 27.4, to 1 real error and trust 69.4.

### 12 · Validate — `validator.py`
13 rule functions, scoped by document type, plus duplicate detection against
every ingested record.

### 13 · Enrich and place — `geocode.py`, `cadastral.py`
Seal and signature presence, table structure, geotagging, anomaly detection.
Each appends issues rather than replacing them.

### 14 · Persist and route — `db.py`
Document, then every field, then a hash-chained audit entry. Auto-approved,
needs review, or blocked.

---

## 3. Where a record ends up on the earth

Two tiers, deliberately kept apart so one can never be mistaken for the other.

| Tier | Modules | Source of position | Stated accuracy |
| --- | --- | --- | --- |
| **Parcel-grade** | `cadastral.py`, `georeference.py` | Khasra number matched to a polygon on a georeferenced map. Transform from a QGIS world file, GeoTIFF tags, or fitted ground control points. | metres, RMS reported |
| **Place name** | `geocode.py` | The record's own village, tehsil or district, looked up in a table of 36 states, 106 districts, 9 localities. | 5 km locality, 40 km district, 300 km state |

**Why the separation is load-bearing.** The place-name tier carries its own
rule name and never the parcel-grade one, so a district centroid can never be
read as a plot location. It is informational only: it never blocks a record and
never validates an area, because a coordinate derived from a name contains no
new information about the parcel.

Parcel geometry always wins. The place-name tier runs only when no parcel
matched.

---

## 4. Modules by role

Twenty-five backend modules. The ones that carry the most weight are not the
largest.

| Role | Modules | Lines |
| --- | --- | --- |
| **Serving** | `server.py`, `db.py` | 3,048 |
| **Reading documents** | `ocr_engine.py`, `office_reader.py`, `handwriting.py`, `trocr_htr.py`, `cnn_denoiser.py` | 2,840 |
| **Structuring** | `field_extractor.py`, `gazetteer.py`, `doc_type.py` | 2,509 |
| **Geospatial** | `cadastral.py`, `shapefile_import.py`, `topology.py`, `georeference.py`, `geocode.py`, `boundary_net.py`, `sam_fallback.py`, `postgis.py` | 3,181 |
| **Validating** | `validator.py`, `fact_checker.py`, `document_authenticity.py`, `ner_extractor.py` | 1,647 |
| **Improving** | `learning.py`, `bhashini.py`, `llm_extractor.py` | 1,142 |

---

## 5. Data and governance

### Storage
SQLite by default, PostgreSQL when `DATABASE_URL` is set. Queries are written
once in SQLite's dialect and translated centrally, so **no caller knows which
engine answered**. The same five documents ingested into a fresh database on
each engine produce byte-identical decisions, trust scores and field counts.

### The audit trail
Every audit row stores the SHA-256 of its own content combined with the
previous row's hash. Editing or deleting a past entry breaks the link every
later row depends on, and verification names the **first** row that no longer
agrees. On PostgreSQL the append also takes a transaction-scoped advisory lock,
because a process lock cannot serialise writers living in different processes.

> **Tamper-evident, not tamper-proof.** Someone with write access to the
> database can recompute the chain from their edit onward and it will verify
> clean. What the chain buys is that tampering can no longer be *silent* — it
> must be deliberate and complete. A copy of the tip hash held anywhere outside
> the database makes even a full recompute detectable.
>
> This is deliberately **not a blockchain**. There is no distributed consensus
> here, and claiming otherwise would be the one dishonest component in a system
> built around not overclaiming.

### Access
Four roles — operator, verifier, admin, auditor — enforced server-side on every
route. The chain verification endpoint is open to all of them, auditor
included: an integrity check only an administrator may run is worth little,
because the administrator is who an auditor is checking.

---

## 6. What runs, and what does not

Optional layers listed with their real state rather than their intended state.

| Capability | State | Detail |
| --- | --- | --- |
| Tesseract OCR | **Active** | 14 Indic language packs |
| PDF text layer | **Active** | PyMuPDF |
| Fact check, anomaly detection | **Active** | scikit-learn |
| Bhashini script bridge | **Opt-in** | Sends place names to a government service; off without consent |
| Indic NER | **Local only** | Needs PyTorch, excluded from the deployment image |
| SAM parcel fallback | **Off** | Fires only when contour tracing finds zero parcels |
| PostGIS | **Verified** | Exercised against PostGIS 3.4: geodesic area, OGC validity, overlap and overlapping-claim detection |
| Donut | **Not used** | Its vocabulary maps Devanagari to `<unk>` |
| Live portal integration | **None** | Export is file-based: CSV, JSON, GeoJSON |

---

## 7. Three decisions that shaped the rest

**Rules before models.** The 17 fields are highly templated across khatauni,
jamabandi and record-of-rights formats. Rules are auditable in a government
context, need no GPU and degrade predictably. The measured result is 99.3%
precision and 100% recall on the digital corpus. Machine learning is used only
where rules genuinely cannot work, and each such choice was measured rather
than assumed — SAM, MuRIL and Donut were all tested against this project's own
data and rejected on the numbers.

**Refusing beats guessing.** No OCR engine means a document is queued for
manual entry, not extracted as empty. Non-Latin handwriting is refused rather
than transcribed into confident English. A projected coordinate system is
refused by name rather than misread as degrees. An unmatched village keeps the
spelling the OCR read.

**Every optional layer announces itself.** `run.py --check` prints what this
machine can actually do, including what it cannot. A clean record never means
"we checked and found nothing" when it actually means "we could not check."
