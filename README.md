# Intelligent Land Record Digitization and Validation System

**Smart India Hackathon 2026 | Problem Statement 26018**
Ministry of Rural Development &middot; Department of Land Resources (DoLR)
Category: Software &middot; Theme: Smart Automation

---

## 1. The problem in one paragraph

India's land records exist as decades of handwritten and typed khatauni, khasra
and 7/12 extracts in a dozen scripts. Digitising them by hand is slow and
error-prone, and a wrong khasra number or a wrong owner name in a land record
is not a data-quality inconvenience: it is a property dispute. PS 26018 asks
for a system that reads these documents, classifies the content into structured
land-record fields, validates them against business rules and master data,
scores its own confidence, routes anything uncertain to a human, learns from
what the human corrects, and keeps a complete audit trail throughout.

## 2. Run it

```bash
python3 run.py
```

Then open **http://127.0.0.1:8000** and click **Load sample corpus** on the
Ingest tab.

Other entry points:

```bash
python3 run.py --check      # honest capability report for this machine
python3 run.py --samples    # regenerate the 13 sample documents
python3 run.py --port 9000  # different port
python3 run.py --host 0.0.0.0  # expose to the LAN for a projector/demo laptop
```

### There is nothing to install

The backend runs on the Python standard library alone: `http.server` for the
API and `sqlite3` for storage. The front-end is plain HTML, CSS and JavaScript
with no build step, no bundler and no CDN dependency.

This was a deliberate engineering decision, not laziness. The single largest
risk to a live hackathon demo is not a weak model, it is a venue with no
working internet and a `pip install` that fails ninety seconds before the
judges arrive. This project runs from a bare `python3 run.py` on any machine
with Python 3.8 or newer.

Optional libraries are used when they happen to be present and degraded
cleanly when they are not:

| Library | Used for | If missing |
| --- | --- | --- |
| PyMuPDF | Native PDF text-layer extraction | Falls back to the OCR path |
| OpenCV | Deskew, denoise, quality scoring | Quality metrics are skipped |
| pytesseract + tesseract | OCR of scanned images | **Degraded mode** (below) |

## 3. Honest degraded mode

Most hackathon OCR projects fail silently: no OCR engine, so the extractor
returns empty strings, and the dashboard cheerfully reports "0 errors".

This system refuses to do that. When no OCR engine is available, a scanned
image is recorded with `engine = degraded_no_ocr`, a trust score of 0, and an
explicit blocking error stating that the document requires manual entry. It
appears in the verification queue as work to be done, not as a success.

**A system that knows what it does not know is worth more to a revenue office
than one that guesses confidently.** Three of the bundled samples exercise this
path on purpose, so the behaviour is visible in the demo rather than hidden.

## 4. Architecture

```
                    +-------------------------------------+
  PDF / PNG / TIFF  |  ocr_engine.py                      |
  ----------------> |  quality assessment -> extraction   |
                    |  3 tiers: pdf text layer /          |
                    |  tesseract OCR / degraded mode      |
                    +------------------+------------------+
                                       | lines + per-line OCR confidence
                                       v
                    +-------------------------------------+
                    |  field_extractor.py                 |
                    |  17 land-record fields              |
                    |  bilingual label matching, fuzzy    |
                    |  fallback, per-field confidence     |
                    +------------------+------------------+
                                       | structured fields
                                       v
                    +-------------------------------------+
                    |  validator.py                       |
                    |  10 rule families, LGD master       |
                    |  cross-check, duplicate detection,  |
                    |  trust score -> routing decision    |
                    +------------------+------------------+
                                       v
         +-----------------------------+-----------------------------+
         |                             |                             |
         v                             v                             v
  auto-approved              needs human review                  blocked
         |                             |                             |
         +-------------> db.py (SQLite + append-only audit) <--------+
                                       |
                                       v
                            learning.py (corrections ->
                            confusions, aliases, calibration)
```

| File | Responsibility |
| --- | --- |
| `backend/ocr_engine.py` | Quality assessment and the three-tier extraction ladder |
| `backend/field_extractor.py` | Field classification and confidence scoring |
| `backend/validator.py` | Business rules, master-data cross-check, trust score |
| `backend/db.py` | SQLite schema, queries, audit trail, statistics |
| `backend/learning.py` | Mining correction signals into an improving model |
| `backend/server.py` | Stdlib HTTP API, RBAC, static file serving |
| `backend/data/admin_master.json` | LGD-style state / district / tehsil master data |
| `frontend/` | Ingest, queue, verification workspace, dashboard, learning, audit |
| `tools/make_samples.py` | Generates the sample corpus with planted edge cases |

## 5. The seventeen fields

`khasra_number`, `khata_number`, `survey_number`, `ulpin`, `owner_name`,
`father_name`, `share`, `area`, `land_classification`, `village`, `tehsil`,
`district`, `state`, `mutation_number`, `mutation_date`,
`registration_number`, `registration_date`.

Six are required: khasra number, khata number, owner name, area, village and
district. A record missing any of them cannot be approved.

## 6. Confidence scoring

Every field carries a confidence that is a weighted product of three
independent signals, so the number means something:

| Signal | Weight | What it measures |
| --- | --- | --- |
| Label match | 0.40 | How cleanly the field label was identified, and whether it sat on a token boundary |
| Pattern match | 0.38 | Whether the value looks like what this field should contain |
| OCR confidence | 0.22 | The engine's own per-line character confidence |

Anything below **0.80** is flagged for human review. The breakdown is stored
per field, so a reviewer can see *why* the system was unsure rather than being
handed a bare percentage.

Two scoring penalties matter in practice. A candidate with no parseable value
is multiplied by 0.35. A line with no `:` separator is multiplied by 0.55,
because a page heading like "उत्तर प्रदेश शासन - राजस्व परिषद" contains the
word "state" but is not a state field, and without this penalty the heading
outranks the real `राज्य / State :` row.

## 7. Validation rules

| Rule | Severity | What it catches |
| --- | --- | --- |
| `REQUIRED_MISSING` | error | A mandatory field was not extracted |
| `DUPLICATE_CONFLICT` | error | Same parcel already recorded with a different owner |
| `AREA_UNIT_MISSING` | error | Area with no unit, so it cannot be standardised |
| `AREA_REGIONAL_UNIT` | warning | Bigha/biswa recorded; conversion is region-dependent |
| `DATE_FUTURE` | error | A date in the future |
| `DATE_ORDER` | error | Registration after mutation |
| `DATE_TOO_OLD` | warning | Pre-independence date needing archival verification |
| `DISTRICT_UNKNOWN` | warning | District absent from the LGD master, with the closest match suggested |
| `TEHSIL_MISMATCH` | warning | Tehsil does not belong to the stated district |
| `OWNER_FATHER_SAME` | error | Owner and father recorded as the same person |
| `SHARE_OVER_UNITY` | error | Ownership share greater than 1 |
| `CLASS_UNMAPPED` | warning | Land classification outside the controlled vocabulary |
| `LOW_CONFIDENCE` | warning | Field extracted below the review threshold |

Rules produce a **trust score** out of 100, which drives routing:
auto-approve, needs review, or blocked. A record with any blocking error
cannot be approved through the API at all, whatever the reviewer clicks.

### Regional units are flagged, not silently converted

`2 बीघा 10 बिस्वा` is parsed into components and converted to 6,322.98 m²
using the common UP/Bihar values, **and flagged**, because a bigha is not the
same size in Rajasthan as it is in Bihar. Hiding that ambiguity behind a
confident-looking number would be the wrong engineering choice for a legal
record.

## 8. Role-based access control

| Role | Rights |
| --- | --- |
| Data Entry Operator | upload, correct |
| Revenue Inspector (verifier) | + approve, reject, revalidate |
| District Land Records Officer (admin) | + retrain, export, purge |
| State Audit Cell (auditor) | read-only |

Rights are enforced server-side on every route. The role switcher in the UI is
a demo convenience; removing it does not remove the enforcement.

## 9. The learning loop

Every human correction is stored as a training signal with the AI value, the
human value, the confidence the model had, and the source line. `learning.py`
mines three things from the accumulated signals:

1. **Character confusions** &mdash; systematic misreads, applied to future numeric
   fields.
2. **Value aliases** &mdash; repeated identical corrections, for example a village
   name the OCR consistently mangles.
3. **Confidence calibration** &mdash; a per-field multiplier derived from observed
   precision on reviewed fields.

Rules activate only after crossing an evidence threshold (4 sightings for a
confusion, 3 for an alias, 8 reviewed fields for calibration), so one typo
never reshapes the model.

**Extraction precision is measured only on fields a human has actually
reviewed.** The system is not allowed to grade its own homework, which is why
the dashboard shows a dash instead of a number until real review data exists.

## 10. Sample corpus

`tools/make_samples.py` generates 13 documents: 10 PDFs with a real text layer
and 3 deliberately degraded raster scans. Each one plants a specific edge case
so the demo exercises the rules rather than describing them.

| Sample | What it demonstrates |
| --- | --- |
| 01 khatauni UP clean | Clean bilingual extraction, auto-approved, trust 94 |
| 02 khasra MP Hindi | Hindi-only labels, Devanagari digits `९०७३` |
| 03 duplicate conflict | Same parcel, different owner &rarr; `DUPLICATE_CONFLICT` |
| 04 bigha/biswa Rajasthan | Composite regional units &rarr; 6,322.98 m², flagged |
| 05 missing unit | Area with no unit, unmapped classification |
| 06 bad dates | Future dates and reversed order &rarr; trust 26, blocked |
| 07 unknown district | `Kanpurr Nagar` &rarr; suggests `Kanpur Nagar` |
| 08 ownership conflict | Owner equals father, share `7/5` |
| 09 Maharashtra 7/12 | Survey number `142/2B`, Marathi classification |
| 10 historical 1938 | Pre-independence record |
| scan 01 / 02 / 09 | Skew, blur, fade &rarr; degraded mode, honest failure |

## 11. API

All routes accept an `X-User` header identifying the actor.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/session` | Current user, rights, machine capabilities |
| GET | `/api/schema` | The 17 field definitions |
| GET | `/api/documents` | Queue, with status filter and search |
| GET | `/api/documents/{id}` | Full record: fields, issues, quality, audit |
| GET | `/api/documents/{id}/preview` | Rendered page image |
| GET | `/api/documents/{id}/text` | Raw extracted text |
| POST | `/api/upload` | Multipart ingestion |
| POST | `/api/seed` | Ingest the bundled sample corpus |
| POST | `/api/documents/{id}/fields` | Correct or confirm a field, then revalidate |
| POST | `/api/documents/{id}/approve` | Approve (blocked if errors remain) |
| POST | `/api/documents/{id}/reject` | Reject with a mandatory reason |
| GET | `/api/stats` | Dashboard aggregates |
| GET | `/api/audit` | Append-only audit trail |
| GET | `/api/learning` | Model state and recent corrections |
| POST | `/api/learning/retrain` | Retrain from accumulated corrections |
| GET | `/api/export/csv`, `/api/export/json` | Structured export for LRMS/DILRMP |

## 12. Integration path

The export endpoints emit the full validated record set with confidence and
audit metadata, which is the shape an LRMS or DILRMP ingestion job expects.
ULPIN is a first-class field, so parcels can be joined to cadastral geometry
without a second matching step. What is deliberately *not* claimed: this
project does not ship a live DILRMP connector, because that requires
credentials and a state-level integration agreement that no hackathon team
has. The seam is built; the handshake is future work.

## 13. Honest limitations

- **No OCR engine in this sandbox.** The OCR tier is implemented and pluggable,
  but it was validated against the PDF text-layer path. On a machine with
  tesseract and the `hin`/`eng` packs installed, `python3 run.py --check` will
  report the OCR tier as available.
- **Handwritten records are out of scope.** The system is built for typed and
  printed records. Handwriting needs a trained HTR model and labelled data
  neither of which exists for these forms yet.
- **Bigha/biswa conversion is regional.** The values used are UP/Bihar. The
  flag exists precisely because a single national constant would be wrong.
- **Master data is a representative subset**, covering 8 states, not the full
  LGD directory.
- **Precision figures start empty.** Until a human reviews fields, the
  dashboard shows a dash. This is intentional.

## 14. Project layout

```
sih26018/
  run.py                     launcher, capability check, sample generation
  README.md
  backend/
    server.py                stdlib HTTP API + RBAC + static serving
    ocr_engine.py            quality assessment, 3-tier extraction
    field_extractor.py       17 fields, bilingual labels, confidence
    validator.py             rules, master-data check, trust score
    db.py                    SQLite schema, audit trail, statistics
    learning.py              correction mining and calibration
    data/admin_master.json   LGD-style master data (8 states)
  frontend/
    index.html  styles.css  app.js
  tools/
    make_samples.py          sample corpus generator
  samples/                   13 generated documents
  storage/                   created at runtime: SQLite DB, uploads, previews
```

`storage/` is created on first run and can be deleted at any time to reset the
demo to a clean state.
