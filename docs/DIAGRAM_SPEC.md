# System Architecture Diagram — content specification

Paste-ready text for the seven-stage horizontal diagram. Keep the existing
layout, colour scheme and learning-loop footer; replace only the content.

**Every claim below is verified against the running code.** Anything that is
not yet true is marked *(planned)* or *(in progress)* rather than stated
flatly — a judge checks the diagram against the demo, and one false box costs
more credibility than three missing ones.

---

## Stage 1 — Access & Intake
*Colour: blue*

**Sign in**
- Supabase Auth (UI session)
- 4 roles, enforced server-side on every route

**Upload**
- 20 formats: PDF · scans · DOCX · XLSX · ODT · ODS · PPTX · CSV · HTML · RTF
- SHA-256 dedup against every prior upload

**Screen**
- Virus scan — ClamAV *(planned)*

> **Why 20 formats matters.** A revenue office holds far more office documents
> than clean scans: a clerk's khatauni extract is usually .docx, a district
> parcel list .xlsx, and offices running LibreOffice produce .odt. Read with
> the Python standard library — zipfile + XML — so it adds no dependency.

---

## Stage 2 — Quality & Restoration
*Colour: teal*

**Quality gate**
- Blur: variance of Laplacian, measured *after* median blur
- Deskew above a measured 2° threshold
- Fail → routed to rescan, never extracted as empty

**Restoration**
- OpenCV · scikit-image
- CNN denoiser, falls back to non-local means
- Otsu + adaptive Gaussian binarisation

> **The ordering is the point.** Sharpness is measured after a median blur so
> that scan noise cannot masquerade as detail — an untouched Laplacian reads a
> noisy-but-blurred page as sharp, which is the exact failure this gate exists
> to catch.

---

## Stage 3 — Reading
*Colour: purple*

**OCR**
- Tesseract, 14 Indic scripts
- PDF text layer tried first — full accuracy, no OCR
- TrOCR handwriting *(Latin now; Devanagari fine-tune in progress)*

**Handwriting safeguard**
- 4 geometric features per line
- Detected handwriting → confidence suppressed → human review

**Field extraction**
- Label-anchored matching, 226 bilingual aliases
- 17 fields across 6 kinds — area, class, date, number, person, text
- Each field carries a decomposed confidence score
- Learned digit repair (4 → 1, from 14 verifier corrections)

> **Measured: 99.3% precision, 100% recall** on the digital corpus.
> Tesseract is trained on print; on cursive it does not fail loudly, it
> returns confident nonsense. That is why handwriting is *detected* and
> routed to a person rather than trusted.

---

## Stage 4 — Indic Language
*Colour: magenta*

**Transliteration**
- Bhashini (ULCA) — Devanagari place names → LGD Latin spelling
- Exonym table for names no model can derive (लखनऊ → Lucknow)

**Entity cross-check**
- XLM-R multilingual NER — Devanagari person and date spans
- spaCy `en_core_web_sm` — Latin script

> **Measured: district resolution 0 → 43 of 43.** The administrative master
> holds 65,869 Latin characters and zero Devanagari, so a correctly-read
> Devanagari district previously matched nothing. Transliteration alone
> reached 31 of 43; the exonym table closed the rest.

---

## Stage 5 — Spatial Intelligence
*Colour: green*

**Parcel extraction**
- Contour tracing — **primary**, follows the drawn survey line
- U-Net · SAM — fallbacks for degraded and unreadable sheets

**Georeferencing**
- GDAL · QGIS world files · GeoTIFF tags · ground control points
- RMS error reported in metres; projected CRS refused by name
- Leaflet map view

**Placing the record**
- Parcel-grade: khasra matched to polygon → metres
- Place-name: village/tehsil/district → 5 km / 40 km / 300 km, stated

> **Measured: 6 of 6 plots in 0.09 s** on real Bhu-Naksha reports. On a
> cadastral sheet the parcel edge *is* the surveyed line, so tracing it is
> exact by construction. SAM was tested and produced 4 of 6 with two parcels
> merged and 300–861 boundary vertices against tracing's 4–11, so it is kept
> strictly as a last resort for sheets where tracing returns nothing.

---

## Stage 6 — Three-Way Validation
*Colour: orange*

**Three checks**
- ① Against itself — 13 rule functions, 63 rule IDs
- ② Against other records — RapidFuzz names, exact plot ID, duplicate signature
- ③ Against an external registry — LGD fact-check, TF-IDF identity match

**Severity decides the route**
- 15 error · 19 warning · 27 info · 2 decided at runtime
- Severity, not score, is what blocks a document

**Trust score and routing**
- Weighted: khasra · owner · area
- Isolation Forest anomaly detection (200 estimators, unsupervised)
- All fields ≥ 0.80 → auto-approve
- Any warning → officer review · any error → blocked

> **Fuzzy retrieves, exact confirms.** A plot identifier must match exactly
> once separators and scripts are normalised. Fuzzy matching is right for
> names, which carry OCR noise, and wrong for plot numbers — 213/1 and 218/1
> differ by one character and are different land. Skipping that distinction
> once matched khasra 213/1 to registry entry 237/4 and blocked a valid
> document over another parcel's owner.

---

## Stage 7 — Data & Outputs
*Colour: navy*

**Store**
- SQLite *or* PostgreSQL, chosen by `DATABASE_URL`
- Hash-chained audit log — every row carries the previous row's SHA-256
- Tamper-*evident*, not tamper-proof — deliberately not a blockchain

**Identify**
- ULPIN / Bhu-Aadhaar — 14-char DILRMP format, **stored when present,
  never generated**

**Deliver**
- REST · CSV · JSON · GeoJSON (RFC 7946)
- GraphQL *(planned — no Indian government source exposes GraphQL yet;
  LGD, data.gov.in and Census are all REST or CSV)*

> **Verified byte-identical across both engines.** The same five documents
> ingested into a fresh SQLite and a fresh PostgreSQL produce identical
> decisions, trust scores, error counts and field-accuracy rows.

---

## Footer — Learning Loop
*Colour: purple, full width*

**↻ LEARNING LOOP** — Every officer correction feeds three derived artefacts:
a character-confusion map, a value-alias table, and per-field confidence
calibration. These improve reading (3), language (4) and checks (6).

Nothing is stored as learned state — all three are rebuilt on demand from the
audit-backed corrections table, so every adjustment can be traced to the
corrections that justify it and no model drifts silently.

---

## What changed from the previous version, and why

| Was | Now | Reason |
| --- | --- | --- |
| "Supabase Auth" | "Supabase Auth (UI session)" | The backend does not verify the Supabase token; it identifies users by header. Supabase protects the interface, RBAC protects the routes. |
| "LayoutXLM · Donut" | *removed* | Neither is imported. Donut's vocabulary maps Devanagari to `<unk>`, so it could not read these documents even if wired in. |
| "MuRIL / Indic NER" | "XLM-R multilingual NER" | MuRIL has no entity head and scored 5 of 8 when measured. IndicNER is a gated repo. XLM-R is what actually runs, at 6 of 6. |
| "GDAL · GeoPandas" | "GDAL · QGIS world files · GCPs" | GeoPandas is not installed. GDAL is used as a command-line tool, not a Python import. |
| "U-Net · SAM" | "Contour tracing (primary) · U-Net · SAM (fallback)" | The primary vectoriser was missing from the diagram. It is also the part that works best. |
| "22 rules" | "13 rule functions, 63 rules total" | 13 functions in `validator.py`; 63 distinct rule IDs across all modules. |
| "ULPIN 14-char ID from geo-coordinates" | "stored when present, never generated" | Generating a Bhu-Aadhaar would claim authority to mint a government identifier. The code deliberately does not. |
| "Scans, PDFs, maps" | "20 formats" | Understated. Office and web formats were added. |
| "Tesseract, 10+ scripts" | "14 Indic scripts" | Understated. |

---

## Every figure in the diagram, and where it comes from

If a judge asks "where does that number come from", this table answers it.
Each was re-counted from source at the time of writing, not carried over from
an earlier draft.

| Figure | Value | Source |
| --- | --- | --- |
| Upload formats | 20 | `ALLOWED_EXT`, `server.py` |
| Fields extracted | 17 | `FIELD_SPECS`, `field_extractor.py` |
| Field kinds | 6 | distinct `kind` across `FIELD_SPECS` |
| Label aliases | 226 | sum of `len(spec.labels)` |
| Roles | 4 | `ROLE_RIGHTS`, `server.py` |
| Rule functions | 13 | `rule_*` in `validator.py` |
| Rule IDs | 63 | distinct IDs across `validator.py`, `fact_checker.py`, `ner_extractor.py`, `server.py` |
| Severity split | 15 / 19 / 27 | error / warning / info, +2 resolved at runtime |
| Confidence threshold | 0.80 | `rule_low_confidence` default |
| Indic scripts | 14 | Tesseract language packs in the image |
| Deskew threshold | 2.0° | `MIN_DESKEW_DEGREES` |

**On the rule count.** The system contains 66 rule IDs if you include
`postgis.py`. That module is written but has never been run against a live
PostGIS instance, so the diagram claims **63** — the rules that ship and
execute. When PostGIS is verified, three geometry rules (`GEOMETRY_INVALID`,
`OVERLAPPING_CLAIM`, `PARCELS_OVERLAP`) move the figure to 66.

**On the two runtime severities.** `LOW_CONFIDENCE` and `PLACEHOLDER_VALUE`
choose their severity from the value they are judging rather than declaring it
statically — a field at 0.79 is a warning, one near zero is an error. Counting
them as fixed would misstate what the engine does.

---

## Three additions worth making if the diagram has room

Each is true today and none is currently shown.

**1. A refusal lane.** The strongest property of this system is what it declines
to do, and the diagram shows none of it. A thin band under stages 2–5 reading:

> **Refusing beats guessing** — no OCR engine → manual queue, never an empty
> success · non-Latin handwriting → refused, not transcribed · projected CRS →
> refused by name · unmatched village → keeps the OCR spelling

**2. Degradation as a property.** The core is Python standard library only —
HTTP server and SQLite, no framework, no build step. Every heavy capability sits
behind an optional import and degrades with an explicit message. That is why
`python3 run.py` works on a bare demo laptop, and it is what makes the offline
claim real rather than aspirational.

**3. `run.py --check`.** One command prints what the machine can actually do,
*including what it cannot*. This is the honesty mechanism behind the whole
design: a clean record never means "we checked and found nothing" when it
actually means "we could not check."
