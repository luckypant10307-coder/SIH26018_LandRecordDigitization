# System Architecture

**Intelligent Land Record Digitization and Validation System**
Smart India Hackathon 2026 · Ministry of Rural Development, DoLR
Problem Statement **26011** — 3D ULPIN Generation and Vertical Property Mapping,
built on the record-digitization pipeline written for **26018**.

How a land record travels from an uploaded file to a verified entry in the
register, how the parcel it describes is placed on the earth, and how the
volumes stacked on that parcel get identifiers of their own.

**Why one system covers both.** A 3D cadastre cannot start from a 3D model; it
starts from a parcel that has been read, validated and located, because a
volume identifier is derived from the parcel identifier and a footprint. The
digitization pipeline is therefore not a previous project that was set aside —
it is the input stage of this one. Section 8 is the part that is new.

Every figure here was measured on the running system, not estimated.

| | |
| --- | --- |
| Backend modules | 31 |
| Backend lines | 18,169 |
| Tests | 779 across 31 files (802 with a live PostGIS) |
| Fields extracted | 17 |
| Validation rules | 68 |
| Upload formats accepted | 20 |
| 3D conflict checks | 3 (overlap · duplicate identifier · unpartitioned level) |

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

**It is no longer one line.** A Bhu-Naksha plot report is four kinds of
content in one file - structured text, a table of co-owners, mutation orders
in prose, and the parcel map itself - so three readers run over the SAME
document and a merge reconciles what they return. Fan-out, not routing.

```
                         ONE DOCUMENT
                              │
          ┌───────────────────┼───────────────────┐
     ① TEXT               ② MAP              ③ HANDWRITING
     01-10 below          11 below           05-06 below
          └───────────────────┼───────────────────┘
                              │
                        12 · MERGE
                verified > grounded > unverified
                   disagreement = a finding
                              │
                     13-16 · validate, place, persist
```

Sixteen stages, numbered in the order a document meets them. Stages marked
*optional* skip cleanly when their dependency is absent.

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

### 10b · Recover from prose — `gazetteer.py`, `field_extractor.py`
A mutation order is prose, and its dates and places sit mid-sentence with no
label. A label-anchored extractor cannot see them. Anchoring on the
administrative noun instead - `जिला`, `तहसील`, `नि0` for *resident of* - took
dates from 7 of 20 to 20 of 20 and villages from 0 to 10. A date inside a
case number is an identifier, not a date, and is excluded.

### 11 · Read the parcel map — `parcel_map.py`
**The second reader.** A plot report embeds its own map, and 20 of 20 real
documents carry one. `cadastral.py` returns zero parcels on these: Otsu picks
a threshold that classifies the light blue boundary lines as background.
A portal map is a vector render, not ink on paper, so ink is detected by
COLOUR - saturated or dark - which took ink from 0.15% of pixels to 6.9%.

The subject parcel is drawn FILLED, so it needs no closed boundary: 24,229 px
and 22 vertices on a real report. Its khasra label and its neighbours are read
by isolating dark-unsaturated text, which took Tesseract from `['a,', 'Sr)']`
to six of seven plot numbers.

### 11b · Structured extraction — `llm_extractor.py` *(opt-in)*
For the shape rules cannot express. 162 owner rows across 20 documents face a
schema holding one `owner_name`. Sarvam returns the owner LIST; every value
must appear verbatim in the source or it is dropped.

### 12 · Merge — `server.py`
Where the readers are reconciled. A value CONFIRMED against the administrative
master outranks one that is not; a grounded value outranks a
label-contaminated one. The map's khasra is compared with the text's - the two
readers take it from different places, so agreement is evidence neither gives
alone.

### 12b · Classify the document — `doc_type.py`
Six types plus an explicit unknown, by weighted keyword scoring. A Bhu-Naksha **plot report** is its own type: it is cut from the map and carries no village or district, and demanding them blocked 18 of 20 real documents. Scoping
required fields to the detected type took a real Power of Attorney from 4
spurious errors and trust 27.4, to 1 real error and trust 69.4.

### 13 · Validate — `validator.py`
13 rule functions, scoped by document type, plus duplicate detection against
every ingested record.

### 14 · Enrich and place — `geocode.py`, `geocode_online.py`, `cadastral.py`
Seal and signature presence, table structure, geotagging, anomaly detection.
Each appends issues rather than replacing them.

### 15 · Persist and route — `db.py`
Document, then every field, then a hash-chained audit entry. The co-owner
list and the map reading are stored as data of their own, because a parcel
with sixteen claimants flattened to one name loses fifteen people with a
claim on the land. Auto-approved, needs review, or blocked.

### 16 · Authenticate the caller — `auth.py`
Not a stage a document passes, but the gate every request does. The API read
its caller from an `X-User` header anyone could set; the Supabase access
token is now verified in stdlib HMAC-SHA256, HS256 only, other algorithms
refused by name. With no secret configured nothing can be verified, so the
header path remains and `run.py --check` says so.

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

Thirty-one backend modules. The ones that carry the most weight are not the
largest.

| Role | Modules | Lines |
| --- | --- | --- |
| **Serving** | `server.py`, `db.py`, `auth.py` | 3,798 |
| **Reading documents** | `ocr_engine.py`, `office_reader.py`, `handwriting.py`, `trocr_htr.py`, `cnn_denoiser.py`, `table_structure.py` | 3,587 |
| **Structuring** | `field_extractor.py`, `gazetteer.py`, `doc_type.py` | 2,932 |
| **Geospatial** | `cadastral.py`, `parcel_map.py`, `shapefile_import.py`, `topology.py`, `georeference.py`, `geocode.py`, `geocode_online.py`, `boundary_net.py`, `sam_fallback.py`, `postgis.py` | 3,982 |
| **Validating** | `validator.py`, `fact_checker.py`, `document_authenticity.py`, `ner_extractor.py`, `anomaly_detector.py` | 1,837 |
| **Improving** | `learning.py`, `bhashini.py`, `llm_extractor.py` | 1,356 |
| **Third dimension** | `vertical.py` | 451 |

---

## 4b. What is a model here, and what is not

Worth stating because the names invite the wrong assumption.

| Called by the code | How |
| --- | --- |
| **PostGIS** | via psycopg - geodesic area, OGC validity, overlapping claims |
| **Tesseract** | the only model that reads a map, picking khasra labels off the drawing |
| **OpenCV** | contour tracing, ink detection, deskew, binarisation |

| NOT called by the code | What it really is |
| --- | --- |
| **QGIS** | How a person produced `village_map_narharpur.pgw`. A preparation step. The backend contains no reference to it. |
| **GDAL / ogr2ogr** | Installed on the development machine, never invoked. |
| **Esri / ArcGIS** | Basemap tiles fetched by the BROWSER through Leaflet. The backend never contacts it, and no record data is in a tile request. |

Parcel tracing is **classical computer vision**, not a model, and
georeferencing is an affine least-squares fit in pure Python -
`georeference.py` imports nothing at all. The two neural options are both
inactive: `boundary_net.py` needs torch and is excluded from the deployment
image, and `sam_fallback.py` was measured worse than tracing.

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
| Embedded parcel map | **Active** | Read from the document itself on 20 of 20 real reports |
| Supabase token verification | **Active when configured** | Without a JWT secret nothing can be verified, so the header path remains and `--check` says so |
| Satellite basemap | **Off by default** | Esri public tiles, no API key. Off because every other part of the map works with no internet |
| Online geocoding | **Opt-in** | `ONLINE_GEOCODING=1`. Takes a district from 300 km to 40 km; villages still out of reach |
| Structured LLM extraction | **Opt-in** | Sarvam. The co-owner list; every value must appear verbatim in the source |
| Bhashini script bridge | **Opt-in** | Sends place names to a government service; off without consent |
| Indic NER | **Local only** | Needs PyTorch, excluded from the deployment image |
| SAM parcel fallback | **Off** | Fires only when contour tracing finds zero parcels |
| PostGIS | **Verified** | Exercised against PostGIS 3.4: geodesic area, OGC validity, overlap and overlapping-claim detection |
| Donut | **Not used** | Its vocabulary maps Devanagari to `<unk>` |
| LayoutLMv3 / LayoutXLM | **Not used** | LayoutLMv3 shreds Devanagari into 61 byte fragments; LayoutXLM needs a fine-tune that has not run |
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

---

## 8. The third dimension — `vertical.py`

This is the PS 26011 layer. Everything above it produces a validated parcel;
this turns that parcel into a stack of separately-identified volumes.

### Why a 2D cadastre cannot answer the question

A parcel record answers *who owns this ground*. It cannot answer *who owns the
third floor*, *who owns the parking two levels down*, or *who owns the air
above the road* — and in a vertical city those are most of the disputes. One
surface parcel may carry a hundred separately-owned volumes, and the existing
ULPIN identifies the parcel, not any of them.

### The identifier

```
UP091223700412-F03-012
└────────────┘ └─┘ └─┘
 parent ULPIN   |   unit on that level, 001-999
 14 chars,      |
 DILRMP 3.0     level: B.. basement · G00 ground · F.. floor
                       A.. air rights · S.. subsurface utility
```

Four properties, each chosen against a specific failure:

| Property | The failure it avoids |
| --- | --- |
| **Derived** — visibly a child of its parcel | An identifier that cannot be traced to its land is useless in a dispute about that land |
| **Deterministic** — no allocator, no counter, no clock | A registry that needs a central server to mint an ID cannot work in a tehsil office with intermittent power |
| **Reversible** — parses back to parcel, level, unit | An opaque hash would satisfy uniqueness and defeat every officer who has to read it off a document |
| **Collision-free within the parcel** | All that is required: the parent ULPIN already separates parcels |

**The level letters are mnemonic, not ordinal.** `F` sorts before `G` in ASCII,
so a plain lexicographic sort puts the third floor below the ground floor.
Readability was worth more than a free sort, so the format keeps the letters a
revenue officer actually reads and `level_sort_key` supplies the physical
order — subsurface, basements deepest-first, ground, floors, air rights. The
frontend re-implements that same ordering, and a test asserts the two agree.

**This is a proposal, not a published standard**, and the code says so wherever
it surfaces. DILRMP 3.0 specifies the 14-character parcel ULPIN and does not
yet specify a vertical extension. The parent comes from the official spec and
only the suffix is ours, separated by a hyphen so the official part stays
extractable. Presenting an invented scheme as a government one would be the
same overclaim this project refuses everywhere else.

### Geometry without a survey

There is no LiDAR here, no drone imagery and no floor plans, so a unit's volume
cannot be measured. It is **declared**: the footprint comes from the parcel
polygon the map reader already recovered, the heights from a nominal storey
height, and every such volume carries `surveyed=False` plus a note saying so in
its own serialised form — so the caveat travels with the data instead of living
in a document nobody reads.

A declared volume is still useful. It detects two units claiming one space,
which is the question a registry has to answer, and it does so without
pretending to a precision nobody measured.

### Conflict detection, and the third answer

A conflict requires **both** a shared footprint and a shared height range.
Either alone is ordinary:

- flats on one landing share a height range and sit side by side
- flats in one column share a footprint and sit one above another
- a flat's ceiling *is* the next flat's floor, so `top == base` on every storey
  of every building — touching is not overlapping

Getting this wrong is not a cosmetic bug. A detector that flags real buildings
is worse than none, because a reviewer learns to dismiss it. Most of the 35
tests in `tests/test_vertical.py` therefore assert **negatives**.

There is a third case, and it is the one worth defending in a review. Several
units on one level, each inheriting the whole parcel as its footprint because
the document carries no floor plan, are **neither** a conflict **nor** cleared.
Calling them an overlap flags every ordinary block of flats. Calling them clean
asserts a separation nobody verified. So they are reported once per level as
`LEVEL_NOT_PARTITIONED` — severity *info*, a stated gap rather than a verdict:

> *Level F01 of UP091223700412 holds 3 units, each recorded with the whole
> parcel as its footprint because the document carries no floor plan. Whether
> they overlap cannot be determined from this record.*

That distinction cannot come from the coordinates, because two flats on an
undivided level are geometrically identical to two owners sold the same flat.
It is a **declared fact**, carried on the volume as `footprint_is_parcel` and
persisted in its own column — a flag that must survive the database round trip,
because without it every multi-unit level reloads as a pile of false overlaps.

### API

| | |
| --- | --- |
| `GET /api/vertical?parcel=…` / `?document=…` | Stored volumes plus findings recomputed across the set |
| `POST /api/vertical/generate` | Declare a stack: parent ULPIN, floors, basements, units per level, storey height |

**Conflicts are never cached.** Two buildings become a conflict the moment the
second one is registered, and a stored verdict from before that would call the
first one clean — exactly the moment a volumetric register has to speak up. So
findings are recomputed on every read.

**Regeneration replaces by `ulpin_3d`.** Correcting a floor count must not
leave the old stack sitting beside the new one, which would read as every unit
being claimed twice.

### What this layer does not do

- **It does not invent a parent ULPIN.** Bhu-Naksha plot reports do not print
  one, so the endpoint refuses rather than composing an official-looking
  identifier from the fields it has. A fabricated ULPIN that looks allocated is
  worse than a missing one.
- **It does not subdivide a floor.** Splitting a footprint into N equal parts
  would manufacture boundaries and then report confident non-overlap based on
  them. See `LEVEL_NOT_PARTITIONED` above.
- **It does not know metres when the sheet has no control points.** The
  footprint is then in map pixels. Overlap detection stays valid, because it is
  geometric and unit-free; floor area in square metres does not, and the API
  labels the coordinate space rather than letting the number be read as metres.
- **It is not stored in PostGIS yet.** Footprints are JSON, for the same reason
  the rest of the system stores geometry that way — SQLite has no geometry type
  and the system must run without a database server. PostGIS adds indexed
  geometry when it is present.
