# Architecture, for the presentation

**Problem Statement 26011 — 3D ULPIN Generation and Vertical Property
Mapping**, built on the record-digitization pipeline written for 26018.

Slide-ready content. Every figure was re-counted from the running system
on 2 Oct 2026 and measured on **20 genuine Bhu-Naksha documents**, not on
the synthetic corpus.

Supersedes the stage content in `DIAGRAM_SPEC.md`, which predates the map
pipeline, the language layer and the verified spatial database.

---

## SLIDE 1 — The one-line claim

> **A land record describes the ground. We read it, validate it, place it on
> the earth — and then give every floor standing on it an identifier of its
> own.**

Two halves, and the second needs the first. A 3D cadastre cannot start from a
3D model; it starts from a parcel that has been read, validated and located,
because a volume identifier is *derived* from the parcel identifier and a
footprint. The digitization pipeline is the input stage, not a previous
project.

A Bhu-Naksha plot report carries structured text, a table of co-owners,
mutation orders in prose, and **the parcel map itself**. Most systems read
the first. We read all four, cross-validate between them, and then build
upward from the polygon that comes out.

---

## SLIDE 2 — The architecture: three readers, one merge

This is the change worth showing. It is **fan-out and merge**, not routing:
the pipelines do not divide documents between them, they all read the
*same* document and their answers are reconciled.

```
                 ┌──────────────────────────────┐
                 │   ONE DOCUMENT (PDF/scan/    │
                 │   DOCX/XLSX — 20 formats)    │
                 └──────────────┬───────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
   ① TEXT                  ② MAP                   ③ HANDWRITING
   text layer → OCR        image inside the PDF    geometric detection
   226 label aliases       highlight = subject     → human review
   17 fields, 6 kinds      labels = neighbours     (TrOCR: Latin)
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │          MERGE               │
                 │  verified > grounded >       │
                 │  unverified                  │
                 │  disagreement = a finding    │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │  68 rules in 3 directions    │
                 │  within · across · registry  │
                 └──────────────┬───────────────┘
                                │
                  auto-approve · review · block
                                │
        ┌─────────────────────────────────────┐
        ▼                                     ▼
  officer corrects a field           2D PARCEL POLYGON
        │                                     │
        ▼                          ┌──────────────────────┐
 teaches the EXTRACTOR             ▼                      ▼
 character confusions ·      existing 2D GIS          ④ 3D MODULE
 value aliases ·             PostGIS overlap          PS 26011
 confidence calibration                                   │
                                                 footprint + levels +
                                                 storey height (DECLARED)
                                                          │
                                                 3D ULPIN per volume
                                                 vertical conflict check
                                                 floor view per level
```

Separately, and NOT from those corrections, the merge's own LLM output is
reused offline to label a corpus for a small local model — see Slide 7. The
two learning paths are independent: one learns from the officer, the other
from the teacher model. Keeping them apart matters, because only the first
is grounded in a human decision.

**The merge is the part to dwell on.** Reader ① and reader ② take the
khasra number from different places — one from a printed field, one from a
label drawn on a polygon — so they fail independently. Agreement between
them is evidence neither can give alone.

---

## SLIDE 3 — What's new, and why it matters

| | What it does | Why a judge should care |
| --- | --- | --- |
| **Map pipeline** | Reads the parcel map embedded in the PDF | Nobody expects the geometry to already be in the document. 20/20 carry one. |
| **Cross-validation** | Map khasra vs text khasra | Two independent readers checking each other, not one model asserting |
| **Co-owner capture** | The full claimant list | 162 owner rows across 20 documents had one slot to go in |
| **Prose recovery** | Dates and places from mutation orders | The data was always there; nothing was reading sentences |
| **Real authentication** | Supabase token verified server-side | The API was previously open to anyone |
| **Spatial database** | PostGIS: geodesic area, overlapping claims | Answers "do two documents claim the same ground?" |
| **Self-generated training data** | Sarvam labels a corpus for LayoutXLM | The system produces its own ground truth instead of waiting for annotators |
| **Satellite basemap** | Parcels drawn over Esri imagery | A traced boundary becomes checkable against the actual fields |
| **3D ULPIN** | Every floor, basement and air-rights envelope gets a derived identifier | PS 26011. A 2D register cannot name the third floor at all |
| **Vertical conflict check** | Two owners sold the same cubic metres | Needs footprint AND height overlap - the pair of them is what makes it real |

---

## SLIDE 4 — Measured, before and after

On the 20 real documents. This is the slide that earns trust.

| | Before | After |
| --- | --- | --- |
| **Records blocked** | **18 / 20** | **1 / 20** |
| Dates extracted | 7 / 20 | **20 / 20** |
| Village | 0 / 20 | **10 / 20** |
| District (verified) | 2 / 20 | **5 / 20** |
| Owner name | `नि.ग्रााम` | correct name |
| Co-owners stored | 0 | **15** on the test parcel |
| Parcel map read | never | **20 / 20** |
| Map confirms khasra | — | **12 confirmed, 4 corroborated** |
| Geotag (Jaunpur) | 300 km | **40 km** |

The one document still blocked is a **portal error page** that carries no
parcel — blocking it is correct.

---

## SLIDE 5 — The strongest single demo

**Two documents, one piece of land.**

PostGIS `find_overlapping_claims` reports documents #101 and #102 claiming
**5,506 m² of the same ground**. No per-document validator can answer that
question — it needs every parcel in one spatial index.

Second-strongest: open a record and show the map panel. *"The document's own
map says khasra 89, and so does its text. Its neighbours are 79, 81, 83, 85,
90, 92."*

---

## SLIDE 6 — Three decisions to defend

**Rules where documents are templated, models where they are not.**
Rules reach 0.94–0.95 on khasra, khata and area — the fields that identify
the parcel. The LLM is used for the one thing rules cannot express: a list
of 16 co-owners in a schema holding one. **Every model output must appear
verbatim in the source or it is dropped.**

**Refusing beats guessing.** No OCR engine → manual queue, never an empty
success. A projected CRS → refused by name. A village absent from the
directory → kept as unverified, never swapped for a similar-sounding one.
*Measured:* fuzzy-matching villages turned `अमारी` into `Amara`, a different
village — so fuzzy matching is now districts only.

**We measured and rejected three models.** SAM: 4/6 plots, 861 vertices
against tracing's 4–11, 64.5 s against 0.09 s. MuRIL: 5/8 where chance is
4/8. LayoutLMv3: its tokenizer shreds `भूमि का विवरण` into **61 byte
fragments** of raw UTF-8, and its Devanagari predictions were incoherent.

That last one has a follow-up worth saying out loud, because it is the
obvious next question: **LayoutXLM** — the multilingual sibling — tokenises
the same text into **13 real words**, so it is the right base for this
problem. It is not in the system yet: `layoutxlm-base` ships with no task
head, and its licence is research-only.

What changed is the reason it was blocked. It needed token-level labels
with bounding boxes for Indian land records, which nobody has — and we now
generate them rather than wait for annotators. See the next slide.

---

## SLIDE 7 — The system generates its own training data

Two mechanisms, and neither needs an annotator.

**① Every officer correction teaches the extractor.** Three artefacts are
derived from the corrections table: a character-confusion map, a value-alias
table, and per-field confidence calibration. Measured: `'4' → '1'` learned
from 6 corrections; `Naharpur → Narharpur` from 4, at 100% agreement.

Nothing is stored as opaque learned state — all three are rebuilt on demand
from the **audit-backed** corrections, so every adjustment traces to the
corrections that justify it. In a government context that is the difference
between "the model learned it" and a list of named corrections.

**② Sarvam labels a corpus for LayoutXLM.** This one does NOT come from
officer corrections - it reuses the extraction the merge already performed,
offline. The expensive cloud model teaches a small local one:

```
  Sarvam (grounded truth)  +  PDF word boxes  →  BIO tags with geometry
       90 s / document              exact          a training example
```

Measured across the whole real corpus: **176 labelled spans over 24 page
records from 19 documents**, generated in one unattended run. Every document
put through the pipeline becomes a training example, automatically.

Four documents were skipped and each skip is explicable rather than silent:
two where the teacher returned nothing, one portal error page carrying no
parcel, and one scanned PDF with no word boxes.

**Why that shape is the right one.** Sarvam is accurate and costs 90 seconds
a document over a network, with the record's text leaving the machine.
LayoutXLM would be ~125M parameters running on CPU in milliseconds, fully
local. Using the slow one once, offline, to teach the fast one is how you
get both.

Stated honestly on the slide: **Sarvam is the teacher, so any systematic
mistake it makes is what the student learns.** The labels are meant to be
spot-checked before training, and the tool says so in its own output.

---

## SLIDE 8 — Indian technology stack

| Layer | Choice | Why |
| --- | --- | --- |
| Language | **Bhashini** (MeitY) | Devanagari → LGD spelling. District resolution 0 → 43/43. |
| Reasoning | **Sarvam-105b** | Indic-first. 38/38 values grounded, 16/16 co-owners correct. |
| Geospatial | **PostGIS** (called) · **QGIS/GDAL** (used to prepare a map, not invoked) | Open source; what government GIS actually runs on |
| Basemap | **Esri imagery**, off by default | OpenStreetMap does not draw these villages at all |

**Data sovereignty:** Indian citizens' land records are processed by Indian
AI infrastructure. It answers the question a judge will ask — *"are you
sending land ownership records to a US provider?"* — before it is asked.

**Say the exception yourself.** The satellite basemap is Esri's, and it is
the one non-Indian service here. The distinction that makes it acceptable is
worth stating rather than hoping nobody notices: a basemap request asks for
*a tile of the earth at these coordinates*. No owner name, no khasra number
and no document text is in it. Record data goes to Bhashini and Sarvam; Esri
is sent a map square. It is also off by default, and Bhuvan can replace it
by changing one URL.

---

## SLIDE 8b — The third dimension (PS 26011)

**The question a 2D register cannot answer.** A parcel record says who owns
the ground. It cannot say who owns the third floor, the parking two levels
down, or the air above the road — and in a vertical city those are most of
the disputes.

```
UP091223700412-F03-012
└────────────┘ └─┘ └─┘
 parent ULPIN   |   unit on that level
 14 chars,      |
 DILRMP 3.0     B.. basement · G00 ground · F.. floor
                A.. air rights · S.. subsurface utility
```

Derived, deterministic, reversible, collision-free within its parcel. No
allocator, no counter, no clock — the same unit yields the same identifier in
a tehsil office with no network as on the state server.

**Say this before a judge asks.** DILRMP 3.0 specifies the 14-character parcel
ULPIN and does **not** specify a vertical extension. The parent is official;
the suffix is our proposal, hyphen-separated so the official part stays
extractable. The code says so in `describe()`, and the slide should too.

**The detail that shows the detector is real.** A conflict needs BOTH a shared
footprint and a shared height range. Either alone is an ordinary building: a
flat's ceiling *is* the next flat's floor, so `top == base` on every storey ever
built, and flats on one landing share a height range by definition. Most of the
35 tests assert **negatives** — because a detector that flags real buildings is
worse than none, since a reviewer learns to dismiss it.

**And the answer worth defending.** Three flats on one floor, each inheriting
the whole parcel footprint because the document has no floor plan, are neither
a conflict nor cleared:

| Call it | And you have |
| --- | --- |
| an overlap | flagged every block of flats in India |
| clean | asserted a separation nobody verified |
| `LEVEL_NOT_PARTITIONED` | stated the gap, once per level, severity *info* |

We take the third. The distinction cannot come from the geometry — two flats on
an undivided level are geometrically identical to two owners sold the same flat
— so it is a declared fact stored with the volume, not an inference.

**Honest about the geometry.** No LiDAR, no drone imagery, no floor plans, so
volumes are DECLARED from the footprint plus a nominal storey height and every
one carries `surveyed=False` in its own serialised form. The caveat travels
with the data rather than living in a document nobody opens.

---

## SLIDE 9 — Honest limits

Put these on a slide. Judges trust a team that names them first.

- **No parcel-grade coordinates yet.** These maps carry no world file and no
  control points. We produce shape, area in pixels and neighbours — not
  latitude and longitude. The state's cadastral shapefile closes this, and
  the importer is already built.
- **Villages reach district level.** OpenStreetMap has no entry for Amari,
  Bikapur or Narharpur. A gazetteer carrying Indian villages (Bhuvan) would.
- **The LLM call takes ~90 s.** It is a reasoning model. It belongs in a
  background job, not a blocking upload.
- **Devanagari handwriting** is detected and routed to a human, not
  transcribed. TrOCR's checkpoints are English.
- **ArcGIS geocoding is untested.** The basemap needed no key and works.
  The location service does: the token supplied returned 498 Invalid Token
  from ArcGIS's own `portals/self` check, so whether it reaches Indian
  villages where OpenStreetMap does not is still an open question, not a
  claim in either direction.
- **Vertical geometry is declared, not surveyed.** Floor heights come from a
  nominal storey height the operator can change, not from measurement, and
  floors are not subdivided into units because no floor plan exists. The
  system reports that gap per level instead of guessing past it.
- **Floor counts are entered, not extracted.** None of the 17 fields is a floor
  or storey count, because no document in the corpus prints one. The operator
  declares it; inferring a building's height from a plot area would be
  inventing the building.
- **Vertical parcels are not in PostGIS yet.** Footprints are stored as JSON,
  so overlap uses a bounding-box candidate test rather than an indexed
  `ST_Relate`. Correct for the volumes in one parcel; it would not scale to a
  city.
- **LayoutXLM is not trained yet.** The dataset is generated and the boxes
  are exact, but no fine-tune has run — there is no CUDA on the development
  machine, so it goes to Kaggle. Its licence is research-only, which a
  department deployment would have to resolve.

---

## Figures, if asked

| | |
| --- | --- |
| Backend modules | 31 |
| Tests | 779 (802 with a live PostGIS) |
| Validation rules | 68 |
| Fields / label aliases | 17 / 226 |
| Document types | 6 |
| Upload formats | 20 |
| Districts in the directory | 108 |
| Indic scripts | 14 |
| Basemaps offered | 4, incl. "None (offline)" as the default |
| LayoutXLM training spans | 176, over 24 page records |

---

## What NOT to claim

- Not "fully offline" — say **"degrades rather than fails."** Full accuracy
  with a network, reduced *and labelled* accuracy without one.
- Not "blockchain" — say **tamper-evident**. Hash-chained, no consensus.
- Not "99.3% accurate" without saying **on the digital corpus**. On real
  Bhu-Naksha documents the honest numbers are in Slide 4.
- Not LayoutLMv3, **LayoutXLM**, Donut or GeoPandas — none are used. Only
  LayoutXLM's TOKENIZER was run, to measure the comparison above. If asked
  "do you use LayoutXLM?", the answer is no, and the reason is worth giving.
- Not "we use ArcGIS." We use **Esri's public basemap tiles**, which need no
  account. The ArcGIS geocoding service has not been tested, because the key
  supplied did not authenticate.
- Not "the geospatial stack is QGIS, GDAL and PostGIS." The backend calls
  **PostGIS** and nothing else: it contains no reference to gdal, ogr2ogr,
  qgis or osgeo. QGIS is how a PERSON produced the world file that
  georeferences the demo sheet - a preparation step, not a component. Say
  "prepared in QGIS", never "powered by".
- Not "3D ULPIN is a government standard." DILRMP 3.0 defines the 14-character
  **parcel** ULPIN. The vertical suffix is **our proposal**, and the code says
  so where it surfaces.
- Not "we model buildings in 3D." We register **volumes**: a footprint and a
  height range per unit. There is no mesh, no BIM and no survey behind them -
  they are declared, and flagged as declared.
- Not "we use a model to trace parcels." Tracing is **OpenCV contour
  detection**, and georeferencing is an affine least-squares fit in pure
  Python. The two neural options are both off: BoundaryNet needs torch and
  is excluded from the deployment image, and SAM was measured worse - 4 of 6
  plots, 861 vertices against tracing's 4-11, 64.5 s against 0.09 s.
