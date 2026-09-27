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
python3 run.py --samples    # regenerate the 14 sample documents
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
| scikit-learn | ML fact-check against the external registry (§7a) | Fact-check step reports `FACT_CHECK_UNAVAILABLE` and is skipped |
| spaCy + en_core_web_sm | ML NER cross-check for person/date fields (§5) | Cross-check step reports `NER_UNAVAILABLE` and is skipped |
| paddleocr[doc-parser] + paddlepaddle | Table/layout structure recognition (§12d) | Reports `TABLE_STRUCTURE_UNAVAILABLE` and is skipped. **Unlike every other row above: needs Python 3.8-3.12 (PaddlePaddle doesn't support this project's own 3.14 interpreter), needs a CPU-inference workaround for a known PaddlePaddle 3.3.0+ crash, and measured ~20.7 minutes per page on CPU with no GPU - not usable for a live demo even where it does run.** |
| None (stdlib `urllib` only) - needs `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`/`NVIDIA_API_KEY` **and** `LLM_FIELD_EXTRACTION_CONSENT=1` | LLM-assisted field suggestions for fields the rule-based extractor couldn't read (§12e) | Off, silently, by design - see §12e. **The only row in this table where "off" is not a missing-dependency problem: this is the sole optional capability that sends document content to a third party, so an API key alone is deliberately not sufficient to turn it on.** |

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

**The project violated its own rule here once, and the fix is worth
recording.** `cv2.imwrite` reports failure by returning `False`, not by
raising. That return value was unchecked, so when this project's own
checkout sat 209 characters deep and the preprocessed-image path crossed
Windows' 260-character `MAX_PATH` limit, the write silently failed, the
path that was never written was handed to Tesseract anyway, and **every
scanned image extracted zero text while still reporting
`engine=tesseract`** - a successful-looking OCR run that quietly found
nothing, which is exactly the failure this section exists to forbid. PDFs
with a text layer were unaffected, which is why it hid for so long: the
sample corpus is mostly PDFs and they all worked.

It was found by running a document end to end and asking why the LLM stage
had been handed 0 characters of text - not by the unit suite, which passed
throughout, because the suite's temp directories are short. `ocr_engine.py`
now checks the write, relocates the file somewhere openable when the
natural path is too long (a `\\?\` extended-length prefix would fix the
OpenCV write but leave the external Tesseract process unable to read it),
and degrades to OCR-ing the unprocessed original with a stated warning
rather than pretending. `tests/test_ocr_engine.py::LongPathTests` covers
it. This is the same `MAX_PATH` class of bug that once aborted `api_seed`'s
entire batch, in a second place - on Windows it is worth assuming there is
a third.

**Two later additions follow the same rule.** The Bhashini script bridge
(S4a) and the Indic NER cross-check (S4b) are both optional, both off when
unconfigured, and both say so in `run.py --check` rather than appearing to
work. Neither can turn a missing capability into a confident answer: without
the bridge a Devanagari district simply fails to match the Latin master and
is reported unknown, exactly as before it existed; without the Indic model a
Devanagari owner name gets no NER verdict at all. The failure mode they are
built to avoid is the opposite one - `en_core_web_sm` run on Hindi does not
return nothing, it returns **wrong** answers (see S4b), and a wrong
corroboration is worse than an absent one.

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
                    |  rule families, LGD master cross-   |
                    |  check, duplicate detection,        |
                    |  fact_checker.py ML registry check, |
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
| `backend/field_extractor.py` | Field classification, confidence scoring, script detection |
| `backend/ner_extractor.py` | ML (spaCy NER) cross-check for person/date fields |
| `backend/validator.py` | Business rules, master-data cross-check, trust score |
| `backend/fact_checker.py` | ML (TF-IDF + cosine similarity) fact-check against an external registry |
| `backend/cadastral.py` | Cadastral map vectorization and georeferencing (S12a) |
| `backend/document_authenticity.py` | Signature/seal/stamp-paper presence detection (S12b) |
| `backend/anomaly_detector.py` | ML (IsolationForest) one-class anomaly baseline (S12c) |
| `backend/table_structure.py` | Table/layout structure recognition via PaddleOCR PP-StructureV3, optional (S12d) |
| `backend/llm_extractor.py` | LLM-assisted field suggestions for low-confidence/missing fields, optional and off by default (S12e) |
| `backend/db.py` | SQLite schema, queries, audit trail, statistics |
| `backend/learning.py` | Mining correction signals into an improving model |
| `backend/server.py` | Stdlib HTTP API, RBAC, static file serving |
| `backend/data/admin_master.json` | Real LGD state / district / tehsil / village master data (see "Honest limitations") |
| `backend/data/registry_master.json` | Bundled parcel-level registry extract for fact-checking |
| `frontend/` | Ingest, queue, verification workspace, dashboard, learning, audit |
| `tools/make_samples.py` | Generates the sample corpus with planted edge cases |

### S4a. The script bridge: Devanagari names against a Latin master

`ocr_engine.py` reads place names in fourteen Indic scripts. The
administrative master they are validated against holds 65,869 Latin
characters and **zero Devanagari**. The consequence was measurable and total:

```
district_exists("Lucknow")  -> "Uttar Pradesh"
district_exists("लखनऊ")     -> None            # the same district
```

Across all 43 districts in the bundled LGD extract, **0 resolved from
Devanagari**. This is not an OCR failure and no amount of better reading
fixes it; the name has to be rendered into the script the reference data
uses.

A hand-written character mapping was tried first and managed 5 of 8 on a
sample of hard cases. It fails on **schwa deletion**: जयपुर is written
`jayapura` and said `Jaipur`, and knowing which implicit vowels to drop is a
genuine problem in Indic text processing. Bhashini's IndicXlit learned it
from data and returns `jaipur`, taking the extract to **31 of 43**.

The remaining twelve are anglicised or orthographic - लखनऊ transliterates to
`lakhanau`, ठाणे to `thhaane` - and **no transliteration model will ever
produce Lucknow or Thane**, because the relationship is historical rather
than phonetic. Those live in `backend/data/place_exonyms.json`, built from
the measured failures with the model's own output recorded beside each entry,
and consulted before the network is touched. Result: **43 of 43**.

Three properties matter more than the number:

* **The master decides, not the model.** `bhashini.resolve_to_reference()`
  takes the authority as a callable, the way `learning.apply_model()` takes
  its corroborator. The client only proposes candidates.
* **Matching is exact after normalisation, never fuzzy.** A fuzzy match here
  would silently *rename* a district rather than fail to find one -
  `Lucknowe` resolves to nothing, deliberately.
* **Values are stored in the master's spelling.** A record reads
  `Uttar Pradesh`, not the model's `uttar pradesh`.

Owner and father names are **not** bridged: nothing validates a person's name
against a Latin list, so transliterating would discard the form printed on
the page for no gain. A village the master does not know keeps exactly what
the OCR read.

TLS note: `dhruva-api.bhashini.gov.in` chains to the Indian emSign root,
present in `certifi` but not in every trust store, so the module builds its
context from certifi and **never** disables verification.

### S4b. Indic NER, and why not MuRIL or IndicBERT

The NER cross-check ran `en_core_web_sm`, an **English** model. On a Hindi
khatauni line it did not degrade quietly, it degraded wrongly:

| Input | Result |
| --- | --- |
| Latin | both people found correctly |
| Devanagari | owner सुनीता देवी missed entirely, देवी tagged **DATE** |

So the check was not merely unavailable on Devanagari - it could manufacture
a false mismatch, which is why `cross_check()` used to skip non-Latin script
outright. A multilingual model closes the gap: measured over three real
land-record lines it recovered **6 of 6** person names and additionally
tagged नरहरपुर / सदर / लखनऊ as locations. spaCy is **kept** for Latin, being
a few tens of MB against the multilingual model's 2.5 GB.

**The obvious names do not work.** `google/muril-base-cased` declares itself
`BertForMaskedLM` with an empty `id2label`, and IndicBERTv2 ships MLM-only -
they predict masked words and have **no entity head at all**. The model that
would be ideal is `ai4bharat/IndicNER`, MuRIL already fine-tuned for this
exact task, but it is a **gated repo** returning 401 without an authenticated
account that has accepted its terms. Set `INDIC_NER_MODEL=ai4bharat/IndicNER`
once that access exists and the module uses it with no code change.

One asymmetry is deliberate. The English path feeds the model a *templated
value*, so it can only ask "is this a person name"; the multilingual model
handles whole lines, so the Indic path is given the **source line** and asks
the stronger question: a line reading सुनीता देवी against an extractor that
produced रीता शर्मा now raises `NER_MISMATCH` instead of waving both through
as plausible names.

**Deployment cost, measured, not estimated:**

| | PyTorch | ONNX int8 |
| --- | --- | --- |
| Model on disk | 1,110 MB | 278 MB |
| Resident memory | 1,195 MB | 759 MB |
| Person recall | 6/6 | 6/6 |
| Fits a 512 MB instance | no | no |

The blocker is XLM-R's 250,002-token vocabulary: that embedding table is
~768 MB in float32, and dynamic quantisation compresses matmul weights but
leaves embeddings alone. Hence this path is **local-only** and absent from
`requirements.txt`; the hosted demo reports it off.

## 5. The seventeen fields

`khasra_number`, `khata_number`, `survey_number`, `ulpin`, `owner_name`,
`father_name`, `share`, `area`, `land_classification`, `village`, `tehsil`,
`district`, `state`, `mutation_number`, `mutation_date`,
`registration_number`, `registration_date`.

Six are required: khasra number, khata number, owner name, area, village and
district. A record missing any of them cannot be approved.

### Script-aware OCR

Tesseract does not choose a language model for itself - it reads with
whatever pack it is handed. This project hardcoded `hin+eng`, which silently
capped the whole system at **one Indic script**: a Tamil or Telugu record
read with a Devanagari model does not fail loudly, it maps unfamiliar glyphs
onto the nearest Devanagari shapes and returns confident nonsense that then
poisons every downstream stage. The quality gate cannot catch it either -
the *image* is fine, so legibility scores high; it is the model that is
wrong.

Two further OCR decisions are made per page rather than fixed, and both were
forced by measurement on a real document.

**Page segmentation is chosen on evidence.** `--psm 6` ("assume a single
uniform block of text") suits a one-block register page and measured 1.1
points of field recall better than `--psm 3` across the scan corpus. But it
performs no layout analysis, and `assess_and_preprocess` strips table ruling
lines - so on a property paper carrying a printed form, handwritten patwari
entries and a sketch map, psm 6 lost its only structural cue, welded the
whole form into one block and read **1 of 8** printed fields where psm 3 read
**8 of 8**. `_ocr_page_best_layout()` now runs both and keeps psm 6 unless
psm 3 finds at least 25% more confident text. The margin is load-bearing:
the multi-region page gave psm 3 a 60% lead while every single-block scan
stayed within 6% either way, so without the bar noise alone would flip the
mode on half the corpus.

**Latin digits are re-read with the English pack.** Picking one language per
page is right for prose and wrong for numerals. On a real Bhu-Naksha plot
report `hin+eng` won script selection by a 13.54 confidence margin -
correctly, it is the only pack that reads the Devanagari - and then misread
every number on the page: khata `00100` became `0000`, plot `184` became
`84`, area `1.6350` became `.6350`, scale `1:1914` became `7:94`. The same
crops under plain `eng` came back exactly right, but `eng` renders
`खसरा नंबर` as `GERI AG`, so neither pack alone is sufficient - Indian land
records are bi-script *per line*. `_repair_digits()` keeps the Indic pass for
the prose and consults a second `eng` pass for the digit runs alone. Matching
the two readings is done on **geometry** (vertical overlap, horizontal IoU,
equal digit-run count), never on text similarity: across scripts the
non-digit remainder never matches, so an earlier text-similarity guard
blocked precisely the repairs it existed to enable. Corpus recall
91.3% -> 91.9%, precision 93.5% -> 94.1%, and khasra/khata on the real report
both became correct.

`ocr_engine.select_languages()` picks the pack per document, and the way
it does so was settled by measurement rather than by the obvious guess:

- **Tesseract's own OSD script detector was tried first and does not work
  here.** On four real pages it reported "Latin" every single time - even on
  a clean, Devanagari-dominant Hindi PDF - and once reported "Japanese,
  rotated 180°". The cause is structural, not a tuning problem: Indian land
  records are bilingual, OSD picks one dominant script for the whole page,
  and the English half wins.
- **What works is trialling the candidate packs and keeping the one
  Tesseract is most confident about.** Verified across four scripts: Tamil
  chose `tam+eng` by a 24.5-point margin, Telugu chose `tel+eng` by 17.0,
  and both Devanagari scans chose `hin+eng`. The winner's output also lands
  in the expected Unicode block while every loser emits Latin noise, which
  is an independent corroboration of the same decision.
- **A "try the default first, trial only if it scores badly" shortcut was
  built and then deleted.** Measured on the preprocessed images selection
  would actually see, right-script pages scored 61.3-71.1 and wrong-script
  pages 58.6-65.7 - overlapping ranges, so no threshold separates them.
  Selection therefore runs on the RAW page, before restoration compresses
  exactly the differences it depends on.
- **One candidate per script, not per language.** Confidence is a strong
  signal for "is this even the right script" and a weak one for choosing
  between two models of the same script: `mar+eng` outscored `hin+eng` on
  all three Devanagari samples, including two that are Hindi documents. A
  more confident model is not necessarily a more accurate one, so Devanagari
  is represented once and Marathi records keep being read by it - correctly,
  because the script is what the recogniser cares about.

Cost is controlled by trialling on a half-size copy (verified to pick the
same winner as full resolution; 0.35x was too far and broke a case) and by
deciding once per document rather than per page. Documents with a PDF text
layer never reach this path at all. The chosen pack and its margin are
recorded in the document's warnings, so a reviewer can see which model read
their record and how clear-cut the choice was.

### Multilingual label recognition

Field labels are matched in Hindi/Marathi (Devanagari), Bengali, Gurmukhi
(Punjabi), Gujarati, Tamil, Telugu and Kannada, alongside English -
`field_extractor.detect_script()` identifies which Unicode script a source
line uses (a deterministic character-block count, not a language model - no
GPU, no training data, auditable like every other rule in this system) and
is recorded per field so a reviewer can see which script a value was
actually read from.

Label coverage is deliberately uneven across fields, on purpose. Universal
concepts - owner, father/husband name, village, tehsil-or-equivalent,
district, state, area, share, land classification, survey number - get real
translations, including region-specific revenue terms where they are
genuinely used (e.g. Karnataka/Maharashtra's "guntha", the South's "cent",
Karnataka's/Tamil Nadu's "survey number", West Bengal's "dag number" as a
khasra equivalent). `khata_number`, `ulpin`, and the mutation/registration
fields are **not** force-translated into the newer languages: their closest
analogues (a West Bengal khatian, Karnataka's Bhoomi mutation record) are
not the same legal instrument as a khata, and guessing wrong in this domain
is worse than not extracting the field at all.

### ML-based NER cross-check

`backend/ner_extractor.py` adds a genuinely different, ML-based signal
alongside the rule-based extractor above, not instead of it: a
general-purpose named-entity recognition model (spaCy's `en_core_web_sm`)
reads each field's own source line independently, and its PERSON/DATE spans
are cross-checked against `owner_name`, `father_name`, `mutation_date` and
`registration_date`. Agreement raises `NER_CONFIRMED` (info); the model
finding a different entity on the same line raises `NER_MISMATCH` (warning);
the model finding nothing raises `NER_UNCONFIRMED` (info - not necessarily
wrong, just uncorroborated). An optional dependency: missing spaCy/model
degrades to an explicit `NER_UNAVAILABLE` info issue, never a silent no-op.

Scope is deliberately narrow, and the narrowing is load-bearing, not
laziness. A generic NER model has no notion of "khasra number" - there is no
honest way to make it corroborate a domain identifier it was never trained
to recognise, so the numeric ID fields get no NER claim at all, ever. Latin
script only: `en_core_web_sm` is an English model, and running it on
Devanagari/Bengali/Tamil/etc. would not fail loudly, it would silently
produce meaningless spans - `field_extractor.detect_script()`'s per-field
script tag gates this instead of guessing.

Two limitations were found by testing against this project's own sample
data, not assumed, and are handled rather than hidden:

- A bare "Label : Value" line gives the small model too little sentence
  structure to recognise a name at all - even the bare name alone finds
  nothing. The value is wrapped as "Mr. {value}." for the NER call only
  (never stored or shown) to restore that context; this does not guarantee
  a match, it measurably improves the chance of a real classification.
- A NER-found DATE span stays in the source's raw format ("27/02/2031")
  while the rule-based `value` is already ISO. Comparing those strings
  directly reported every correct match as a mismatch purely from
  formatting - both sides are now parsed to ISO with the same parser
  (`field_extractor.parse_date`) before comparing.

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
| `REQUIRED_MISSING` | error | A mandatory field was not extracted. The parcel identifier is an "at least one of" group - `khasra_number` **or** `survey_number` - because South Indian states issue no khasra (see S13) |
| `DUPLICATE_CONFLICT` | error | Same parcel already recorded with a different owner |
| `AREA_UNIT_MISSING` | error | Area with no unit, so it cannot be standardised |
| `AREA_REGIONAL_UNIT` | warning | Bigha/biswa recorded; conversion is region-dependent |
| `DATE_FUTURE` | error | A date in the future |
| `DATE_ORDER` | error | Registration after mutation |
| `DATE_TOO_OLD` | warning | Pre-independence date needing archival verification |
| `DISTRICT_UNKNOWN` | warning | District absent from the LGD master, with the closest match suggested |
| `TEHSIL_MISMATCH` | warning | Tehsil does not belong to the stated district |
| `VILLAGE_TEHSIL_MISMATCH` | warning | Village is known in this district under a *different* tehsil than stated |
| `OWNER_FATHER_SAME` | warning | Owner and father recorded as the same person (likely column bleed) |
| `SHARE_OVER_UNITY` | error | Ownership share greater than 1 (fraction or percentage) |
| `SHARE_INVALID` | error | Ownership share has a zero/negative denominator or percentage |
| `CLASS_UNMAPPED` | warning | Land classification outside the controlled vocabulary |
| `LOW_CONFIDENCE` | warning | Field extracted below the review threshold |
| `AREA_RANGE` (hard bound) | error | Area outside any plausible parcel size (not just unusual) |
| `MUTATION_DATE_MISSING` / `REGISTRATION_DATE_MISSING` | warning | A record number was extracted without its paired date |
| `MUTATION_NUMBER_MISSING` / `REGISTRATION_NUMBER_MISSING` | warning | A record date was extracted without its paired number |
| `PLACEHOLDER_VALUE` | error/warning | A field holds a placeholder ('N/A', '-', 'Unknown') instead of real data |

Rules produce a **trust score** out of 100, which drives routing:
auto-approve, needs review, or blocked. A record with any blocking error
cannot be approved through the API at all, whatever the reviewer clicks.

### 7a. External fact-checking (ML-based cross-registry verification)

`backend/fact_checker.py` adds the third leg the problem statement asks for
("cross-database verification") against an *external* authority, distinct
from the internal business rules above and from the exact-signature duplicate
check. In production the authority is a DILRMP/LGD parcel-level registry
reached over an API; no hackathon team has state credentials for that, so
this bundles a representative extract in the same shape
(`backend/data/registry_master.json`) - identical in spirit to how
`AdminMaster` stands in for the LGD administrative directory.

OCR-extracted identity fields (khasra, khata, village, district) rarely match
a registry's spelling character-for-character, so an exact-key lookup would
silently miss a large fraction of genuine hits. This module instead fits a
scikit-learn character-n-gram TF-IDF vectoriser over the registry's identity
strings and retrieves the nearest one by cosine similarity, so near-miss
spellings still resolve to the right parcel. Once a parcel is matched, owner
name, area and land classification are compared against the registry entry:

| Rule | Severity | What it catches |
| --- | --- | --- |
| `FACT_CHECK_VERIFIED` | info | Owner, area and classification agree with the matched registry record |
| `FACT_CHECK_OWNER_MISMATCH` | error | The registry lists a different owner for this parcel |
| `FACT_CHECK_AREA_MISMATCH` | warning / error | Extracted area differs from the registry by ≥15% (warning) or ≥50% (error) |
| `FACT_CHECK_CLASS_MISMATCH` | warning | Extracted land classification disagrees with the registry |
| `FACT_CHECK_NOT_FOUND` | info | No registry entry matched this parcel closely enough |
| `FACT_CHECK_UNAVAILABLE` | info | scikit-learn is not installed; the step was skipped, not silently passed |

Every match names the exact registry record and similarity score that
produced it, and every mismatch names the two conflicting values, so the
result stays fully explainable rather than an opaque "fraud score" -
consistent with the design intent for every other rule in this system.

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

### S8a. Tamper-evident audit trail

Every `audit_log` row stores the SHA-256 of its own content combined with the
previous row's hash. Editing an old entry, or deleting one, breaks the link
every later row depends on, and `verify_audit_chain()` reports the **first**
row that no longer agrees - because that row is where the history stops being
trustworthy, regardless of whether later rows were themselves touched.

```
GET /api/audit/verify
{"ok": true, "entries": 81, "verified": 81, "broken_at": null,
 "tip": "57d6d68f4e08bbe8259b40f44d78f69c..."}
```

Three implementation details are load-bearing:

* **Reading the tip and appending happen inside one lock.** The server is
  threaded, so two writers reading the same tip would fork the chain and make
  an honest log fail verification ever after.
* **Sealing never recomputes a row that already has a hash**, which would
  quietly repair exactly what verification exists to report.
* **Rows written before the chain existed are sealed on migration.** That
  establishes a baseline; it proves nothing about what happened to them
  beforehand, and the docstring says so.

The endpoint is open to **every** signed-in role, auditor included. An
integrity check only an administrator may run is worth little, because the
administrator is who an auditor is checking up on.

**This is tamper-EVIDENT, not tamper-proof, and deliberately not a
blockchain.** Someone with write access to the file can recompute the chain
from their edit onward and it will verify clean. What the chain buys is that
tampering can no longer be *silent* - it must be deliberate and complete -
and a copy of the tip hash held anywhere outside the database (a nightly
export, a printout, a second office) makes even a complete recompute
detectable. There is no distributed consensus here and claiming otherwise
would be the one dishonest component in an otherwise honest system.

`tests/test_audit_chain.py` is written from the attacker's side: it performs
the edit a dishonest operator would actually attempt - rewriting an owner
name, deleting a correction, changing who acted, backdating an entry, forging
a row by hand - and asserts that verification catches it **and names the right
row**. One test matters more than the rest: a later honest append must not
repair an existing break, or an attacker could edit a row and wait for
ordinary traffic to bury it.

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

`tools/make_samples.py` generates 14 documents: 11 PDFs with a real text layer
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
| 11 cadastral-linked | Khasra `213/1`, village नरहरपुर - the same parcel S12a's bundled demo cadastral map already has, so approving this one is a live, working demonstration of the map's document-linking (not just wired, verified end to end: upload &rarr; auto-approve (trust 94.3) &rarr; the parcel lights up on the Cadastral Map tab) |
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
| GET | `/api/audit/verify` | Re-verify the hash chain; names the first broken row (S8a) |
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

### 12a. Cadastral map vectorization and georeferencing

`backend/cadastral.py` turns a raster parcel-boundary map into georeferenced
vector geometry, closing the "join parcels to cadastral geometry" claim
above with a working pipeline rather than just a compatible field (ULPIN).
Two independent, classical (non-ML) stages:

1. **Vectorization** - threshold the scan so boundary ink is the only dark
   content, find each enclosed parcel interior as an isolated connected
   region (`cv2.findContours`), then simplify its pixel-level outline with
   Douglas-Peucker polygon simplification (`cv2.approxPolyDP`) into a
   handful of real corner vertices. A best-effort OCR pass
   (`read_parcel_labels`, using the same Tesseract path as document
   ingestion) reads each parcel's printed khasra number from a small crop
   around its centroid; a missed label leaves `khasra_number: null` rather
   than blocking the polygon, the same honest-degradation pattern as the
   rest of this system.
2. **Georeferencing** - fit a 6-parameter affine transform from ground
   control points (pixel position <-> known real lon/lat) via
   `numpy.linalg.lstsq`, then apply it to every vectorized vertex. Three
   points give an exact fit; more give a least-squares fit whose per-point
   residuals are reported, so georeferencing quality is measurable rather
   than assumed. Output is a standard GeoJSON `FeatureCollection`.

**Honesty note, matching README S13's pattern below:** there is no real
cadastral map or real surveyed ground control point available to this
project. `tools/make_cadastral_map.py` generates a synthetic demo map
(`samples/cadastral/village_map_narharpur.png`) - continuing the
Narharpur/Lucknow village from the text-record demo samples - with
illustrative, clearly-labelled placeholder control points
(`control_points.json`). Every GeoJSON produced from that bundled data
carries an explicit `_disclaimer` field saying so. The pipeline itself is
real and runs identically on a genuine scanned map and genuine surveyed
GCPs once they exist; verified against the demo map's known ground truth
(`ground_truth.json`): 14/14 parcels vectorized, 14/14 khasra numbers read
correctly, georeferencing residuals near zero (`tests/test_cadastral.py`).

**Since verified against real data too** - two actual Bhu-Naksha single-plot
reports (Agra district, UP) - which surfaced two genuine findings, not just
confirmations:

- Real exports surround the map with a title and an owner/attribute table.
  Without excluding that, it gets picked up as false parcels and skews the
  area filters enough to wrongly reject real ones too. `detect_map_frame()`
  finds the map's own bordered panel (reliably the largest, most rectangular
  contour on the page - ~700x larger than the next candidate on the reports
  tested) and restricts detection to it, matching what `vectorize()`'s
  `crop_to_frame` parameter does by default.
- A real single-plot export only guarantees a closed boundary for plots
  fully interior to whatever extent was rendered; a neighbouring plot shown
  only for context has its far/outer edge left undrawn wherever it falls
  outside that extent, so it is not a closed shape. `vectorize()` correctly
  leaves those undetected rather than guessing at a boundary that was never
  drawn - confirmed on both real reports tested (the queried plot and every
  plot fully interior to the shown extent vectorized correctly; only the
  plots clipped by the extent's edge did not, exactly as expected).

Not yet fixed by this: `read_parcel_labels`'s OCR crop window was tuned
against the synthetic map's font size and label placement, and reads
garbage on the real reports' different scale - vectorization is solid on
real data, the label-reading step still needs recalibrating against it.

**GIS integration: a map view, not a new format.** The GeoJSON above was
already a standard GIS interchange format - any desktop GIS tool (QGIS,
ArcGIS) opens it natively with no conversion. What did not exist yet was
somewhere to *see* it without leaving the app: `GET /api/cadastral/parcels`
(`backend/server.py`) now serves the vectorized/georeferenced demo map as
JSON, cached after the first request since the input is fixed, and the new
**Cadastral Map** tab renders it with Leaflet - each parcel a clickable
polygon showing its khasra number and pixel area. Leaflet is vendored into
`frontend/vendor/leaflet/` (downloaded once, committed, served by the same
stdlib static-file handler as everything else) rather than loaded from a
CDN, preserving the "nothing to install, works with no internet" property
every other part of this front end already has - the one CDN script this
project does load (`supabase-js`, for an unrelated auth experiment) is not
a precedent this addition follows. No basemap tiles are shown, deliberately:
there is no tile-provider API key here, and fetching third-party tiles
would make a live network call a silent dependency of what is supposed to
be an offline-capable demo. The map shows parcel geometry and labels only -
exactly what georeferencing actually produces and what S12a's accuracy
claims above are about.

**Geo-tagging an uploaded document, honestly scoped.** A khatauni/khasra
text document carries no coordinates of its own - OCR and field extraction
cannot conjure geometry out of a text field, and this project will not
pretend otherwise. What it *can* do, and now does: once a document is
extracted, validated and routed, `_link_cadastral_documents()`
(`backend/server.py`) matches its `khasra_number` against the parcels
already produced by S12a's vectorization, and `GET /api/cadastral/parcels`
attaches the matching document's id, status, owner name and trust score to
that parcel - a real join against already-known geometry, not synthesised
geometry. This is recomputed on every request (unlike the cached parcel
geometry itself), so approving, correcting or rejecting a document changes
what the map shows on the next visit, live. On the Cadastral Map tab a
linked parcel is coloured by the linked document's status (the same
green/orange/red vocabulary as the verification queue) and its popup links
straight back into the verification workspace; an unmatched parcel stays
neutral grey. Sample 11 (`sample_11_cadastral_linked.pdf`, §10) exists
specifically to make this demonstrable out of the box: its khasra number
(`213/1`) and village (नरहरपुर) were deliberately chosen to match the
bundled demo cadastral map's own last parcel, so uploading it and watching
that parcel light up is a real, reproducible verification of the whole
loop - not just wiring that has never actually been run.

**Matching is scoped by village, not khasra number alone.** Khasra numbers
are only unique *within* a village - two villages routinely reuse the same
numbers - so matching on the number by itself would cheerfully tag a Bhopal
record onto a Lucknow parcel. `control_points.json` names the village its
map covers (with aliases, so `नरहरपुर` and `Narharpur` both match) and a
record is only linked when the village agrees. Verified: with khasra `213/1`
held constant, Narharpur matches in either script, while `बरखेडी` (Bhopal)
and `Shirur Kasar` (Pune) correctly do not. A record with no village is not
matched at all rather than guessed at.

**Geo-tagging validates, it does not only display.** A matched record gains
real coordinates (`GEO_TAGGED`, the parcel centroid) and - more usefully -
its stated area is cross-checked against the area its parcel actually
measures. `cadastral.polygon_area_m2()` converts a lon/lat ring to square
metres via an equirectangular projection taken at the parcel's own latitude
(the longitude scale shrinks ~20% between Kanyakumari and Ladakh, so a
single national constant would be wrong), and the result is compared with
`field_extractor.parse_area()`'s `sqm` for the record. Beyond 15% they
disagree materially - about 1,500 m² on a one-hectare plot, a real strip of
land - and `GEO_AREA_MISMATCH` sends the record to review.

This is deliberately phrased as a *disagreement between the register and the
map*, never as "the record is wrong": vectorising a scanned map and fitting
an affine transform both carry error, so the geometry is at least as likely
to be the inaccurate side. It is also non-blocking for the same reason.

The bundled demo shows this working end to end: `sample_11` states 0.8100
hectare, its parcel measures 0.9651 hectare, and the 16% gap moves the
document from `auto_approved` to `needs_review` - a record that previously
sailed through is now caught by a check no amount of text validation could
have performed. Every such message carries an explicit caveat that the
demo map's control points are illustrative rather than surveyed, so the
absolute numbers are not presented as real-world measurements.

**Real maps, not just the bundled demo.** The vectorizer was verified against
actual Bhu-Naksha exports early on, but the system could only ever use the one
synthetic map compiled into the source. It now holds as many village maps as an
office adds:

```
storage/cadastral/<village>/
    map.png                 the scanned cadastral map
    control_points.json     village name, aliases, district, and the GCPs
```

A map is just a directory - no migration, no database row - so one can be
dropped in by hand on a machine where the upload path does not work.
`POST /api/cadastral/maps` (multipart: `map` + `control_points`) adds one
through the API, admin-gated on the same right as model retraining, because a
map is reference data that every record in that village is then validated
against: a wrong one mis-tags a whole village rather than one document. The
upload refuses a control-points file that does not name its village, or that
carries fewer than 3 points (an affine fit needs three), rather than accepting
a map it cannot place. `GET /api/cadastral/maps` lists them and the Cadastral
Map tab grows a village selector once more than one exists.

**Why village scoping is load-bearing, demonstrated rather than argued.** A
second synthetic village (Barkhedi, Bhopal) was generated deliberately reusing
the demo village's khasra numbers, then uploaded. With khasra `213/1` held
constant:

| Record's village | Resolves to | Location | Measured area |
| --- | --- | --- | --- |
| नरहरपुर (Lucknow) | `demo` parcel 1 | 26.8516, 80.9497 | 9,651 m² |
| बरखेडी (Bhopal) | `barkhedi` parcel 7 | 23.2591, 77.4133 | 16,590 m² |
| `Barkhedi` (Latin alias) | `barkhedi` parcel 7 | same | same |
| किशनपुरा (no map) | nothing | - | - |

Two records quoting the same khasra number resolve to parcels ~600 km apart,
each in its own district. Without scoping they would resolve to the same
parcel, and a Bhopal holding would be geo-tagged onto Lucknow land - which is
precisely the kind of silent, confident error a land-record system must not
make.

The remaining honest limit: a document whose village has no cadastral map
loaded has nothing to link to, and is correctly reported as unmatched rather
than silently ignored or guessed at. Coverage is now an operational question
(add the map) rather than a code limitation.

### 12b. Document integrity signals: signature, seal, stamp paper

`backend/document_authenticity.py` detects whether a signature, a
department/notary seal, and India Non-Judicial stamp-paper indicators
appear to be present on a page.

**Scope, stated as plainly as possible: this detects presence, not
authenticity.** It cannot tell a genuine signature from a forged one, or a
real government/notary seal from a fake one - that needs a reference
signature database or a live government stamp/notary registry, neither of
which exists here (the same honesty gap as the DILRMP connector and the
cadastral module's ground control points). What it does do: flag whether an
expected mark is there, so a human reviewer does not have to be the one who
first notices it is missing. `SEAL_DETECTED`/`SIGNATURE_DETECTED` mean a
stamp-shaped or signature-shaped mark was found - never whose it is.

Three classical (non-ML) techniques, each chosen because a simpler one
measurably failed against real reference material (real India Non-Judicial
stamp paper and a real notarised affidavit with a genuine notary seal,
supplied for this purpose - not bundled into this repository, since it
carries a named individual's personal and financial details):

| Signal | Approach | What failed first |
| --- | --- | --- |
| Stamp paper | Highest-saturation horizontal band in the top ~45% of the page (+ an OCR keyword check when it fires) | A single average over the top region: the real banner did not start at row 0, it began ~15% down the page, so averaging from the top edge diluted it below any reasonable threshold |
| Seal | HSV-saturation ink, merged by morphological closing, kept only if its bounding box is near-square | `cv2.HoughCircles` found 14 "circles" on a real page and only 1 was the actual seal - it hallucinates circles from watermark texture and coloured text rows; bounding-box aspect ratio (1.00 for the real seal vs. 2.4-9.2 for every false candidate) is what actually discriminates |
| Signature | Ink-density ratio within a bounded region (default: bottom third of the page) between a "blank" floor and a "dense paragraph" ceiling | - |

Wired into `server.py`'s ingestion pipeline (not into `validator.py`, since
it reads the rendered page image, not extracted field values):
`STAMP_PAPER_DETECTED`/`SEAL_DETECTED`/`SIGNATURE_DETECTED` are informational.
`SEAL_MISSING` is **mandatory and blocking** - a department/management seal
is required on every land record document in this domain, so a missing one
is an `error` (the same severity as a missing required field) and sets the
record to `blocked`, with no registration/mutation gate. `SIGNATURE_MISSING`
stays a `warning`, raised only when the record already has a registration or
mutation number - i.e. it claims to be a registered instrument - since a
bare text extract is not held to that standard. `AUTHENTICITY_CHECK_UNAVAILABLE`
reports honestly when OpenCV/numpy are absent, the same degradation pattern
as everywhere else in this project.

Because `SEAL_MISSING` blocks, it has to be re-evaluated everywhere a
document's validation result is recomputed, not just at first ingestion -
`revalidate()` (run after every field correction) and the `POST
.../approve` gate both call the same authenticity/anomaly checks now.
Missing that would have meant a document that is genuinely missing its
seal could get silently unblocked by correcting an unrelated field, or
approved outright through a gate that never looked at the image at all -
found and fixed while wiring the mandatory rule in, not assumed safe.

### 12c. Anomaly baseline (ML, one-class)

`backend/anomaly_detector.py` learns what a "normal" document looks like
from documents this office has already approved, and flags a new one as
`ANOMALY_DETECTED` when it is a statistical outlier against that baseline -
using scikit-learn's `IsolationForest` (already a bundled optional
dependency via `fact_checker.py`, so this adds nothing new to install).

**Why one-class, not a fake/genuine classifier**: training a classifier
needs labelled examples of both classes, and no labelled forged-land-record
dataset exists - building one would mean fabricating forged government
documents, which this project will not do. One-class anomaly detection
sidesteps that entirely: it only ever needs genuine, already-approved
documents, which accumulate naturally as the system is used. The trade-off
is the honesty rule stated everywhere else in this project: an anomalous
document is statistically unusual, not proven fake, and a normal-scoring
one is not proven authentic - see `document_authenticity.py`'s identical
distinction for seals and signatures.

Features are not computed freshly - every one already exists elsewhere in
the pipeline (validator trust score, field-extraction confidence and
completeness, OCR legibility, the seal/signature/stamp-paper presence
booleans from S12b). Training happens on demand from the admin's Learning
tab / `POST /api/learning/retrain`, which now also rebuilds this baseline
from every `approved` document, cold-start-protected honestly:
`ANOMALY_BASELINE_NOT_TRAINED` is reported explicitly below the minimum
sample size, not silently skipped.

**Two things this needed actual measurement to get right, not intuition**:

- **`MIN_TRAINING_SAMPLES` is 50, not a smaller "reasonable-sounding"
  number.** Measured directly: holding out an obviously extreme document
  (trust score 12 against a training range of 90-97) and scoring it across
  5 random seeds at each training size, 15 samples caught it 4/5 times, 20
  samples caught it only 2/5 - *worse*, non-monotonically - and 25+ caught
  it 5/5 every time. A subtler real anomaly needs more margin than an
  obvious one, hence 50.
- **`contamination` (default 0.1) is not a tuning knob to overlook.** It
  tells `IsolationForest` to assume that fraction of *any* scored set -
  including the training set itself - are outliers. Confirmed directly:
  two of thirty synthetic training documents scored as anomalous purely
  for sitting at the edge of the fixture's value range, not for anything
  resembling a real problem. An office should expect roughly 1-in-10 of its
  own approved documents to trip this if scored against their own baseline,
  by design - lower it if that is too aggressive for how this gets used.

### 12d. Table/layout structure recognition (PaddleOCR PP-StructureV3, optional)

`backend/table_structure.py` wraps PaddleOCR's PP-StructureV3 pipeline to
recognise page layout regions (table / text / image / seal / formula) and,
for detected tables, their structure as HTML - read off the *rendered page
image*, on top of and independent from the rule-based `field_extractor.py`.
It never writes into extracted field values and never affects the
validation decision (`decision`, `error_count`, `warning_count` are all
untouched); it only ever adds informational issues
(`TABLE_STRUCTURE_DETECTED` / `TABLE_STRUCTURE_UNAVAILABLE` /
`TABLE_STRUCTURE_FAILED`) a reviewer can read, the same tier as
`STAMP_PAPER_DETECTED` in S12b.

**This is a different kind of optional dependency from every other one in
this project, and is documented as such rather than presented the same
way.** PyMuPDF, OpenCV, pytesseract, scikit-learn and spaCy are all small,
fast, CPU-only installs. PP-StructureV3 pulls in the PaddlePaddle deep-
learning framework and downloads roughly half a dozen separate model
bundles on first use (layout detection, document/textline orientation, OCR
detection and recognition, table classification and cell detection,
formula recognition - all fetched the first time `analyze()` actually runs,
not at install time).

**Three real problems were found by actually running this against a real
bundled sample scan in a compatible environment, not assumed from
documentation:**

1. **Python 3.13+ is unsupported outright.** This project's own system
   interpreter is Python 3.14; PaddlePaddle does not build for it, so
   `pip install paddleocr` fails at the dependency-resolution stage with
   "no matching distribution found" - it does not install and then
   misbehave, it simply cannot install. `run.py --check` reports this
   explicitly. Verification below was done in a separate Python 3.11
   virtual environment (PaddlePaddle's supported range is 3.8-3.12).
2. **The default pipeline crashes on real inference on this exact
   PaddlePaddle build.** `PPStructureV3()` constructed fine and the model
   weights downloaded fine, but calling `.predict()` raised
   `NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
   [pir::ArrayAttribute<pir::DoubleAttribute>]` - a known regression in the
   CPU/oneDNN execution backend introduced in PaddlePaddle 3.3.0
   ([PaddlePaddle/Paddle#77340](https://github.com/PaddlePaddle/Paddle/issues/77340)),
   not a bug in this project's wrapper code. The documented workaround,
   `PPStructureV3(enable_mkldnn=False)`, does avoid the crash, and
   `table_structure.py` now passes it unconditionally (falling back to the
   plain constructor only if a future PaddleOCR version rejects the kwarg).
3. **Real CPU throughput is far worse than "slow but usable."** With the
   crash worked around, one real page from this project's own sample corpus
   took **~20.7 minutes** to process on this machine's CPU (no GPU
   available). That is a measured result, not a worst-case estimate. It
   rules this pipeline out for anything resembling a live demo or an
   interactive review workflow on CPU-only hardware; PaddleOCR's own
   documentation assumes GPU acceleration for PP-StructureV3 at a usable
   speed, which this project's target environment (a judge's or a revenue
   office's ordinary laptop) does not have.

Given all three, this feature is wired in correctly and does work - but
"works" here means "produces a correct result eventually, on a compatible
Python version, with the mkldnn workaround applied," not "usable in this
project's actual demo." It is included because reliable, honest
degradation was already the project's standard for every other optional
dependency, and this one deserved the same treatment rather than being left
half-wired because it turned out to be impractical.

Install (into a compatible Python, 3.8-3.12):

```bash
pip install "paddleocr[doc-parser]" paddlepaddle
```

The PP-StructureV3 result object's exact attribute surface is not something
this project controls and has changed across PaddleOCR releases, so every
access into it (`_markdown_text`, `_layout_label_counts`) is defensive -
dict-or-attribute, wrapped in `try/except` - and a shape this code does not
recognise degrades to "ran but found nothing to report", never a crash.
For the same reason the automated test suite (`tests/test_table_structure.py`)
exercises that defensive parsing with fixture objects rather than
constructing the real pipeline on every test run, which would make the
suite take minutes instead of seconds and require model weights and network
access just to run `python -m unittest discover`.

### 12e. LLM-assisted field suggestions (optional, off by default)

`backend/llm_extractor.py` asks an LLM (Anthropic, OpenAI, or NVIDIA's
hosted NIM catalog at `integrate.api.nvidia.com`, stdlib `urllib` only - no
SDK) to suggest a value for fields `field_extractor.py`
already reported as missing or below the 0.80 confidence threshold
`validator.rule_low_confidence` uses. This is the one deliberate exception
to this project's otherwise universal "everything ML-related runs on this
machine" rule (fact_checker's TF-IDF matching, ner_extractor's spaCy NER,
anomaly_detector's IsolationForest, table_structure's PaddleOCR - all
local). It exists because it was explicitly requested with that trade-off
understood, after the alternative (a local model, slow on this project's
GPU-less hardware in the same way S12d's PaddleOCR turned out to be) was
laid out first.

**Activation requires two separate things, not just an API key:**
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `NVIDIA_API_KEY` set, **and**
`LLM_FIELD_EXTRACTION_CONSENT=1` set. An API key can exist in an
environment for a reason that has nothing to do with this project (a
developer's own unrelated tooling, a shared CI environment) and must never
be silent permission to export land-record PII - owner names, khasra
numbers, addresses - to a third party. Both flags must be set, deliberately,
before a single byte of document text leaves the machine. `run.py --check`
reports the status honestly either way, but - unlike every other optional
dependency in this project - being "off" is never reported as a bare
"install this to enable" nudge, because what turning it on actually means
is materially different from installing a library.

**Every suggestion must be grounded in the document, and this is the part
that matters most.** On the first real end-to-end run against a degraded
Madhya Pradesh scan, the model returned a fluent, complete, entirely
**fabricated** land record - `Khasra No. 1234`, `Shri Ram Kumar`,
`District - Alwar`, `State - Rajasthan`, mutation number `1111` - while the
document's real values (`42/4`, `सुनीता बाई`, `Bhopal`, `Madhya Pradesh`)
sat in the very prompt it had been handed, and while the same model on the
same input had read them correctly minutes earlier. Telling a model not to
fabricate does not stop it fabricating, and in a land-record system a
confident invented khasra number is worse than no suggestion at all.

`llm_extractor.is_grounded()` closes this: extraction has the convenient
property that a genuine extracted value must, by definition, appear in the
source document, so any suggestion that cannot be found in the OCR text is
discarded no matter how plausible it reads. Digits are normalised across
all seven Indic scripts first (reusing `field_extractor.normalise_digits`),
so a model that helpfully rewrites `९०७३` as `9073` is not punished for it,
and whitespace is collapsed so OCR spacing quirks do not cause false
rejections. Measured against the real fabricated output above: all 5
invented values rejected, all 7 genuine values kept. The effect is to turn
the LLM from something whose word is taken into something that can only
point at text which demonstrably exists - a non-deterministic component
wrapped in a deterministic, auditable check, which is the only shape in
which it belongs anywhere near a legal record.

**What it does not do, on principle:** a suggestion is tagged
`source="llm:<model>"` and shown to a reviewer as an informational
`LLM_SUGGESTION` issue alongside the rule-based extraction - it never
overwrites the accepted field value the way a confirmed correction does,
and never affects `decision`/`error_count`/`warning_count`. LLM output
cannot be read and audited line by line the way a regex or a gazetteer
lookup can, which is exactly why this project has stayed rule-based
everywhere else; this module is a bounded exception to that stance, not a
reversal of it. It also runs only at ingestion (`process_document`), never
on `revalidate()` - the source text never changes when a human corrects a
field, so re-asking would resend the same document content and pay for it
again for zero new information (see `revalidate()`'s docstring).

**Anthropic and OpenAI are not verified against a live API call in this
environment** - no key for either exists in the sandbox this was built in,
so `tests/test_llm_extractor.py` exercises the activation gate, grounding,
JSON parsing, and suggestion-filtering logic with the provider call
monkeypatched for those two. The **NVIDIA path was verified end to end
through the real ingestion pipeline** (`process_document`), not just by
calling `suggest_fields()` in isolation, using a key supplied for that
purpose and the bundled synthetic scans (fictional data, not a real
person's). On `scan_01_khatauni_up_clean_skew_blur.png` the rule-based
extractor handled 6 fields on its own and the LLM contributed 10 grounded
suggestions for the rest - and the value it adds is visible in the
difference: where the rule-based path returned `+ Lucknow`, `+ Uttar
Pradesh` and `2 YP0972237004` with OCR label-noise attached, the LLM
returned `Lucknow`, `Uttar Pradesh` and `YP0972237004`. Reading around OCR
noise is the specific thing it is good at.

Insisting on the *full pipeline* rather than the isolated function is what
made that verification worth doing: it is how the silent `MAX_PATH` OCR
failure in S3 was found (the LLM stage was being handed 0 characters of
text), and neither the unit suite nor a direct `suggest_fields()` call
would ever have revealed it. The same run also surfaced two bugs in this
module - a 20-second timeout that turned working calls into silent no-ops
at random, and transport failures being swallowed into an empty result
indistinguishable from "found nothing" (now raised, and reported as
`LLM_SUGGESTION_FAILED`). Testing also surfaced a real,
non-obvious finding worth recording rather than glossing over: NVIDIA's
`GET /v1/models` lists roughly 80 catalog entries, but a given API key is
not necessarily provisioned to invoke all of them - this project's test
key returned `HTTP 404 "Function ... Not found for account"` on
`mistral-7b-instruct-v0.3` and several Nemotron/Mixtral/Gemma/Granite
variants, and only `meta/llama-3.2-11b-vision-instruct` actually worked.
That is why the NVIDIA default is a vision-instruct model for a text-only
task - it is simply the model this project could confirm works, not a
mismatch - and why `NVIDIA_MODEL` exists as an override: a different key
may have a different working set, and there is no way to know except by
trying, the same way this project found out.

### 12f. Post-OCR vocabulary correction (`backend/gazetteer.py`)

TrOCR and Tesseract both make mistakes on faded ink, and a place name is not
free text - it comes from a closed list. This layer corrects extracted values
against known vocabularies with RapidFuzz, and it is the largest module in
the project after `server.py` and `ocr_engine.py`.

**Scoped top-down, because khasra numbers are not unique.** District is
matched against all 43 in the bundled LGD extract; tehsil only within the
resolved district; village only within the resolved tehsil. Two villages
routinely reuse the same khasra number, so matching a village name against
every village in India would cheerfully tag a Bhopal record onto a Lucknow
parcel.

**Crossing scripts through a phonetic key.** RapidFuzz is script-agnostic but
the bundled LGD extract is entirely romanised, so a Devanagari value scores
0.0 against every candidate - and `extractOne` still names a winner, which
is a confident answer computed from nothing. Devanagari values are therefore
reduced to a phonetic key first: `romanise_devanagari()` (a self-contained
~90-line table, see below) then Hindi schwa deletion, so `नरहरपुर` keys as
*narharpur*, which is how LGD writes it. Measured on 15 district pairs whose
romanisation is independently known: **13 applied, 13 correct, 0 wrong, 2
refused.** A transliterated match always goes to a human.

**The romanisation table is hand-written on purpose.** `indic-transliteration`
was used first and worked, but installing it **broke spaCy** on this machine -
Application Control then refused `spacy/pipeline/multitask.pyd` and the NER
cross-check silently dropped to five skipped tests. Uninstalling it restored
spaCy, which identified the cause beyond doubt. Trading a working feature for
a new one is not a trade worth making, and the library was doing far more
than is needed here.

**Per-level thresholds, measured rather than guessed:** district and state
84, tehsil 88, village 92, suggest-only 80, ambiguity margin 3.0. Villages
get the strictest bar because village names are numerous and mutually
similar; the effect is that the layer **refuses** far more often than it
corrects, which is the right trade for land records. Identifiers are not
fuzzy-matched at all - `khasra`, `khata` and `ULPIN` get regex shape checks
(ULPIN is pinned at exactly 14 characters per DILRMP 3.0).

**What it cannot do.** No similarity threshold separates an OCR error from a
genuinely different village. Measured recovery against character error rate,
using the real LGD village list: at 10% CER only ~20% of village names are
recovered, at 30% CER ~5%. What the layer *reliably* delivers is a **0-1%
wrong-correction rate** across every condition tested. It is a safety net,
not an accuracy multiplier, and stacking a language model behind it does not
change that - MuRIL scored 2 of 5 on the same task and *preferred* digit-
corrupted forms such as `भ0पाल` over `भोपाल`.

### 12g. Handwriting: detection (`handwriting.py`) and transcription (`trocr_htr.py`)

Old land records are printed forms with handwritten patwari entries, so the
two have to be told apart before either is trusted.

**Detection is novelty scoring, not recognition.** Four features per line -
`stroke_width_cv`, `baseline_residual`, `height_cv`, `gap_cv` - scored
against a print profile fitted by `tools/fit_print_profile.py` at the 0.995
empirical quantile. A flagged line has its confidence clamped to **0.35**,
because Tesseract does not fail loudly on handwriting: it returns plausible
text at an ordinary confidence, so the confidence is the thing that has to be
corrected or extraction trusts it downstream.

Two failures shaped this and are worth recording. Scoring **whole lines**
flagged fully-printed form rows at 9-15 sigma purely from the label-to-value
gap; the fix was segmenting at wide gaps. And the first detector was
**inert** - fitted on degraded print, `baseline_residual`'s standard
deviation exceeded its mean, so a mean+4-sigma bar sat at 0.41 while real
handwriting measures 0.065. Its 1.40% false-positive rate looked excellent
because it flagged nothing. Empirical quantiles replaced the sigma bar, with
a sigma fallback below 100 samples.

A line that *cannot* be judged keeps `handwriting=None`, which is
deliberately distinct from a verdict of "printed" - the caller must not be
able to mistake "we did not look" for "we looked and it was fine".

**Honest number:** on a genuine old handwritten Devanagari manuscript,
recall is roughly **17%** with 3 of 5 flags wrong. It is reported as
detection, never as reading.

**Transcription via TrOCR** runs through ONNX Runtime rather than torch, and
`SUPPORTED_SCRIPTS = {"latin"}` refuses other scripts **before** the model
runs rather than returning confident nonsense. The encoder must be fp32: an
int8 encoder turned a cursive "industrie" into `insalums true`, while an
int8 *decoder* paired with an fp32 encoder measured as good as fp32/fp32.

### 12h. Georeferencing and shapefile import

`backend/georeference.py` reads georeferencing a surveyor has already done in
ArcGIS or QGIS, rather than asking them to redo it as control points:
ESRI world files (six coefficients in order **A, D, B, E, C, F** - not the
order most people assume), GeoTIFF `ModelPixelScale` / `ModelTiepoint` /
`ModelTransformation` tags with the half-pixel corner-to-centre shift
applied, and EPSG recovery from GeoTIFF geokeys or an ArcGIS `.aux.xml`
sidecar. Longitude/latitude plausibility is checked on **all four corners**,
not one.

`backend/shapefile_import.py` reads `.shp` geometry, `.dbf` attributes and
`.prj` projection with `struct` - no GDAL, no pyproj, no geopandas, because
that stack is the single most common reason a demo will not start on a
locked-down Windows machine. Two details that were bugs first: deleted DBF
rows are retained as tombstones (`DELETED_KEY`) because dropping them shifts
positional alignment and gave parcel 4 parcel 5's khasra number; and
Shapefile outer rings wind **clockwise** while RFC 7946 GeoJSON requires
**counter-clockwise**, so rings are re-wound on import.

`fit_transform_robust()` runs the loop a surveyor runs by hand: fit, inspect
per-point residuals, drop the worst control point, re-fit, until the RMS is
within tolerance (default 2.0 m, the loose end of the DILRMP village band).
Residuals are reported **in metres**, because 0.02 degrees is over 2 km and
must not read as "0.02". It stops at 4 points: an affine fit has 6
parameters, so 3 points fit exactly and always report zero residual, which
would manufacture a perfect-looking fit out of a bad one - `redundancy:
False` says so explicitly. Both the pre-drop and post-drop RMS are reported,
and all of it now reaches `audit_log`, which it previously did not: a
boundary dispute turns on who georeferenced a sheet, from which points, at
what residual.

### 12k. Approximate geotagging from place names (`backend/geocode.py`)

Every path in 12a and 12h needs an external reference: a matched cadastral
parcel, a world file, at least 3 GCPs, or a shapefile. Most documents people
actually upload have none of them. A deed, an attorney grant or a mutation
order names a village and a district and contains no coordinate at all.

**Why that is a hard limit, not a missing feature.** A pixel-to-lon/lat
affine transform is six numbers:

```
lon = a*px + b*py + c
lat = d*px + e*py + f
```

A scale bar pins the scale `s`; a north arrow pins the rotation `theta`.
Between them they determine `a, b, d, e` - the linear part, meaning size and
orientation. They say nothing about `c` and `f`, and `c, f` **is** the
position. Two unknowns, zero equations.

This is demonstrable rather than merely arguable: place the same vectorised
parcel at Bhopal, Narela and Mysuru with the same scale and rotation, and
every internal measurement - side lengths, angles, bearings, adjacency - is
**bit-identical** (max difference 0.000e+00 m across the three). Every
measurable property of the image is invariant under translation, so no
measurement of the image can distinguish the placements. Absolute position
is absent from the input, and no model recovers it.

A useful corollary the code does **not** yet exploit: with a legible scale
bar and north arrow, the linear part is already known, so **one** identified
point suffices (2 equations, 2 unknowns). `fit_affine_transform` currently
requires 3 unconditionally. Reducing that to 1 for sheets with a scale bar
is a real, solvable reduction in operator effort.

**What this module does instead.** It positions the record by the place
names it carries, against `backend/data/place_coords.json` - 36 states/UTs,
106 district headquarters, 9 localities, 189 aliases. Measured on the real
Delhi GPA, whose village field reads `VILLAGE NARELA, SABOLI ROAD, DELHI`
and for which `georeference.discover()` returns `None`, this yields
`28.8530, 77.0920` at locality precision.

The whole design is about not letting that pass for a survey result:

| | parcel path (12a/12h) | this module |
|---|---|---|
| rule | `GEO_TAGGED` | `GEO_PLACE_APPROXIMATE` |
| precision | parcel | locality 5 km / district 40 km / state 300 km |
| `accuracy_m` | `null` | always populated |
| area cross-check | yes (`GEO_AREA_MISMATCH`) | **never** |
| can change a decision | yes | no |

The area cross-check is deliberately absent. A district centroid cannot
confirm or deny 57 square yards, and running the comparison anyway would
manufacture agreement out of nothing. More generally a coordinate inferred
from a name carries no new information about the parcel, so it cannot
contradict anything the record says - hence `info` severity only. Parcel
geometry always wins; this runs only when `_parcel_for_record` returns
nothing.

Three details that were found by measurement, not design:

- **`gazetteer.normalise()` does not case-fold.** The all-caps token
  `NARELA` lifted from a deed never met the table entry `Narela`, so the
  real GPA resolved to *nothing* while the tidier probe `Narela` resolved
  fine. `_fold()` fixes it.
- **The romaniser doubles long vowels**, so नरेला becomes `narelaa` and भोपाल
  `bhopaala`; `bhopaala`/`bhopal` scores 0.857 and falls under the fuzzy
  threshold. Collapsing vowel runs before the phonetic key recovers it.
- **Phonetic bridging alone reached only 3 of 8** measured Devanagari/Latin
  pairs. दिल्ली romanises to `dillii` and लखनऊ to `lakhanou`, which no
  phonetic key will ever join to *Delhi* or *Lucknow* - those are
  transliteration-convention differences, not phonetic ones. Explicit
  Devanagari aliases in the reference data carry them, which is why the data
  file is as much the feature as the code. Historical English names
  (Allahabad/Prayagraj, Bombay/Mumbai, Bangalore/Bengaluru) are aliased for
  the same reason: records are written across decades.

A place name that contradicts the record's own stated state (Narela with
state Kerala) raises `GEO_PLACE_CONFLICT` as a **warning** and downgrades an
auto-approval. The stated state wins the coordinate, but the disagreement is
surfaced rather than hidden behind a confident pin.

**Still not wired:** an embedded map on an uploaded document is OCR'd as
text and discarded. `process_document` does not call `cadastral.vectorize`,
so geometry only enters via maps imported separately through
`/api/cadastral/maps`. That is a wiring job, and unlike the translation
problem above it is entirely solvable.

### 12i. Topology validation (`backend/topology.py`)

Vectorising a village map produces polygons that look right and can still be
topologically wrong, and the wrong ones are the ones that matter: a gap
between two parcels is land belonging to nobody on the map, an overlap is
land given to two people at once. Neither is visible by eye on a sheet with
hundreds of parcels. Pure Python geometry, no shapely.

Five rule types - invalid rings (self-intersection, degenerate, zero area),
overlaps, slivers, containments, unsnapped vertices - plus idempotent vertex
snapping. Area against the recorded khasra area is checked separately by
`GEO_AREA_MISMATCH` at 15% tolerance, as a *warning*, because a disagreement
can as easily mean the map was digitised imprecisely as that the register is
wrong.

Two design points carry the whole module:

- **A shared edge is not an overlap.** Every interior boundary on a village
  map is shared by two parcels, so a plain intersection test reports every
  correctly-digitised sheet as broken - which is worse than no check, because
  it trains a reviewer to ignore the report. `segments_properly_cross()`
  requires an interior crossing, and `point_in_ring()` returns `boundary` as
  a third answer distinct from `inside`.
- **Containment is not an overlap either.** The vectoriser also picks up the
  sheet's outer envelope. Measured on the real Bhu-Naksha report, a naive
  test reported **6 overlaps and every one of them was the envelope
  containing the parcels inside it** - zero real findings. Separating
  containment onto its own channel left **1 overlap and 5 containments**, and
  the clean synthetic demo map still returns zero of everything.

Tolerances must be converted before use: `suggested_tolerance()` exists
because 0.5 read as degrees is a 55 km tolerance that would collapse a
district onto one point, silently.

### 12j. Learned boundary segmentation (`backend/boundary_net.py`) - trained, and not transferring

A U-Net with a transformer bottleneck, trained to predict parcel boundaries.
It is **optional, off by default, and the classical vectoriser remains the
default**, for reasons the numbers below make plain.

**Why it was built.** The classical vectoriser is excellent on a clean sheet
and brittle in one specific way. Measured on the real plot report, and then
on synthetic sheets one damage family at a time:

| damage | classical parcel recall |
| --- | --- |
| undegraded | 7 of 7 |
| **line breakage 5%** | **0 of 7** |
| fade 85% | 7 of 7 |
| noise sigma 25 | 2 of 7 |
| `breaks` only (20 sheets) | **0.0%** - 20 of 20 sheets yielded zero parcels |
| `fade` / `blur` / `speckle` / `stain` / `crease` / `bleed` / `noise` only | 98-100% |

Fading is survivable because thresholding handles contrast. **Broken linework
is fatal**, because contour extraction needs a topologically closed boundary
and a single gap lets the region leak away. The task *is* line breakage;
nothing else meaningfully defeats the classical path.

**The transformer is not decoration.** Closing a gap in a long boundary needs
evidence from both sides of it, often a hundred pixels apart. Convolutions
reach that distance only by stacking depth, which blurs the boundary they are
trying to localise; attention at the coarsest level relates every position to
every other in one step. The positional embedding is bilinearly
**interpolated** to whatever grid it is handed - an earlier version skipped
it on a size mismatch, which meant training on 256px crops and inferring on
512px tiles applied it at inference and never during training.

**Training data is synthetic** (`tools/make_boundary_dataset.py`), because a
U-Net needs thousands of annotated sheets and this project has one. A drawn
map knows its own geometry, so ground truth is pixel-exact and free. The mask
contains **boundaries only** - khasra numerals, north arrow, scale bar and
sheet frame are drawn into the image and excluded from the mask, which is the
entire learning signal; without them the task is thresholding with extra
steps. One shared warp is applied to the whole parcel network rather than
per-parcel jitter, so adjacent parcels keep sharing corners instead of the
dataset teaching the model that correct sheets contain sliver gaps.

**Results - and the part that matters:**

| evaluation | model | classical |
| --- | --- | --- |
| in-family (val) | IoU 0.758, **93.3%** parcel recall | 69.5% |
| out-of-family (`dashes`, unseen mechanism) | IoU 0.753, **62.0%** | 0.7% |
| **the one real scanned sheet** | **0 of 7 parcels** | **7 of 7** |

The out-of-family design took two attempts. The first holdout was `bleed` -
then measurement showed classical extraction scores 100% on bleed-only
sheets, so the number would have compared against a baseline that was never
challenged. `breaks` has to stay in training because it is the capability
being bought, so the honest holdout is a *different breakage mechanism*:
`dashes` removes whole contiguous segments (classical: 1.7%). On that, IoU
barely moved and parcel recall was 88x the classical baseline - the model
genuinely learned to complete a boundary rather than to fill small holes.

**And it still recovers nothing on a real sheet.** Not zero output - 1.65% of
pixels fire above 0.5 - but boundaries that do not close into regions. Scale
mismatch was tested and ruled out as the cause (0 parcels at every
downscaling except 0.25x, which gave 2). 2,400 samples from one generator is
one narrow visual world, and a real scan differs in line character, paper
texture, compression artefacts and annotation style in ways the generator
does not reproduce. This is the denoiser's synthetic-to-real gap
(+5.8 dB in-family, +1.6 dB out-of-family) except worse: positive on
synthetic, useless on real.

Closing it needs the domain gap measured against real sheet statistics and
then, most likely, few-shot fine-tuning on hand-annotated real maps. Until
that is done and measured, this layer earns nothing and claims nothing.

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
- **Master data is real but bounded.** `admin_master.json` now carries real
  state/district/tehsil codes and names from the Local Government Directory
  (LGD, Ministry of Panchayati Raj) for 8 states / 43 districts - not
  fabricated placeholders - via the community CSV mirror at
  [planemad/india-local-government-directory](https://github.com/planemad/india-local-government-directory)
  (snapshot 2022-03-11), also cross-published on
  [data.gov.in](https://www.data.gov.in/catalog/local-government-directory-lgd).
  District and tehsil coverage is complete for the districts included.
  Village coverage is a capped, deduplicated subset (up to 15 real villages
  per tehsil, ~6,500 total) - a real village absent from the bundle is
  unverifiable, not invalid, which is why `village_tehsil_conflict()` only
  ever raises an issue when a village is positively found under a *different*
  tehsil, never merely because it is missing from the subset.
- **Precision figures start empty.** Until a human reviews fields, the
  dashboard shows a dash. This is intentional.
- **Multilinguality is layered, not uniform - answer it per layer.** OCR is
  genuinely multilingual: all ten Indic packs are installed, all ten are
  selection candidates, and the right pack was chosen 9 times out of 9 when
  the same record was rendered in nine scripts. Extraction is not:
  `field_extractor` carries 218 labels across eight scripts but unevenly -
  Devanagari is complete, Bengali and Gurmukhi lack khata labels, and
  **Malayalam and Odia have no labels at all** despite working OCR packs.
  The gazetteer bridge is Devanagari-only (`_TRANSLITERABLE`), and the NER
  cross-check is Latin-only. Measured end-to-end recall by script:
  Devanagari 100%, Gujarati 100%, Bengali 80%, Tamil 50%, Kannada 50%,
  Gurmukhi 40%, Telugu 25%, Malayalam and Odia untestable. "It supports ten
  scripts" is true of OCR and of nothing else.
- **The learned boundary segmenter does not transfer to real sheets.** See
  S12f. It is trained, it works on synthetic sheets, and it recovers **0 of
  7** parcels on the one real scanned map this project has. It is wired as
  optional and off; the classical vectoriser remains the default.
- **Topology validation detects, it does not measure.** `topology.py` reports
  *that* two parcels overlap and where, not how many square metres they
  share - that needs a general polygon clipper. Gaps are inferred from
  near-coincident vertices rather than a true topological union, so a
  genuinely unmapped hole in the interior of a village would not be found.
  Both caveats are returned in the report itself rather than left implicit.
- **The parcel identifier is now region-aware (fixed).** This entry used to
  record a real defect: `khasra_number` was hardcoded mandatory, but
  Karnataka, Tamil Nadu, Kerala, Andhra Pradesh and Telangana issue no
  khasra at all - their parcel identifier is the survey number, and West
  Bengal's is the *dag*. A flawlessly-extracted Tamil Nadu record, every
  field at 0.95+ confidence, was still `blocked` on `REQUIRED_MISSING`,
  which made the system's accuracy on half the country irrelevant.
  `validator.IDENTIFIER_GROUP` now treats khasra-or-survey as one
  "at least one of" requirement: either satisfies it, both missing is still
  an error, and the message names both forms rather than demanding a
  document that does not exist in the reader's state.
- **Script selection was verified on four scripts, not all ten, and on
  rendered rather than scanned non-Devanagari pages.** Tamil and Telugu test
  pages were generated with a Unicode font (there is no real Tamil or
  Telugu land record in this project to test against), so they are clean
  digital renders - a genuinely scanned, faded Tamil khatauni is a harder
  case than anything selection has been shown. Bengali, Kannada, Gujarati,
  Gurmukhi, Malayalam, Odia and Urdu are wired in and untested. The
  mechanism is script-agnostic, but "wired in" and "verified" are different
  claims and only Devanagari, Tamil and Telugu have the second one.
- **Selection picks a script, not a language.** A Marathi record is read by
  the Hindi model because both are Devanagari; that is correct for the
  recogniser but means `mar`, `san` and `nep` are installed and never used.
  Choosing between same-script models needs a ground-truth accuracy
  measurement this project does not have - confidence alone was shown to be
  the wrong signal for it.
- **Selection costs 6-14 seconds per scanned document** (ten trial passes,
  dominated by Tesseract process startup rather than image size, which is
  why downscaling barely helps). Paid once per document and never for PDFs
  with a text layer, but it is a real slowdown of the scanned-image path in
  exchange for reading it with the right model at all.
- **Deskewing was being done in the band where it hurt, and the synthetic
  corpus could never have shown it (fixed).** The rotation threshold was 0.4
  degrees. On 32 real pages (the 15 Bhu-Naksha reports plus a 13-page
  notarised GPA) only 7 produce a Hough skew estimate at all, and every one
  falls between **1.22 and 1.88 degrees** - so all 7 were rotated, and
  `warpAffine` with `INTER_CUBIC` resampled every pixel, softening strokes
  for no layout gain because Tesseract absorbs skew in that band unaided.
  The synthetic corpus is skewed **2.4 and 3.6 degrees by construction**
  (`tools/make_samples.py`), so it never contained a page in the range where
  the bug lived, and gated Hough deskew duly measured *exactly zero
  difference* there. The threshold is now `MIN_DESKEW_DEGREES = 2.0`, which
  sits in a genuine gap (1.878 below, 2.291 above - the estimator's reading
  of the 2.4-degree synthetic scan), so rotation is retained for genuinely
  crooked scans and declined where it was measured to cost. Visible on one
  real field: rotated, the GPA's area read `MEAS. 57 5..0 . ..25.` and
  parsed to nothing; unrotated, `MEAS. 57 SQ.YDS.` parsing to 47.659 m2.
  This is why the 84.4% real-corpus figure was previously only reproducible
  with scikit-image pinned off - a dependency being *installed* changed
  accuracy, which is the kind of thing that should never be true silently.
- **Cadastral vectorization runs only on the bundled synthetic map.** No
  real cadastral map and no real surveyed ground control points exist here
  (S12a) - only a generated demo map and illustrative placeholder GCPs. The
  vectorization and georeferencing code is real, not the input it was
  verified against.
- **LLM field suggestions (S12e) send land-record PII to a third party when
  enabled.** This is the one capability in the whole project that is not
  fully local, which is why it needs two separate opt-in flags instead of
  one and is off by default. The NVIDIA path was verified against a real
  API call (see S12e); Anthropic and OpenAI were not, since no key for
  either exists in the environment this was built in, and are only
  unit-tested with the provider call monkeypatched.
- **The LLM fabricates, and the grounding guard is a floor, not a
  guarantee.** Observed directly, not theorised: given a degraded scan the
  model produced a complete invented land record (`Khasra No. 1234`,
  `District - Alwar`) for a Madhya Pradesh document. `is_grounded()` (S12e)
  discards any suggestion not present in the source text, which caught
  every fabricated value in that run - but it can only prove a value
  *appears in the document*, never that it was read from the *right field*.
  A model that returns the area value as the share, or one row's date for
  another's, produces something grounded and still wrong. That is why these
  are advisory `LLM_SUGGESTION` issues for a human to accept or reject and
  never auto-applied field values, and it is a good reason to treat this
  feature as an assist for a reviewer rather than a step toward removing
  one.
- **A model listed in NVIDIA's `/v1/models` catalog is not necessarily
  invokable by a given key.** Verified directly, not assumed: this
  project's test key got HTTP 404 on `mistral-7b-instruct-v0.3` and
  several Nemotron/Mixtral/Gemma/Granite entries despite all of them
  appearing in the catalog listing, and only
  `meta/llama-3.2-11b-vision-instruct` actually worked. A different key
  may have a different working set - `NVIDIA_MODEL` exists specifically so
  this can be overridden per key rather than hardcoded as if universal.
- **Document-to-map geo-tagging only works for the one village that has a
  vectorized map.** Every real document from any other village correctly
  shows as unmatched on the Cadastral Map tab - there is no geometry to link
  it to, and this project does not fabricate any. Matching is now scoped by
  village (S12a), so khasra-number collisions between villages cannot
  produce a false match; the limit is coverage, not correctness. Such a
  document now still receives an **approximate** place-name geotag (S12k) at
  5-300 km precision, under its own rule name and with `accuracy_m` always
  stated, so it can never be mistaken for the parcel-grade result.
- **Absolute coordinates cannot be derived from an unreferenced scan, and
  this is geometric rather than a gap in the code.** A pixel-to-lon/lat
  affine transform has six parameters; a scale bar and a north arrow
  determine only the four that set size and orientation, leaving the two
  that set position with zero equations. Measured: the same parcel placed at
  Bhopal, Narela and Mysuru yields bit-identical internal geometry (max side
  difference 0.000e+00 m), so no measurement of the image can distinguish
  them. Position must come from a world file, GCPs, a shapefile or a
  gazetteer - see S12k, including the 1-GCP shortcut the code does not yet
  take.
- **The place-coordinate table is a curated subset, not a gazetteer.** 36
  states/UTs, 106 district headquarters and 9 localities. A district outside
  it degrades to its state (300 km), and a village outside it degrades to
  its district - never to a guess. Localities are Delhi-and-NCR heavy
  because that is where the validation documents came from; this is
  coverage, and it is the single easiest thing in the project to extend.
- **A district headquarters is not a district centroid.** Large districts
  carry real bias, which is why `accuracy_m` is reported as 40 km rather
  than anything tighter, and why no area or boundary check is ever run
  against a place-name coordinate.
- **The area cross-check compares against a map that is itself approximate,
  and on demo data the coordinates are illustrative.** Vectorising a scanned
  map and fitting an affine transform both carry error, so `GEO_AREA_MISMATCH`
  reports a disagreement, never a verdict on which side is wrong, and never
  blocks. On the bundled demo the control points are placeholders rather
  than a survey, so the measured areas are plausible in scale but are not
  real-world measurements - every message says so. The 15% tolerance is a
  judgement about what counts as a material difference, not a measured
  error bound, and should be retuned against real survey data.
- **Signature/seal/stamp-paper detection finds presence, never authenticity
  (S12b).** `SEAL_DETECTED` means a stamp-shaped coloured mark exists on the
  page, not that it is a genuine government or notary seal; the same
  applies to signatures. Confirming genuineness needs a reference signature
  database or a live government stamp/notary registry - the same category
  of gap as the DILRMP connector above. Treating a detected mark as proof of
  authenticity would be a real mistake in a land-record context, not just
  an inaccurate claim.
- **`SEAL_MISSING` blocking means a detector false negative now blocks a
  genuinely-stamped document, not just flags it.** A faint, small, or
  unusually coloured seal that the colour/shape heuristics in S12b miss
  will read as mandatory-and-absent. There is no separate override path
  for this yet beyond a verifier correcting whatever is actually wrong
  with the source scan (a clearer rescan, better lighting) and
  re-triggering revalidation.
- **The anomaly baseline (S12c) is only as good as an office's own approval
  history, and needs 50 approved documents before it activates at all.** A
  fresh install has no baseline and honestly reports
  `ANOMALY_BASELINE_NOT_TRAINED` rather than guessing. It also flags
  "statistically unusual", never "fake" - the same distinction as S12b.
- **Table/layout structure recognition (S12d) is off by default on this
  project's own reference environment, and impractically slow even where it
  does run.** PaddlePaddle does not support this machine's Python 3.14, so
  on a fresh checkout `TABLE_STRUCTURE_UNAVAILABLE` is what everyone will
  actually see unless they specifically set up a separate Python 3.8-3.12
  environment for it. In that compatible environment it was verified to
  work against a real scanned sample, but only after working around a real
  PaddlePaddle 3.3.0+ CPU-inference crash - and, working or not, one real
  page took ~20.7 minutes on this machine's CPU with no GPU. This is
  measured, not estimated. Unlike every other optional dependency in this
  project, "installed and working out of the box" and "practical to
  actually use on this hardware" are two separate claims here, and only the
  first one holds.

## 13a. Deployment

The backend is standard library only, so nothing is needed to merely run it.
`requirements.txt` pins the eight packages that make the **hosted** demo show
the full rule-based pipeline inside a free tier's memory limit, and documents
what each exclusion costs. Every pin was checked for a Linux wheel on Python
3.12, so the image compiles nothing.

Deliberately excluded, totalling ~680 MB: `torch` (BoundaryNet refinement),
`onnxruntime` (TrOCR transcription - handwriting is still *detected* and
routed to a human, which is the part that protects the record), `spacy`, and
`paddleocr`. That is the difference between fitting a 512 MB instance and
being evicted by it.

The `Dockerfile` installs Tesseract's **language packs**, which is why this
cannot be a plain buildpack deploy: pip cannot provide them, and without them
a Devanagari khatauni yields confident nonsense. It ends with a build gate
that greps `run.py --check`, so a missing pack fails the *build* rather than
the demo.

`run.py` reads `$PORT` and `$HOST`, because a managed host assigns the port
and requires `0.0.0.0` - bound to localhost a container passes its own health
check and is unreachable. Explicit flags still win, so a laptop run stays
private. `AUTO_SEED=1` ingests the bundled corpus on first boot, in a
background thread started *after* the socket is listening, since a health
check that cannot connect during a forty-second seed kills the container; it
drives the real `/api/seed` endpoint over loopback rather than keeping a
second, untested copy of the ingest loop.

**Measured on the built image:** 1.38 GB, 16 Tesseract languages present,
first response in ~10 s, peak memory **216 MiB** against a 512 MB cap.

**One honest limitation of the free tier.** Render's free CPU is heavily
throttled, and OCR is nothing but sustained CPU. Digital PDFs are unaffected
(text-layer extraction, 4.5 s each) but a scanned page that takes **8.7 s**
locally **did not finish in 600 s** hosted. The public demo therefore shows
the eleven digital documents; the OCR path is demonstrated locally. This is a
CPU limit, not a memory one - verified by running the same container under a
hard 512 MB cap, where all fourteen documents ingested including the scans.

`.dockerignore` keeps `.env` and the local database out of image layers: a
baked secret is readable by anyone who can pull the image, and deleting it in
a later layer does not remove it from the history. Secrets are supplied
through the platform's own environment settings; `.env.example` is the
committed template, carrying names and comments and no values.

## 14. Project layout

24 backend modules, 9 tools, 26 test files. Every line count below is real.

```
sih26018/
  run.py                     launcher, capability check, sample generation
  README.md
  backend/
    server.py                stdlib HTTP API + RBAC + static serving      1,700
    ocr_engine.py            quality gate, 3-tier extraction, psm + digit  1,250
    gazetteer.py             post-OCR vocabulary correction (S12f)           950
    field_extractor.py       17 fields, 218 labels in 8 scripts, confidence  845
    validator.py             rules, identifier group, trust score            680
    handwriting.py           print-profile novelty detection (S12g)          595
    shapefile_import.py      .shp/.dbf/.prj reader, no GDAL (S12h)           495
    db.py                    SQLite schema, hash-chained audit trail (S8a)    656
    bhashini.py              Indic script bridge + exonyms (S4a)             459
    ner_extractor.py         NER cross-check, English + Indic (S4b)          374
    topology.py              gaps/overlaps/containment/snap (S12i)           465
    boundary_net.py          U-Net + transformer, optional, off (S12j)       460
    georeference.py          world files, GeoTIFF tags, .aux.xml (S12h)      445
    geocode.py               approximate place-name geotag (S12k)          385
    cadastral.py             vectorization + robust GCP fitting (S12a)       490
    learning.py              correction mining and calibration               375
    trocr_htr.py             handwriting transcription via ONNX (S12g)       350
    llm_extractor.py         LLM suggestions, optional, off by default       305
    cnn_denoiser.py          DnCNN-lite denoiser, pure NumPy                 295
    fact_checker.py          ML fact-check against the external registry     280
    document_authenticity.py signature/seal/stamp-paper signals (S12b)       260
    ner_extractor.py         spaCy NER cross-check, Latin-script gated       225
    table_structure.py       PP-StructureV3 wrapper, optional (S12d)         200
    anomaly_detector.py      one-class anomaly baseline (S12c)               190
    data/admin_master.json     LGD-style master data (8 states)
    data/registry_master.json  bundled parcel registry extract
  frontend/
    index.html  styles.css  app.js        primary client, no build step
    react.html  react/                    alternate client, React 18 UMD
    login.html  supabase-config.js        client-side auth gate (see S13)
    graphql-client.js                     scaffold, no endpoint set
    vendor/leaflet/   vendor/react/       vendored, no CDN, no npm
  tools/
    make_samples.py            sample corpus generator
    make_cadastral_map.py      synthetic cadastral map + demo control points
    make_boundary_dataset.py   synthetic (image, boundary mask) pairs (S12j)
    train_boundary_net.py      trains the boundary segmenter (S12j)
    train_denoiser.py          trains cnn_denoiser, with gradient check
    train_learning_loop.py     mines confusions from verifier corrections
    fit_print_profile.py       fits the handwriting print profile (S12g)
    measure_accuracy.py        per-field extraction accuracy harness
    fetch_trocr.py             downloads the TrOCR ONNX weights
  tests/                       21 files, 502 tests (see S15)
  samples/                     14 generated documents
  samples/cadastral/           synthetic map, ground truth, demo control points
  storage/                     created at runtime: SQLite DB, uploads, previews,
                               model weights, boundary dataset
```

`storage/` is created on first run and can be deleted at any time to reset the
demo to a clean state. It also holds anything trained locally
(`storage/models/`), which is why the repository carries no weights.

## 15. Running the tests

Run everything:

```bash
PYTHONPATH=backend:tools python3 -m unittest discover -s tests -t tests -q
```

**657 tests, 3 skipped, about a minute.** Or run one file at a time:

```bash
python3 tests/test_validator.py -v            # business rules, identifier group
python3 tests/test_field_extractor.py -v      # script detection, 8-script labels
python3 tests/test_ocr_engine.py -v           # restoration, quality gate, psm, digits
python3 tests/test_gazetteer.py -v            # vocabulary correction, phonetic bridge
python3 tests/test_learning.py -v             # confusion mining and calibration
python3 tests/test_handwriting.py -v          # print-profile novelty detection
python3 tests/test_trocr_htr.py -v            # ONNX transcription + script guard
python3 tests/test_cadastral.py -v            # vectorization, robust GCP fitting
python3 tests/test_topology.py -v             # gaps, overlaps, containment, snap
python3 tests/test_georeference.py -v         # world files, GeoTIFF, .aux.xml
python3 tests/test_geocode.py -v              # place-name geotag, precision honesty
python3 tests/test_shapefile_import.py -v     # .shp/.dbf/.prj, winding, tombstones
python3 tests/test_boundary_net.py -v         # U-Net contract + dataset ground truth
python3 tests/test_cnn_denoiser.py -v         # conv primitives, tiled inference
python3 tests/test_fact_checker.py -v         # ML fact-check pipeline
python3 tests/test_ner_extractor.py -v        # NER cross-check, script routing
python3 tests/test_ner_indic.py -v            # Indic NER: labels, floor, degradation
python3 tests/test_bhashini.py -v             # script bridge, exonyms, authority
python3 tests/test_audit_chain.py -v          # tamper detection, from the attacker's side
python3 tests/test_document_authenticity.py -v
python3 tests/test_anomaly_detector.py -v
python3 tests/test_table_structure.py -v
python3 tests/test_llm_extractor.py -v
python3 tests/test_extraction_accuracy.py -v
python3 tests/test_cadastral_maps.py -v
```

All of it is stdlib `unittest`, so no test runner needs installing - there is
no pytest dependency and none is wanted. Tests that exercise an optional
layer are skipped (not failed) when it is absent, matching the project's
honest-degradation philosophy: the ML fact-check path skips without
scikit-learn, `test_boundary_net.py` skips its model half without torch while
still checking the dataset ground truth, and
install scikit-learn (`pip install scikit-learn`) to run them for real.
`test_table_structure.py` only exercises defensive result-parsing with
fixture objects, never the real PP-StructureV3 pipeline, so it runs in
milliseconds and passes identically whether or not paddleocr is installed -
see S12d for why that pipeline is deliberately never constructed inside the
test suite itself.
