"""
Field extraction and confidence scoring.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Turns unstructured OCR lines into the structured land-record schema named in the
problem statement: landowner details, survey number, khasra number, khata
number, plot area, village, tehsil, district, land classification, mutation
records and registration information.

Approach: a label-anchored extractor.
  * Each field owns a bilingual label lexicon (English + Devanagari + common
    transliterations) and a value pattern.
  * We locate the label, take the text to its right (or on the following line),
    then validate the candidate against the field's pattern.
  * Every extracted value gets a decomposed confidence score, so the
    verification queue can be ordered by genuine doubt rather than guesswork.

Why rules and not a large model: the target fields are highly templated across
revenue formats (khatauni, jamabandi, RoR), rules are auditable in a government
context, they need no GPU, and they degrade predictably. The learning loop in
`learning.py` layers statistical correction on top of this deterministic base.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------

# Devanagari digits -> ASCII. Real records mix both freely.
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("૦૧૨૩૪૫૬૭૮૯")})  # Gujarati
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("੦੧੨੩੪੫੬੭੮੯")})  # Gurmukhi

# Characters Tesseract routinely confuses inside numeric fields.
_OCR_NUMERIC_FIXES = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "|": "1", "!": "1",
    "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2",
})


def normalise_digits(text: str) -> str:
    return text.translate(_DIGIT_MAP)


def normalise(text: str) -> str:
    """NFC normalise, unify digits, collapse whitespace and stray punctuation."""
    text = unicodedata.normalize("NFC", text or "")
    text = normalise_digits(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u2010-\u2015]", "-", text)      # dash variants
    text = re.sub(r"[:：;]", ":", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_numeric(value: str) -> str:
    """
    Normalise a numeric/identifier value, repairing common OCR confusions.

    Subdivision suffixes are protected: survey and khasra numbers legitimately
    look like "142/2B" or "14A", so an uppercase letter sitting directly after
    a digit is treated as part of the parcel number. Without this guard the
    B->8 confusion rule would quietly rewrite a valid parcel as "142/28" - a
    corruption far worse than the misread it was meant to fix.
    """
    v = normalise_digits(value)
    kept = []
    for i, ch in enumerate(v):
        prev = v[i - 1] if i else ""
        nxt = v[i + 1] if i + 1 < len(v) else ""
        if ch.isupper() and prev.isdigit() and not nxt.isalpha():
            kept.append(ch)                 # subdivision suffix - keep verbatim
            continue
        ch = ch.translate(_OCR_NUMERIC_FIXES)
        if ch.isdigit() or ch in "/-.":
            kept.append(ch)
    return "".join(kept).strip("-/. ")


# --------------------------------------------------------------------------
# Area units
# --------------------------------------------------------------------------

# Conversion to square metres. Bigha/biswa are region-dependent; we use the
# common UP/Bihar values and flag the ambiguity rather than hiding it.
AREA_UNITS: Dict[str, Dict] = {
    "hectare": {"sqm": 10000.0, "labels": ["hectare", "hect", "ha", "हेक्टेयर", "हेक्टर"]},
    "acre":    {"sqm": 4046.86, "labels": ["acre", "एकड़", "एकड"]},
    "sqm":     {"sqm": 1.0, "labels": ["sq.m", "sqm", "sq m", "square metre", "square meter", "वर्ग मीटर"]},
    "sqft":    {"sqm": 0.092903, "labels": ["sq.ft", "sqft", "sq ft", "square feet", "वर्ग फुट"]},
    "bigha":   {"sqm": 2529.29, "labels": ["bigha", "बीघा", "बिघा"], "regional": True},
    "biswa":   {"sqm": 126.44, "labels": ["biswa", "बिस्वा", "विस्वा"], "regional": True},
    "guntha":  {"sqm": 101.17, "labels": ["guntha", "गुंठा"], "regional": True},
    "kanal":   {"sqm": 505.86, "labels": ["kanal", "कनाल"], "regional": True},
    "marla":   {"sqm": 25.29, "labels": ["marla", "मरला"], "regional": True},
    "cent":    {"sqm": 40.47, "labels": ["cent", "सेंट"], "regional": True},
}

_UNIT_LOOKUP: List[Tuple[str, str]] = sorted(
    ((lbl.lower(), key) for key, meta in AREA_UNITS.items() for lbl in meta["labels"]),
    key=lambda p: -len(p[0]),
)


def parse_area(raw: str) -> Optional[dict]:
    """Parse '0.4820 hectare', '2 बीघा 10 बिस्वा', '1,250 sq.m' -> structured area."""
    if not raw:
        return None
    text = normalise(raw).lower().replace(",", "")
    components: List[dict] = []

    # \b cannot be used here: Devanagari unit names such as "बीघा" end in a
    # combining matra, which Python treats as a non-word character, so "\b"
    # never matches after it and every bigha/biswa area silently failed to
    # parse. An explicit "not followed by a letter or matra" lookahead is used.
    for label, unit in _UNIT_LOOKUP:
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*" + re.escape(label)
                             + r"(?![\w\u0900-\u097f])", text):
            components.append({"value": float(m.group(1)), "unit": unit,
                               "label": label, "span": m.span()})

    # Drop components swallowed by a longer match (e.g. 'sq.m' inside 'sq.mtr').
    components.sort(key=lambda c: c["span"][0])
    kept: List[dict] = []
    for comp in components:
        if any(comp["span"][0] >= k["span"][0] and comp["span"][1] <= k["span"][1]
               for k in kept):
            continue
        kept.append(comp)

    if not kept:
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if not m:
            return None
        return {"raw": raw.strip(), "value": float(m.group(1)), "unit": None,
                "sqm": None, "components": [], "unit_missing": True}

    total_sqm = sum(c["value"] * AREA_UNITS[c["unit"]]["sqm"] for c in kept)
    regional = any(AREA_UNITS[c["unit"]].get("regional") for c in kept)
    return {
        "raw": raw.strip(),
        "value": kept[0]["value"] if len(kept) == 1 else None,
        "unit": kept[0]["unit"] if len(kept) == 1 else "composite",
        "sqm": round(total_sqm, 3),
        "components": [{"value": c["value"], "unit": c["unit"]} for c in kept],
        "regional_unit": regional,
        "unit_missing": False,
    }


# --------------------------------------------------------------------------
# Land classification vocabulary
# --------------------------------------------------------------------------

LAND_CLASSES: Dict[str, List[str]] = {
    "irrigated_agricultural": ["irrigated", "sinchit", "सिंचित", "nahri", "नहरी"],
    "unirrigated_agricultural": ["unirrigated", "un-irrigated", "asinchit", "असिंचित", "barani", "बारानी"],
    "agricultural": ["agricultural", "agriculture", "krishi", "कृषि", "खेती"],
    "residential": ["residential", "abadi", "आबादी", "आवासीय", "awasiya", "gharat"],
    "commercial": ["commercial", "vyavsayik", "व्यावसायिक", "वाणिज्यिक"],
    "industrial": ["industrial", "audyogik", "औद्योगिक"],
    "barren": ["barren", "banjar", "बंजर", "parti", "परती", "usar", "ऊसर"],
    "forest": ["forest", "van", "वन", "jungle", "जंगल"],
    "government": ["government", "sarkari", "सरकारी", "gram sabha", "ग्राम सभा", "shamlat"],
    "water_body": ["pond", "talab", "तालाब", "nadi", "नदी", "water body", "जलाशय"],
    "road": ["road", "rasta", "रास्ता", "सड़क", "path"],
}


# --------------------------------------------------------------------------
# Field schema
# --------------------------------------------------------------------------

@dataclass
class FieldSpec:
    key: str
    display: str
    labels: List[str]                    # bilingual label lexicon
    pattern: Optional[str] = None        # expected value shape
    kind: str = "text"                   # text | number | area | date | class | person
    required: bool = False
    max_len: int = 120


FIELD_SPECS: List[FieldSpec] = [
    FieldSpec("khasra_number", "Khasra Number",
              ["khasra", "khasra no", "khasra number", "खसरा", "खसरा संख्या", "खसरा नं", "gata", "गाटा", "gata sankhya"],
              r"^\d{1,5}(?:[/-]\d{1,4})*(?:\s*(?:क|ख|ग|अ|ब|[a-zA-Z]))?$", "number", required=True),
    FieldSpec("khata_number", "Khata Number",
              ["khata", "khata no", "khata number", "खाता", "खाता संख्या", "खाता नं", "khewat", "खेवट"],
              r"^\d{1,6}(?:[/-]\d{1,4})?$", "number", required=True),
    FieldSpec("survey_number", "Survey Number",
              ["survey", "survey no", "survey number", "सर्वे", "सर्वे संख्या", "sy no", "s.no", "resurvey"],
              r"^\d{1,6}(?:[/-][\dA-Za-z]{1,4})*$", "number"),
    FieldSpec("ulpin", "ULPIN / Bhu-Aadhaar",
              ["ulpin", "bhu-aadhaar", "bhu aadhaar", "भू-आधार", "unique land parcel"],
              r"^[A-Z0-9]{10,16}$", "text"),
    FieldSpec("owner_name", "Landowner Name",
              ["owner", "owner name", "landowner", "name of owner", "khatedar", "खातेदार",
               "स्वामी", "भूमिस्वामी", "स्वामी का नाम", "नाम", "bhumiswami", "raiyat", "रैयत"],
              None, "person", required=True),
    FieldSpec("father_name", "Father / Husband Name",
              ["father", "father name", "s/o", "d/o", "w/o", "पिता", "पिता का नाम",
               "पति", "पति का नाम", "walid"],
              None, "person"),
    FieldSpec("share", "Ownership Share",
              ["share", "hissa", "हिस्सा", "ansh", "अंश", "share fraction"],
              r"^\d{1,4}\s*/\s*\d{1,4}$|^\d{1,3}(?:\.\d+)?\s*%$", "text"),
    FieldSpec("area", "Plot Area",
              ["area", "total area", "plot area", "क्षेत्रफल", "रकबा", "rakba", "kshetrafal", "extent"],
              None, "area", required=True),
    FieldSpec("land_classification", "Land Classification",
              ["land type", "land classification", "classification", "bhumi prakar",
               "भूमि का प्रकार", "भूमि प्रकार", "वर्गीकरण", "nature of land", "land use"],
              None, "class"),
    FieldSpec("village", "Village",
              ["village", "gram", "ग्राम", "गाँव", "गांव", "mauza", "मौजा", "revenue village"],
              None, "text", required=True),
    FieldSpec("tehsil", "Tehsil",
              ["tehsil", "tahsil", "taluka", "taluk", "तहसील", "तालुका", "mandal"],
              None, "text"),
    FieldSpec("district", "District",
              ["district", "zila", "जिला", "ज़िला", "जनपद"],
              None, "text", required=True),
    FieldSpec("state", "State",
              ["state", "rajya", "राज्य", "प्रदेश"], None, "text"),
    FieldSpec("mutation_number", "Mutation Number",
              ["mutation", "mutation no", "namantaran", "नामांतरण", "नामान्तरण", "दाखिल खारिज",
               "dakhil kharij", "intkal", "इंतकाल"],
              r"^[A-Za-z0-9]{1,8}(?:[/-][A-Za-z0-9]{1,6})*$", "text"),
    FieldSpec("mutation_date", "Mutation Date",
              ["mutation date", "date of mutation", "नामांतरण दिनांक", "नामांतरण तिथि"],
              None, "date"),
    FieldSpec("registration_number", "Registration Number",
              ["registration", "registration no", "reg no", "deed no", "panjikaran",
               "पंजीकरण", "पंजीयन", "रजिस्ट्री", "बही", "document no"],
              r"^[A-Za-z0-9]{1,10}(?:[/-][A-Za-z0-9]{1,6})*$", "text"),
    FieldSpec("registration_date", "Registration Date",
              ["registration date", "date of registration", "deed date", "पंजीकरण दिनांक",
               "पंजीयन तिथि", "रजिस्ट्री दिनांक", "dated", "दिनांक"],
              None, "date"),
]

FIELD_BY_KEY: Dict[str, FieldSpec] = {f.key: f for f in FIELD_SPECS}

# Every label paired with the field that owns it. Used to settle collisions
# between a generic label and a more specific one on the same line.
_ALL_LABELS: List[Tuple[str, str]] = [
    (lbl.lower(), spec.key) for spec in FIELD_SPECS for lbl in spec.labels
]


# --------------------------------------------------------------------------
# Date parsing
# --------------------------------------------------------------------------

_HINDI_MONTHS = {
    "जनवरी": 1, "फरवरी": 2, "मार्च": 3, "अप्रैल": 4, "मई": 5, "जून": 6,
    "जुलाई": 7, "अगस्त": 8, "सितंबर": 9, "सितम्बर": 9, "अक्टूबर": 10,
    "नवंबर": 11, "नवम्बर": 11, "दिसंबर": 12, "दिसम्बर": 12,
}
_EN_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def parse_date(raw: str) -> Optional[dict]:
    """Parse Indian-format dates. Returns ISO date plus the assumption made."""
    if not raw:
        return None
    text = normalise(raw)
    notes: List[str] = []

    m = re.search(r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{2,4})\b", text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000 if y <= 30 else 1900
            notes.append("Two-digit year expanded.")
        if d > 31 and mo <= 12:      # yyyy-mm-dd style caught by loose regex
            return None
        if mo > 12 and d <= 12:
            d, mo = mo, d
            notes.append("Day/month order corrected.")
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return {"raw": raw.strip(), "iso": f"{y:04d}-{mo:02d}-{d:02d}", "notes": notes}

    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            return {"raw": raw.strip(), "iso": f"{y:04d}-{mo:02d}-{d:02d}", "notes": notes}

    for name, mo in _HINDI_MONTHS.items():
        m = re.search(r"(\d{1,2})\s*" + name + r"\s*(\d{4})", text)
        if m:
            return {"raw": raw.strip(),
                    "iso": f"{int(m.group(2)):04d}-{mo:02d}-{int(m.group(1)):02d}",
                    "notes": notes}

    m = re.search(r"(\d{1,2})\s*([A-Za-z]{3,9})\.?\s*(\d{4})", text)
    if m:
        mo = _EN_MONTHS.get(m.group(2)[:3].lower())
        if mo:
            return {"raw": raw.strip(),
                    "iso": f"{int(m.group(3)):04d}-{mo:02d}-{int(m.group(1)):02d}",
                    "notes": notes}
    return None


# --------------------------------------------------------------------------
# Label matching
# --------------------------------------------------------------------------

def _label_similarity(candidate: str, label: str) -> float:
    """
    Tolerant label match. OCR mangles labels ('Khasra' -> 'Khasm'), so an exact
    match is too brittle; we fall back to a character-bigram Dice coefficient.
    """
    c, l = candidate.lower().strip(), label.lower().strip()
    if not c or not l:
        return 0.0
    if c == l:
        return 1.0
    if l in c:
        return 0.92
    cb = {c[i:i + 2] for i in range(len(c) - 1)} or {c}
    lb = {l[i:i + 2] for i in range(len(l) - 1)} or {l}
    inter = len(cb & lb)
    if not inter:
        return 0.0
    return 2.0 * inter / (len(cb) + len(lb))


def _best_label_hit(line: str, spec: FieldSpec) -> Tuple[float, Optional[str], int]:
    """Return (score, matched_label, end_index_of_label) for the strongest label."""
    low = normalise(line).lower()
    best = (0.0, None, -1)
    for label in spec.labels:
        ll = label.lower()
        idx = low.find(ll)
        if idx >= 0:
            # Prefer labels that start the line or follow a separator.
            boundary = 1.0 if idx == 0 or low[idx - 1] in " |:,-([" else 0.82
            score = 0.95 * boundary
            if score > best[0]:
                best = (score, label, idx + len(ll))
            continue
        # fuzzy: compare against the leading segment before a colon
        head = low.split(":")[0]
        if len(head) <= 40:
            sim = _label_similarity(head, ll)
            if sim >= 0.68 and sim > best[0]:
                best = (sim * 0.85, label, len(head) + 1)
    return best


_STRIP_LEADING = re.compile(r"^[\s:：\-–—|=.,)\]}>]+")
_TRAIL_JUNK = re.compile(r"[\s:：\-–—|=.,(\[{<]+$")


def _candidate_after_label(line: str, end_idx: int) -> str:
    tail = line[end_idx:] if 0 <= end_idx <= len(line) else ""
    # Bilingual forms print rows as "खाता संख्या / Khata Number : 1428". Matching only the
    # Hindi half of the label leaves the English half sitting in front of the
    # real value. In a label-value row the value always follows the final
    # separator, so cut there instead of trusting where the matched label
    # happened to end.
    sep = max(tail.rfind(":"), tail.rfind("："))
    if 0 <= sep <= 60:
        tail = tail[sep + 1:]
    tail = _STRIP_LEADING.sub("", tail)
    # A second label on the same line ends this value (tabular rows).
    tail = re.split(r"\s{3,}|\s\|\s", tail)[0]
    return _TRAIL_JUNK.sub("", tail).strip()


# --------------------------------------------------------------------------
# Value validation per kind
# --------------------------------------------------------------------------

def _validate_value(spec: FieldSpec, raw: str) -> Tuple[Optional[str], float, dict, List[str]]:
    """
    Returns (normalised_value, pattern_confidence, structured_extra, notes).
    pattern_confidence is 0..1 and expresses 'does this look like the field'.
    """
    notes: List[str] = []
    raw = (raw or "").strip()
    if not raw:
        return None, 0.0, {}, ["Empty value."]
    if len(raw) > spec.max_len:
        raw = raw[:spec.max_len]
        notes.append("Value truncated to field length.")

    if spec.kind == "number":
        cleaned = _clean_numeric(raw)
        if not cleaned:
            return None, 0.05, {}, ["No digits found in a numeric field."]
        if _clean_numeric(raw) != normalise_digits(raw).strip():
            notes.append("OCR character substitutions applied (O->0, l->1).")
        conf = 0.95 if (spec.pattern and re.match(spec.pattern, cleaned)) else 0.45
        if not (spec.pattern and re.match(spec.pattern, cleaned)):
            notes.append("Value does not match the expected format for this field.")
        return cleaned, conf, {}, notes

    if spec.kind == "area":
        parsed = parse_area(raw)
        if not parsed:
            return None, 0.05, {}, ["Could not parse an area value."]
        conf = 0.94
        if parsed.get("unit_missing"):
            conf = 0.40
            notes.append("Area unit missing - cannot convert to square metres.")
        if parsed.get("regional_unit"):
            conf = min(conf, 0.78)
            notes.append("Regional unit (bigha/biswa/kanal) - conversion varies by state.")
        return parsed["raw"], conf, {"area": parsed}, notes

    if spec.kind == "date":
        parsed = parse_date(raw)
        if not parsed:
            return None, 0.08, {}, ["Could not parse a date."]
        notes.extend(parsed["notes"])
        return parsed["iso"], 0.92 if not parsed["notes"] else 0.72, {"date": parsed}, notes

    if spec.kind == "class":
        low = normalise(raw).lower()
        for canon, words in LAND_CLASSES.items():
            for w in words:
                if w.lower() in low:
                    return canon, 0.93, {"raw_class": raw}, notes
        return raw, 0.35, {"raw_class": raw}, ["Land class not in the controlled vocabulary."]

    if spec.kind == "person":
        v = re.sub(r"\s+", " ", raw).strip(" .,-")
        v = re.sub(r"^(shri|sri|smt|mr|mrs|श्री|श्रीमती)\s+", "", v, flags=re.I)
        if len(v) < 3:
            return None, 0.1, {}, ["Name too short to be valid."]
        if re.search(r"\d", v):
            notes.append("Name contains digits - likely bleed-through from an adjacent column.")
            return v, 0.38, {}, notes
        tokens = [t for t in v.split() if t]
        conf = 0.90 if 2 <= len(tokens) <= 5 else (0.66 if len(tokens) == 1 else 0.55)
        if len(tokens) == 1:
            notes.append("Single-token name - father/husband name may be merged or missing.")
        return v, conf, {}, notes

    # plain text (village / tehsil / district / state / misc)
    v = re.sub(r"\s+", " ", raw).strip(" .,-")
    if spec.pattern:
        ok = bool(re.match(spec.pattern, v))
        return v, 0.92 if ok else 0.45, {}, ([] if ok else ["Value does not match expected format."])
    if len(v) < 2:
        return None, 0.1, {}, ["Value too short."]
    conf = 0.86 if len(v) <= 40 else 0.6
    if len(v) > 40:
        notes.append("Unusually long value - may include neighbouring text.")
    return v, conf, {}, notes


# --------------------------------------------------------------------------
# Extraction result types
# --------------------------------------------------------------------------

@dataclass
class ExtractedField:
    key: str
    display: str
    value: Optional[str]
    raw_text: str = ""
    confidence: float = 0.0
    label_confidence: float = 0.0
    pattern_confidence: float = 0.0
    ocr_confidence: float = 0.0
    page: int = 1
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    source_line: str = ""
    notes: List[str] = dc_field(default_factory=list)
    extra: dict = dc_field(default_factory=dict)
    status: str = "extracted"     # extracted | missing | needs_review

    def to_dict(self) -> dict:
        return {
            "key": self.key, "display": self.display, "value": self.value,
            "raw_text": self.raw_text, "confidence": round(self.confidence, 4),
            "confidence_breakdown": {
                "label": round(self.label_confidence, 3),
                "pattern": round(self.pattern_confidence, 3),
                "ocr": round(self.ocr_confidence, 3),
            },
            "page": self.page, "bbox": list(self.bbox),
            "source_line": self.source_line, "notes": self.notes,
            "extra": self.extra, "status": self.status,
        }


# Weights for the composite confidence score.
W_LABEL, W_PATTERN, W_OCR = 0.40, 0.38, 0.22


def extract_fields(lines: List, review_threshold: float = 0.80) -> List[ExtractedField]:
    """
    `lines` is a list of ocr_engine.Line (or any object with .text/.confidence
    /.page/.bbox). Returns one ExtractedField per schema field.
    """
    norm: List[dict] = []
    for ln in lines:
        text = normalise(getattr(ln, "text", "") or "")
        if not text:
            continue
        norm.append({
            "text": text,
            "ocr": float(getattr(ln, "confidence", 0.0) or 0.0),
            "page": int(getattr(ln, "page", 1) or 1),
            "bbox": tuple(getattr(ln, "bbox", (0, 0, 0, 0)) or (0, 0, 0, 0)),
        })

    results: List[ExtractedField] = []

    for spec in FIELD_SPECS:
        best: Optional[ExtractedField] = None

        for i, row in enumerate(norm):
            label_conf, matched, end_idx = _best_label_hit(row["text"], spec)
            if label_conf < 0.55 or matched is None:
                continue

            candidate = _candidate_after_label(row["text"], end_idx)
            carried_idx = None
            # Label alone on its line -> the value sits on a following line.
            # Printed forms frequently strand the ":" separator on its own
            # line, so lines carrying no payload are skipped rather than
            # accepted as the value.
            if len(candidate) < 2:
                for j in range(i + 1, min(i + 4, len(norm))):
                    nxt = _candidate_after_label(norm[j]["text"], 0)
                    if nxt and nxt.strip(" :;-.।|_"):
                        candidate, carried_idx = nxt, j
                        break
            carried = carried_idx is not None

            value, pat_conf, extra, notes = _validate_value(spec, candidate)
            ocr_conf = (row["ocr"] if not carried
                        else min(row["ocr"], norm[carried_idx]["ocr"]))
            if carried:
                notes = notes + ["Value read from the line below its label."]

            # Cross-field disambiguation. The row "नामांतरण दिनांक / Mutation Date"
            # also contains "दिनांक", a label Registration Date matches on. When the
            # same line carries a longer, more specific label owned by another
            # field, this field is demoted so each date row feeds its own
            # field instead of both fields copying the first date they see.
            _low_line = normalise(row["text"]).lower()
            _rival = any(key != spec.key and len(lbl) > len(matched)
                         and lbl in _low_line for lbl, key in _ALL_LABELS)

            score = W_LABEL * label_conf + W_PATTERN * pat_conf + W_OCR * ocr_conf
            if _rival:
                score *= 0.35
            # A genuine label-value row nearly always carries an explicit
            # separator. Prose lines - page headings, seals, footnotes - that
            # merely happen to contain a label word are demoted, otherwise a
            # heading like "उत्तर प्रदेश शासन" outranks the actual State row.
            if ":" not in row["text"] and "：" not in row["text"]:
                score *= 0.55
            if value is None:
                score *= 0.35

            cand = ExtractedField(
                key=spec.key, display=spec.display, value=value,
                raw_text=candidate, confidence=score,
                label_confidence=label_conf, pattern_confidence=pat_conf,
                ocr_confidence=ocr_conf, page=row["page"], bbox=row["bbox"],
                source_line=row["text"], notes=notes, extra=extra,
            )
            if best is None or cand.confidence > best.confidence:
                best = cand

        if best is None or best.value is None:
            results.append(ExtractedField(
                key=spec.key, display=spec.display, value=None,
                confidence=0.0, status="missing",
                notes=["Field not found in the document."] if spec.required
                      else ["Field not present."],
            ))
            continue

        best.status = "needs_review" if best.confidence < review_threshold else "extracted"
        results.append(best)

    return results


def fields_to_dict(fields: List[ExtractedField]) -> dict:
    return {f.key: f.to_dict() for f in fields}


def summarise(fields: List[ExtractedField]) -> dict:
    """Document-level rollup used by the queue and the dashboard."""
    present = [f for f in fields if f.value is not None]
    required = [f for f in fields if FIELD_BY_KEY[f.key].required]
    missing_required = [f.key for f in required if f.value is None]
    low_conf = [f.key for f in present if f.confidence < 0.80]
    mean_conf = round(sum(f.confidence for f in present) / len(present), 4) if present else 0.0
    return {
        "fields_total": len(fields),
        "fields_extracted": len(present),
        "completeness": round(len(present) / len(fields), 4) if fields else 0.0,
        "required_missing": missing_required,
        "low_confidence_fields": low_conf,
        "mean_field_confidence": mean_conf,
    }
