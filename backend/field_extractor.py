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
from collections import Counter
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------

# Native digits -> ASCII, one script at a time. Real records mix native and
# ASCII digits freely within the same document, sometimes the same field.
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate("०१२३४५६७८९")}          # Devanagari
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("૦૧૨૩૪૫૬૭૮૯")})  # Gujarati
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("੦੧੨੩੪੫੬੭੮੯")})  # Gurmukhi
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("০১২৩৪৫৬৭৮৯")})  # Bengali
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("೦೧೨೩೪೫೬೭೮೯")})  # Kannada
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("౦౧౨౩౪౫౬౭౮౯")})  # Telugu
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate("௦௧௨௩௪௫௬௭௮௯")})  # Tamil

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


# --------------------------------------------------------------------------
# Script identification
# --------------------------------------------------------------------------

# Each major Indic script occupies its own contiguous Unicode block, so
# counting characters per block identifies the *script* deterministically -
# no model, no training data, no GPU. This identifies script, not language:
# Devanagari is shared by Hindi, Marathi, Nepali and others, which is exactly
# why the label lexicons below group by script rather than by language.
_SCRIPT_RANGES: Tuple[Tuple[str, int, int], ...] = (
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
)

# Same ranges, as a regex character-class fragment - used wherever a match
# must not be allowed to continue into a combining mark from any of these
# scripts (see the lookahead in parse_area()).
_INDIC_BLOCKS = "".join(f"\\u{lo:04x}-\\u{hi:04x}" for _, lo, hi in _SCRIPT_RANGES)


def detect_script(text: str) -> str:
    """
    Identify the dominant script in a line of text by Unicode block.

    Returns one of the names in _SCRIPT_RANGES, "latin" (English or
    romanised text), "mixed" (no script reaches a clear majority - typical
    of bilingual "लेबल / Label" rows), or "unknown" (no letters at all).
    """
    counts: Counter = Counter()
    for ch in text:
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            counts["latin"] += 1
            continue
        cp = ord(ch)
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] += 1
                break
    if not counts:
        return "unknown"
    total = sum(counts.values())
    top_script, top_count = counts.most_common(1)[0]
    if len(counts) > 1 and top_count / total < 0.55:
        return "mixed"
    return top_script


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
    "hectare": {"sqm": 10000.0, "labels": [
        "hectare", "hect", "ha", "हेक्टेयर", "हेक्टर",
        "হেক্টর", "ਹੈਕਟੇਅਰ", "હેક્ટર", "ஹெக்டேர்", "హెక్టారు", "ಹೆಕ್ಟೇರ್",
    ]},
    "acre": {"sqm": 4046.86, "labels": [
        "acre", "एकड़", "एकड",
        "একর", "ਏਕੜ", "એકર", "ஏக்கர்", "ఎకరం", "ಎಕರೆ",
    ]},
    "sqm": {"sqm": 1.0, "labels": [
        "sq.m", "sqm", "sq m", "square metre", "square meter", "वर्ग मीटर",
        "বর্গ মিটার", "ਵਰਗ ਮੀਟਰ", "ચોરસ મીટર", "சதுர மீட்டர்", "చదరపు మీటర్", "ಚದರ ಮೀಟರ್",
    ]},
    "sqft": {"sqm": 0.092903, "labels": [
        "sq.ft", "sqft", "sq ft", "square feet", "वर्ग फुट",
        "বর্গফুট", "ਵਰਗ ਫੁੱਟ", "ચોરસ ફૂટ", "சதுர அடி", "చదరపు అడుగు", "ಚದರ ಅಡಿ",
    ]},
    # Square yards, and its North Indian name, gaj.
    #
    # This was missing and it is not a minor omission: gaj/square yards is
    # THE unit for urban and peri-urban plots across North India, so every
    # city property document reported AREA_UNIT_MISSING. Measured on a real
    # Delhi GPA - "LAND AREA MEAS. 57 SQ.YDS." - the parser read the 57 and
    # returned unit=None, sqm=None, which the validator then correctly but
    # uselessly flagged as an error. The pipeline could size an agricultural
    # holding in bigha and not a plot in the capital.
    #
    # 1 sq yd = 0.83612736 m2 exactly (the yard is defined as 0.9144 m).
    # Not marked regional: unlike bigha, a square yard does not vary by
    # district, so no conversion caveat is warranted.
    "sqyd": {"sqm": 0.83612736, "labels": [
        "sq.yds", "sq yds", "sqyds", "sq.yd", "sq yd", "sqyd",
        "square yard", "square yards", "sq.yards", "sq yards",
        "gaj", "गज", "गज़", "वर्ग गज", "ਗਜ਼", "ગજ",
    ]},
    "bigha": {"sqm": 2529.29, "labels": [
        "bigha", "बीघा", "बिघा", "বিঘা",
    ], "regional": True},
    "biswa": {"sqm": 126.44, "labels": ["biswa", "बिस्वा", "विस्वा"], "regional": True},
    "guntha": {"sqm": 101.17, "labels": [
        "guntha", "गुंठा", "ಗುಂಠೆ",   # Maharashtra and Karnataka both use guntha
    ], "regional": True},
    "kanal": {"sqm": 505.86, "labels": ["kanal", "कनाल", "ਕਨਾਲ"], "regional": True},
    "marla": {"sqm": 25.29, "labels": ["marla", "मरला", "ਮਰਲਾ"], "regional": True},
    "cent": {"sqm": 40.47, "labels": [
        "cent", "सेंट", "சென்ட்", "సెంటు", "ಸೆಂಟ್",   # a common small-plot unit across the South
    ], "regional": True},
}

_UNIT_LOOKUP: List[Tuple[str, str]] = sorted(
    ((lbl.lower(), key) for key, meta in AREA_UNITS.items() for lbl in meta["labels"]),
    key=lambda p: -len(p[0]),
)


def _leading_area_unit(text: str) -> Optional[str]:
    """
    The unit name a line STARTS with, if any.

    Used only to recover an area whose unit wrapped onto the next line. It
    must match at the start: a unit appearing later in the line belongs to a
    different measurement - a deed reciting boundaries mentions feet several
    times - and grabbing one of those would attach a wrong unit to a value
    that merely lacked one.
    """
    low = normalise(text or "").lower().lstrip(" ,.:;-")
    best = None
    for label, _key in _UNIT_LOOKUP:
        if low.startswith(label) and (best is None or len(label) > len(best)):
            best = label
    return best


def parse_area(raw: str) -> Optional[dict]:
    """Parse '0.4820 hectare', '2 बीघा 10 बिस्वा', '1,250 sq.m' -> structured area."""
    if not raw:
        return None
    text = normalise(raw).lower().replace(",", "")
    components: List[dict] = []

    # \b cannot be used here: Indic unit names such as "बीघा" end in a
    # combining matra, which Python treats as a non-word character, so "\b"
    # never matches after it and the area would silently fail to parse. An
    # explicit "not followed by a letter or matra" lookahead is used instead,
    # covering every Indic script a unit label might be written in - not
    # just Devanagari, now that units are also labelled in Bengali, Gurmukhi,
    # Gujarati, Tamil, Telugu and Kannada.
    for label, unit in _UNIT_LOOKUP:
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*" + re.escape(label)
                             + r"(?![\w" + _INDIC_BLOCKS + r"])", text):
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
        # Second pass: tolerate ONE short OCR artefact between the number and
        # its unit.
        #
        # A smudged scan drops a few characters into the gap. Measured on a
        # real Delhi GPA, "LAND AREA MEAS. 57 SQ.YDS." came out of OCR as
        # "MEAS. 57 ong sq.yds" - the value and the unit both read perfectly,
        # separated by three junk characters, and the area was lost entirely.
        #
        # Deliberately a FALLBACK rather than a loosening of the main pattern:
        # a clean document never reaches here, so nothing that already parses
        # can be changed by it. The intervening token must be short and purely
        # alphabetic, which stops the rule bridging across a second number -
        # "17.3 x 3 sq.yds" must still attach the unit to the 3, not the 17.3.
        for label, unit in _UNIT_LOOKUP:
            for m in re.finditer(r"(\d+(?:\.\d+)?)\s+[a-z]{1,4}\.?\s+"
                                 + re.escape(label)
                                 + r"(?![\w" + _INDIC_BLOCKS + r"])", text):
                kept.append({"value": float(m.group(1)), "unit": unit,
                             "label": label, "span": m.span(),
                             "recovered_across_noise": True})
            if kept:
                break

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
    # Regional revenue terminology matters as much as the standard words: a
    # Rajasthan jamabandi writes चाही/बारानी and a Maharashtra 7/12 writes
    # बागायत/जिरायत, never सिंचित/असिंचित. Missing them meant the field was
    # dropped entirely on those states' records.
    "irrigated_agricultural": [
        "irrigated", "sinchit", "सिंचित", "nahri", "नहरी",
        "chahi", "चाही",                      # well-irrigated (Rajasthan/Punjab)
        "bagayat", "बागायत", "बागायती",        # irrigated (Maharashtra)
        "সেচ", "সেচযুক্ত", "ਸਿੰਚਾਈ", "સિંચાઈ", "பாசனம்", "నీటిపారుదల", "ನೀರಾವರಿ",
    ],
    "unirrigated_agricultural": [
        "unirrigated", "un-irrigated", "asinchit", "असिंचित", "barani", "बारानी", "बरानी",
        "jirayat", "जिरायत", "जिराईत",         # dry-crop (Maharashtra)
        "khushki", "खुश्की",                   # dry (Rajasthan)
        "বৃষ্টিনির্ভর", "ਬਾਰਾਨੀ", "બિનપિયત", "மானாவாரி", "మెట్ట", "ಮಳೆಯಾಶ್ರಿತ",
    ],
    "agricultural": [
        "agricultural", "agriculture", "krishi", "कृषि", "खेती",
        "কৃষি", "ਖੇਤੀਬਾੜੀ", "ખેતી", "விவசாயம்", "వ్యవసాయం", "ಕೃಷಿ",
    ],
    "residential": [
        "residential", "abadi", "आबादी", "आवासीय", "awasiya", "gharat",
        "আবাসিক", "ਰਿਹਾਇਸ਼ੀ", "રહેણાંક", "குடியிருப்பு", "నివాస", "ವಸತಿ",
    ],
    "commercial": [
        "commercial", "vyavsayik", "व्यावसायिक", "वाणिज्यिक",
        "বাণিজ্যিক", "ਵਪਾਰਕ", "વ્યાપારી", "வணிக", "వాణిజ్య", "ವಾಣಿಜ್ಯ",
    ],
    "industrial": [
        "industrial", "audyogik", "औद्योगिक",
        "শিল্প", "ਉਦਯੋਗਿਕ", "ઔદ્યોગિક", "தொழிற்துறை", "పారిశ్రామిక", "ಕೈಗಾರಿಕಾ",
    ],
    "barren": [
        "barren", "banjar", "बंजर", "parti", "परती", "usar", "ऊसर",
        "পতিত", "ਬੰਜਰ", "પડતર", "தரிசு", "బంజరు", "ಬಂಜರು",
    ],
    "forest": [
        "forest", "van", "वन", "jungle", "जंगल",
        "বন", "ਜੰਗਲ", "જંગલ", "காடு", "అడవి", "ಅರಣ್ಯ",
    ],
    "government": [
        "government", "sarkari", "सरकारी", "gram sabha", "ग्राम सभा", "shamlat",
        "সরকারি", "ਸਰਕਾਰੀ", "સરકારી", "அரசு", "ప్రభుత్వ", "ಸರ್ಕಾರಿ",
    ],
    "water_body": [
        "pond", "talab", "तालाब", "nadi", "नदी", "water body", "जलाशय",
        "জলাশয়", "ਤਲਾਬ", "તળાવ", "குளம்", "చెరువు", "ಕೆರೆ",
    ],
    "road": [
        "road", "rasta", "रास्ता", "सड़क", "path",
        "রাস্তা", "ਸੜਕ", "રસ્તો", "சாலை", "రోడ్డు", "ರಸ್ತೆ",
    ],
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


# khata_number, ulpin, mutation_number/date and registration_number/date are
# deliberately NOT extended with Bengali/Gurmukhi/Gujarati/Tamil/Telugu/Kannada
# labels below. Unlike khasra, survey number, owner, village etc., these
# concepts do not map cleanly across revenue systems - West Bengal's closest
# analogue to "khata" is a khatian, Karnataka's Bhoomi portal and Tamil Nadu's
# patta system record mutation differently again, and guessing a term is
# worse than not extracting the field: a wrong khata/mutation identifier is a
# property-dispute risk, not a cosmetic miss. These fields fall back to their
# existing Hindi/Marathi/English labels for those languages.
FIELD_SPECS: List[FieldSpec] = [
    FieldSpec("khasra_number", "Khasra Number",
              ["khasra", "khasra no", "khasra number", "खसरा", "खसरा संख्या", "खसरा नं", "gata", "गाटा", "gata sankhya",
               # Bhu-Naksha plot reports - the live public output of the UP
               # portal - print the same identifier as "Plot No". Verified
               # against 15 real reports: the "Plot No" value equals the
               # "खसरा नंबर" value on every one of them. Its absence here cost
               # twice over, because without it "PlotNo:50" was not a label at
               # all: it sat unrecognised inside the khata window and handed
               # khata the value 50 instead of 00620.
               "plot no", "plot number", "plot no.",
               "ਖਸਰਾ", "ਖਸਰਾ ਨੰਬਰ",             # Punjab also uses khasra terminology
               "দাগ", "দাগ নম্বর"],              # West Bengal's parcel-identifier equivalent is "dag"
              r"^\d{1,5}(?:[/-]\d{1,4})*(?:\s*(?:क|ख|ग|अ|ब|[a-zA-Z]))?$", "number", required=True),
    FieldSpec("khata_number", "Khata Number",
              ["khata", "khata no", "khata number", "खाता", "खाता संख्या", "खाता नं", "khewat", "खेवट"],
              r"^\d{1,6}(?:[/-]\d{1,4})?$", "number", required=True),
    FieldSpec("survey_number", "Survey Number",
              ["survey", "survey no", "survey number", "सर्वे", "सर्वे संख्या", "sy no", "s.no", "resurvey",
               "সার্ভে নম্বর", "ਸਰਵੇ ਨੰਬਰ", "સર્વે નંબર",
               "சர்வே எண்", "సర్వే నంబర్", "ಸರ್ವೆ ನಂಬರ್"],
              r"^\d{1,6}(?:[/-][\dA-Za-z]{1,4})*$", "number"),
    FieldSpec("ulpin", "ULPIN / Bhu-Aadhaar",
              ["ulpin", "bhu-aadhaar", "bhu aadhaar", "भू-आधार", "unique land parcel"],
              r"^[A-Z0-9]{14}$", "text"),     # DILRMP 3.0 pins ULPIN at 14 characters
    FieldSpec("owner_name", "Landowner Name",
              # Deed and attorney-grant vocabulary. On a GPA or sale deed
              # the person whose interest is being dealt with is the
              # "first party" / "purchased by" / "vendor", not an
              # "owner" - measured on a real Delhi e-Stamp GPA, where
              # none of the schema's 221 aliases matched any of the
              # document's 12 printed labels.
              #
              # "second party" is DELIBERATELY NOT mapped to
              # father_name. On a GPA the second party is the attorney
              # receiving the power, not a parent; filing them as
              # father/husband would be confidently wrong genealogy in a
              # land record, which is worse than leaving the field blank.
              ["first party", "purchased by", "vendor", "executant",
               "owner", "owner name", "landowner", "name of owner", "khatedar", "खातेदार",
               "स्वामी", "भूमिस्वामी", "स्वामी का नाम", "नाम", "bhumiswami", "raiyat", "रैयत",
               "মালিক", "মালিকের নাম", "জমির মালিক",
               "ਮਾਲਕ", "ਮਾਲਕ ਦਾ ਨਾਮ",
               "માલિક", "માલિકનું નામ",
               "உரிமையாளர்", "உரிமையாளர் பெயர்", "பட்டதாரர்",
               "యజమాని", "యజమాని పేరు", "పట్టాదారు",
               "ಮಾಲೀಕ", "ಮಾಲೀಕರ ಹೆಸರು"],
              None, "person", required=True),
    FieldSpec("father_name", "Father / Husband Name",
              ["father", "father name", "s/o", "d/o", "w/o", "पिता", "पिता का नाम",
               "पति", "पति का नाम", "walid",
               "পিতার নাম", "স্বামীর নাম",
               "ਪਿਤਾ ਦਾ ਨਾਮ", "ਪਤੀ ਦਾ ਨਾਮ",
               "પિતાનું નામ", "પતિનું નામ",
               "தந்தை பெயர்", "கணவர் பெயர்",
               "తండ్రి పేరు", "భర్త పేరు",
               "ತಂದೆಯ ಹೆಸರು", "ಗಂಡನ ಹೆಸರು"],
              None, "person"),
    FieldSpec("share", "Ownership Share",
              ["share", "hissa", "हिस्सा", "ansh", "अंश", "share fraction",
               "অংশ", "ਹਿੱਸਾ", "હિસ્સો", "பங்கு", "వాటా", "ಪಾಲು"],
              r"^\d{1,4}\s*/\s*\d{1,4}$|^\d{1,3}(?:\.\d+)?\s*%$", "text"),
    FieldSpec("area", "Plot Area",
              ["area", "total area", "plot area", "क्षेत्रफल", "रकबा", "rakba", "kshetrafal", "extent",
               "এলাকা", "জমির পরিমাণ", "ਖੇਤਰਫਲ", "ਰਕਬਾ", "ક્ષેત્રફળ",
               "பரப்பளவு", "విస్తీర్ణం", "ವಿಸ್ತೀರ್ಣ"],
              None, "area", required=True),
    FieldSpec("land_classification", "Land Classification",
              ["land type", "land classification", "classification", "bhumi prakar",
               "भूमि का प्रकार", "भूमि प्रकार", "वर्गीकरण", "nature of land", "land use",
               "किस्म भूमि", "किस्म जमीन", "भूमि श्रेणी", "जमिनीचा प्रकार",
               "জমির শ্রেণী", "ਜ਼ਮੀਨ ਦੀ ਕਿਸਮ", "જમીનનો પ્રકાર",
               "நில வகை", "భూమి రకం", "ಭೂಮಿಯ ಪ್ರಕಾರ"],
              None, "class"),
    FieldSpec("village", "Village",
              # "Property Description" is the field that names the place on a
              # deed or e-Stamp - "VILLAGE NARELA, SABOLI ROAD, DELHI" on the
              # measured GPA. It is a free-text locality rather than a bare
              # village name, so the gazetteer correctly reports it as
              # "no comparable vocabulary" instead of snapping it to a match.
              ["property description",
               "village", "gram", "ग्राम", "गाँव", "गांव", "mauza", "मौजा", "revenue village",
               "গ্রাম", "মৌজা", "ਪਿੰਡ", "ગામ", "கிராமம்", "గ్రామం", "ಗ್ರಾಮ", "ಹಳ್ಳಿ"],
              None, "text", required=True),
    FieldSpec("tehsil", "Tehsil",
              ["tehsil", "tahsil", "taluka", "taluk", "तहसील", "तालुका", "mandal",
               "ਤਹਿਸੀਲ",              # Punjab uses tehsil too
               "તાલુકા",              # Gujarat: taluka
               "தாலுகா",              # Tamil Nadu: taluk
               "మండలం",               # Telangana/Andhra: mandal
               "ತಾಲ್ಲೂಕು"],           # Karnataka: taluk
              None, "text"),
    FieldSpec("district", "District",
              ["district", "zila", "जिला", "ज़िला", "जनपद",
               "জেলা", "ਜ਼ਿਲ੍ਹਾ", "જિલ્લો", "மாவட்டம்", "జిల్లా", "ಜಿಲ್ಲೆ"],
              None, "text", required=True),
    FieldSpec("state", "State",
              ["state", "rajya", "राज्य", "प्रदेश",
               "রাজ্য", "ਰਾਜ", "ਸੂਬਾ", "રાજ્ય", "மாநிலம்", "రాష్ట్రం", "ರಾಜ್ಯ"],
              None, "text"),
    FieldSpec("mutation_number", "Mutation Number",
              ["mutation", "mutation no", "namantaran", "नामांतरण", "नामान्तरण", "दाखिल खारिज",
               "dakhil kharij", "intkal", "इंतकाल"],
              r"^[A-Za-z0-9]{1,8}(?:[/-][A-Za-z0-9]{1,6})*$", "text"),
    FieldSpec("mutation_date", "Mutation Date",
              ["mutation date", "date of mutation", "नामांतरण दिनांक", "नामांतरण तिथि"],
              None, "date"),
    FieldSpec("registration_number", "Registration Number",
              # An e-Stamp certificate number IS the instrument's
              # registration identifier for these documents.

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


# Words that mean "this line is a sentence", not a label-value row.
#
# Deeds and court orders recite the same vocabulary the schema looks for, in
# flowing prose. Measured on a real Delhi GPA, the label "PLOT NO" was found
# 38 characters into
#
#     "| owner/s and in possession of PLOT NO.55, LAND AREA MEAS. 57 Song"
#
# and everything after it was taken as the parcel number. The same failure
# shape put area, village and owner values from a Bhu-Naksha court-order
# narrative into fields that should have come from its PlotInfo block.
#
# Label DEPTH is not the discriminator, which is what makes this subtle: real
# tabular rows legitimately carry labels deep into the line -
# "Khata No: 00620 PlotNo:50 Area: 0.3120 Hectare" has its area label at
# character 26 of 46 and is perfectly good. What separates them is that prose
# is glued together by connectives and a tabular row is not.
_PROSE_WORDS = frozenset({
    "and", "of", "the", "in", "is", "at", "with", "to", "for", "by", "that",
    "this", "said", "above", "whose", "which", "from", "hereinafter",
    "possession", "liberty", "shall", "being", "who", "her", "his", "their",
    "having", "situated", "bounded", "executed", "whereas",
    # Devanagari connectives - the same problem in Hindi order text.
    "का", "के", "की", "में", "से", "और", "है", "को", "पर", "द्वारा", "अनुसार",
    "उक्त", "तथा", "हेतु", "किया", "गया",
})

_WORD_RE = re.compile(r"[a-z']+|[ऀ-ॿ]+")

# How much a label hit is discounted once its line looks like prose.
# A value scraped out of a sentence is never trusted like a tabular read,
# however confident the label match was.
PROSE_CONFIDENCE_CAP = 0.50

PROSE_DISCOUNT_STRONG = 0.45      # 3+ connectives: a sentence
PROSE_DISCOUNT_WEAK = 0.75        # exactly 2: ambiguous
NO_SEPARATOR_DISCOUNT = 0.70      # no ":" straight after the label


def _row_likeness(line: str, end_idx: int) -> float:
    """
    How much this line behaves like a label-value ROW rather than a sentence.

    1.0 is a clean row. Returning a multiplier rather than a hard reject
    matters: it lets a genuine row in a wordy document still win, and lets a
    prose hit through when nothing better exists anywhere - the field is then
    simply low-confidence and routed to a human, which is the correct outcome
    for a value scraped out of a paragraph.
    """
    tail = line[end_idx:end_idx + 4]
    separated = (":" in tail or "：" in tail
                 or tail[:1] in ("-", "=", "–", "—"))
    hits = sum(1 for w in _WORD_RE.findall(line.lower()) if w in _PROSE_WORDS)

    factor = 1.0
    if not separated:
        factor *= NO_SEPARATOR_DISCOUNT
    if hits >= 3:
        factor *= PROSE_DISCOUNT_STRONG
    elif hits == 2:
        factor *= PROSE_DISCOUNT_WEAK
    return factor


def _best_label_hit(line: str, spec: FieldSpec,
                    allow_prose: bool = False) -> Tuple[float, Optional[str], int]:
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
            if not allow_prose:
                score *= _row_likeness(low, idx + len(ll))
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


def _next_label_start(line: str, after_idx: int, own: "FieldSpec") -> int:
    """
    Where the next field's label begins on this line, or len(line).

    Real government output puts several labelled values on ONE line:

        Khata No: 00620  PlotNo:50  Area: 0.3120 Hectare

    Measured across 15 real Bhu-Naksha plot reports, that single line shape
    cost **every** area value - 0 of 15 extracted - and khata survived only
    because a later "Owner Details For Khata No.:- 00620" line happened to
    repeat it. Knowing where the next label starts is what turns one line
    into three separate label-value pairs.
    """
    low = normalise(line).lower()
    best = len(line)
    for spec in FIELD_SPECS:
        if spec is own:
            continue
        for label in (spec.labels or []):
            ll = label.lower().strip()
            if not ll:
                continue
            # Whitespace inside a label is optional, because OCR does not
            # reliably preserve it: the real reports come back as "PlotNo:50"
            # where the label is "plot no". Matching literally missed that,
            # left "PlotNo:50" inside the khata window, and handed khata the
            # value 50 instead of 00620.
            pattern = r"\s*".join(re.escape(part) for part in ll.split())
            m = re.compile(pattern).search(low, after_idx)
            if not m:
                continue
            idx = m.start()
            # Only a label sitting at a word boundary ends the value; "area"
            # inside a longer word is not a new field.
            if idx > 0 and low[idx - 1].isalnum():
                continue
            # A label only STARTS A NEW PAIR if a separator has already
            # appeared since the current label ended - meaning this field's
            # own value has been introduced and is complete.
            #
            # Without this, a word belonging to the CURRENT field's own label
            # phrase truncates its value. "पिता का नाम : रमेश" matches father
            # on "पिता", and "नाम" four characters later is an owner_name
            # label - so the window became " का " and the name was lost. That
            # cost four father names. In "Khata No: 00620 PlotNo:50" the
            # colon after 00620 shows khata's value is already done, so
            # "PlotNo" legitimately ends it.
            if not any(c in ":：|" for c in low[after_idx:idx]):
                continue
            if idx < best:
                best = idx
    return best


def _candidate_after_label(line: str, end_idx: int,
                           stop_idx: Optional[int] = None) -> str:
    """
    The value belonging to a label that ended at `end_idx`.

    `stop_idx` bounds the value at the next label on the line. Without it a
    multi-label row hands every field the LAST value on the line.
    """
    if not (0 <= end_idx <= len(line)):
        return ""
    stop = len(line) if stop_idx is None else max(end_idx, min(stop_idx, len(line)))
    tail = line[end_idx:stop]
    # Bilingual forms print rows as "खाता संख्या / Khata Number : 1428". Matching only the
    # Hindi half of the label leaves the English half sitting in front of the
    # real value. In a label-value row the value always follows the final
    # separator, so cut there instead of trusting where the matched label
    # happened to end.
    #
    # This searches only WITHIN the current field's window. Searching the
    # whole line - which it used to - made the separator of a LATER field win:
    # on "Khata No: 00620 PlotNo:50 Area: 0.3120 Hectare" the colon of
    # "Area:" was the last one, so khata was handed "0.3120 Hectare".
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

# Fields where a shape violation means "not this field" rather than "this
# field, read badly". These are the parcel and account identifiers that
# downstream records key on; a wrong one silently points at someone else's
# land, so a blank is strictly safer. Everything else keeps the old tolerance.
STRICT_IDENTIFIERS = frozenset({
    "khasra_number", "khata_number", "survey_number", "ulpin",
})


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
        matches = bool(spec.pattern and re.match(spec.pattern, cleaned))
        if matches:
            return cleaned, 0.95, {}, notes

        # A shape violation on a STRICT IDENTIFIER means "this is not that
        # field", not "this field, badly".
        #
        # Accepting it at 0.45 was how garbage reached the record. Measured on
        # a real Delhi GPA: the label "PLOT NO" was matched 38 characters into
        # the prose line "owner/s and in possession of PLOT NO.55, LAND AREA
        # MEAS. 57 Song", everything after it was taken as the value, and
        # "5505.570" - which does not match the khasra pattern at all - was
        # stored as the khasra number. The validator then flagged it as merely
        # low-confidence, so a reviewer saw a plausible-looking parcel id
        # rather than a blank.
        #
        # This module's own comment on the khasra spec already states the
        # principle: "a wrong khata/mutation identifier is a property-dispute
        # risk, not a cosmetic miss". Returning nothing leaves the field
        # `missing`, which is honest and reviewable; returning a wrong parcel
        # number is neither.
        #
        # Only STRICT_IDENTIFIERS are refused this way. Free-form numeric
        # fields keep the old tolerance, because there a mangled reading is
        # still better than no reading.
        if spec.key in STRICT_IDENTIFIERS:
            return None, 0.05, {}, notes + [
                f"Rejected: '{cleaned}' does not have the shape of a "
                f"{spec.display} and an identifier read wrongly is worse "
                f"than one left blank."]

        notes.append("Value does not match the expected format for this field.")
        return cleaned, 0.45, {}, notes

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
        # The LONGEST matching term wins, not the first one found. Several
        # classification terms contain a shorter term of the opposite meaning:
        # 'असिंचित' (unirrigated) contains 'सिंचित' (irrigated), and
        # 'unirrigated' contains 'irrigated'. First-match-wins iterated the
        # table in insertion order and so read every unirrigated parcel in the
        # corpus as irrigated - a silent inversion of a field that drives land
        # valuation, revenue assessment and compensation. Preferring the
        # longest match makes the more specific term win in every such pair.
        best_canon, best_len = None, 0
        for canon, words in LAND_CLASSES.items():
            for w in words:
                wl = w.lower()
                if wl in low and len(wl) > best_len:
                    best_canon, best_len = canon, len(wl)
        if best_canon:
            return best_canon, 0.93, {"raw_class": raw}, notes
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
        if not ok and spec.key in STRICT_IDENTIFIERS:
            # ULPIN reaches this branch, not the numeric one, because its kind
            # is "text" - so listing it in STRICT_IDENTIFIERS alone left it
            # un-enforced and a malformed Bhu-Aadhaar still passed through at
            # 0.45. A 14-character national parcel id read wrongly points at
            # another parcel just as surely as a wrong khasra does.
            return None, 0.05, {}, [
                f"Rejected: '{v}' does not have the shape of a "
                f"{spec.display} and an identifier read wrongly is worse "
                f"than one left blank."]
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
    script: str = "unknown"       # dominant script of the source line - see detect_script()
    notes: List[str] = dc_field(default_factory=list)
    extra: dict = dc_field(default_factory=dict)
    status: str = "extracted"     # extracted | missing | needs_review

    def to_dict(self) -> dict:
        # `script` rides inside `extra` rather than as a top-level key: the
        # DB layer (db.py insert_field) maps this dict onto a fixed set of
        # SQL columns and silently drops anything outside them, whereas
        # `extra` -> extra_json is already the established, round-tripped
        # channel for this kind of per-field metadata (area/date details,
        # raw classification text use the same channel).
        extra = dict(self.extra)
        extra["script"] = self.script
        return {
            "key": self.key, "display": self.display, "value": self.value,
            "raw_text": self.raw_text, "confidence": round(self.confidence, 4),
            "confidence_breakdown": {
                "label": round(self.label_confidence, 3),
                "pattern": round(self.pattern_confidence, 3),
                "ocr": round(self.ocr_confidence, 3),
            },
            "page": self.page, "bbox": list(self.bbox),
            "source_line": self.source_line,
            "notes": self.notes, "extra": extra, "status": self.status,
        }


# Weights for the composite confidence score.
W_LABEL, W_PATTERN, W_OCR = 0.40, 0.38, 0.22


# Below this composite confidence a field is marked needs_review instead of
# extracted. Chosen from measurement, not taste: tools/measure_accuracy.py
# --scans scores the corpus (including the degraded scans) and reports how
# many wrong values each threshold would wave through -
#
#     0.80 -> 155 auto-accepted,  5 wrong  |  11 sent to review
#     0.85 -> 152 auto-accepted,  3 wrong  |  14 sent to review
#     0.90 -> 146 auto-accepted,  1 wrong  |  20 sent to review
#     0.95 ->  42 auto-accepted,  0 wrong  | 124 sent to review
#
# 0.90 removes four of the five silently-accepted errors for nine extra field
# reviews out of 166. 0.95 buys the last error at the cost of sending 75% of
# all fields to a human, which would make the system pointless. Re-run the
# harness after changing extraction and move this number if the data moves.
REVIEW_THRESHOLD = 0.90


def extract_fields(lines: List, review_threshold: float = REVIEW_THRESHOLD) -> List[ExtractedField]:
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
        # Set when a candidate was found but every one of them demonstrably
        # belonged to a different field (see the rival check below).
        rival_only = False
        from_prose = False

        # TWO PASSES, and the second one matters as much as the first.
        #
        # Pass 1 applies the prose discount, so a label buried in a sentence
        # cannot beat a real label-value row. That is what stopped a court
        # order's narrative supplying the khasra number.
        #
        # Pass 2 drops the discount and runs only for fields pass 1 left
        # EMPTY. Without it the discount is not a preference but a ban, and
        # that is wrong for a whole class of document: a deed or an attorney
        # grant describes its property in a sentence, so prose is the only
        # place its area and plot number exist. Measured on the real Delhi
        # GPA, "LAND AREA MEAS. 57 SQ.YDS." lives inside
        # "owner/s and in possession of PLOT NO.55, ..." and nowhere else, so
        # a ban loses it entirely.
        #
        # Anything recovered on pass 2 is capped at PROSE_CONFIDENCE_CAP, so
        # it always routes to a human and sorts to the top of the review
        # queue rather than being trusted like a tabular read.
        for _pass, allow_prose in enumerate((False, True)):
          if best is not None:
            break
          for i, row in enumerate(norm):
            label_conf, matched, end_idx = _best_label_hit(
                row["text"], spec, allow_prose=allow_prose)
            if label_conf < 0.55 or matched is None:
                continue
            from_prose = allow_prose

            candidate = _candidate_after_label(
                row["text"], end_idx,
                _next_label_start(row["text"], end_idx, spec))
            carried_idx = None
            # Label alone on its line -> the value sits on a following line.
            # Printed forms frequently strand the ":" separator on its own
            # line, so lines carrying no payload are skipped rather than
            # accepted as the value.
            #
            # The emptiness test is "nothing left", not "shorter than two
            # characters". Plenty of real values are a single character - an
            # old khasra number like "5", a share of "1" - and treating those
            # as blank made the extractor walk past the printed value and
            # carry an unrelated number up from the line below it. Separator
            # leftovers are already removed by _candidate_after_label, so an
            # empty string is a sound test for a value-less row.
            if not candidate:
                for j in range(i + 1, min(i + 4, len(norm))):
                    nxt = _candidate_after_label(norm[j]["text"], 0)
                    if nxt and nxt.strip(" :;-.।|_"):
                        candidate, carried_idx = nxt, j
                        break
            carried = carried_idx is not None

            value, pat_conf, extra, notes = _validate_value(spec, candidate)

            # An area whose UNIT landed on the next line.
            #
            # A deed wraps its property description mid-phrase, so the number
            # and its unit are split across two OCR lines. Measured on a real
            # Delhi GPA: "LAND AREA MEAS. 57" ends one line and "SQ.YDS.,
            # (17.3'X ...)" begins the next, so the area parsed as a bare 57
            # with unit_missing - which the validator then flagged as an error
            # on a document that states its area perfectly clearly.
            #
            # Only ever ADDS a unit to a number that has none. It cannot
            # change a value or override a unit already read, so the worst it
            # can do is leave the field exactly as it was.
            # The parse rides under an "area" key inside extra, not at the top
            # level - reading it one level too shallow made this whole recovery
            # silently never fire.
            if (spec.kind == "area" and value and isinstance(extra, dict)
                    and (extra.get("area") or {}).get("unit_missing")):
                for j in range(i + 1, min(i + 3, len(norm))):
                    unit_head = _leading_area_unit(norm[j]["text"])
                    if not unit_head:
                        continue
                    merged = f"{candidate} {unit_head}"
                    m_val, m_conf, m_extra, m_notes = _validate_value(spec, merged)
                    if m_val and not (m_extra.get("area") or {}).get("unit_missing", True):
                        value, pat_conf, extra = m_val, m_conf, m_extra
                        notes = m_notes + [
                            f"Unit '{unit_head}' read from the following line."]
                    break

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
            # The rival must compete for the SAME value, which means its
            # label has to OVERLAP this one - not merely appear somewhere on
            # the line. Testing mere presence was wrong on real government
            # output, where one line carries several independent fields:
            #
            #     Khata No: 00620  PlotNo:50  Area: 0.3120 Hectare
            #
            # There "khata" (5 chars) is longer than "area" (4), sits 30
            # characters away, and was silently discarding the area value.
            # Measured over 15 real Bhu-Naksha plot reports, that cost every
            # single area: 0 of 15 extracted.
            _own_start = max(0, end_idx - len(matched))
            _rival = False
            for lbl, key in _ALL_LABELS:
                if key == spec.key or len(lbl) <= len(matched):
                    continue
                _at = _low_line.find(lbl)
                if _at < 0:
                    continue
                if _at < end_idx and _own_start < _at + len(lbl):
                    _rival = True
                    break

            # A demoted rival must never become the answer. Demoting it to
            # 0.35 still let it win whenever it was the ONLY candidate, which
            # is precisely the case where it is guaranteed wrong: a 7/12
            # extract that records a mutation date and no registration date
            # was reporting the mutation date as the registration date, and a
            # jamabandi with no registration number was reporting a date as
            # one. The line belongs to another field, so for this field the
            # document simply has no value - and "missing" sends it to a
            # reviewer honestly, where a wrong date silently passes as data.
            if _rival:
                rival_only = True
                continue

            score = W_LABEL * label_conf + W_PATTERN * pat_conf + W_OCR * ocr_conf
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
                source_line=row["text"], script=detect_script(row["text"]),
                notes=notes, extra=extra,
            )
            if best is None or cand.confidence > best.confidence:
                best = cand

        if best is None or best.value is None:
            if rival_only:
                note = ("The only match for this field sat on a row labelled "
                        "for a different field, so no value was taken from it.")
            elif spec.required:
                note = "Field not found in the document."
            else:
                note = "Field not present."
            results.append(ExtractedField(
                key=spec.key, display=spec.display, value=None,
                confidence=0.0, status="missing", notes=[note],
            ))
            continue

        if best is not None and from_prose:
            best.confidence = round(min(best.confidence, PROSE_CONFIDENCE_CAP), 4)
            best.notes = list(best.notes) + [
                "Read from prose rather than a label-value row - no structured "
                "occurrence of this field was found anywhere in the document."]
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
