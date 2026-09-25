"""
What KIND of land document is this?
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Why this exists
---------------
The 17-field schema in field_extractor.py is a Record-of-Rights schema:
khasra, khata, area, tehsil, land classification. That is the right schema for
a khatauni or a Bhu-Naksha extract, and the wrong schema for every other
paper in a property file.

Measured on a real notarised Power of Attorney (Delhi e-Stamp, 13 pages): the
OCR read the printed values correctly - "Purchased by : PUSHPA" at 0.90,
"First Party : RAJBIR SINGH", "Stamp Duty Amount(Rs.) : 50",
"Description of Document : Article 48(c) Power of attorney - GPA" - and the
extractor returned essentially nothing, because **0 of that document's 12
printed field labels exist among the schema's 221 aliases**. It was looking
for a khasra number on a document that structurally has none.

That is a document-type coverage gap, not an OCR failure, and it cannot be
fixed by better reading. It needs the right schema chosen first.

How detection works
-------------------
These documents announce themselves. A GPA prints "Power of attorney" in its
own Description of Document field; a sale deed says "sale deed" or
"बैनामा"; a Bhu-Naksha extract says "Plot Information" and "खसरा". So
detection is keyword scoring over the OCR text, not a classifier - there is
no training data for a classifier here, and a keyword that appears in the
document's own self-description is stronger evidence than anything a model
trained on fifteen samples would learn.

What this deliberately does NOT do
----------------------------------
It does not guess. A document matching nothing returns "unknown", and the
caller then applies the Record-of-Rights schema as before and reports what it
could not find. Silently applying a GPA schema to an unrecognised paper would
turn "we do not know what this is" into confident wrong output, which is the
failure mode this whole project is built to avoid.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# Document types, each with the evidence that identifies it.
#
# Weights are crude on purpose. A phrase from the document's own
# self-description ("power of attorney", "Plot Information") is worth far more
# than an incidental term ("stamp duty") that appears on nearly every
# instrument, and that ordering is the whole of the tuning here.
SIGNALS: Dict[str, List[Tuple[str, float]]] = {
    "record_of_rights": [
        ("plot information", 4.0), ("owner details", 3.0),
        ("khasra", 3.0), ("खसरा", 3.0), ("khatauni", 4.0), ("खतौनी", 4.0),
        ("record of rights", 4.0), ("भूमि का विवरण", 4.0),
        ("khata no", 2.0), ("खाता", 2.0), ("गाटा", 2.0),
        ("land record", 2.0), ("भू-अभिलेख", 3.0), ("अधिकार अभिलेख", 4.0),
        ("hectare", 1.0), ("हेक्टेयर", 1.0), ("bigha", 1.0),
    ],
    "power_of_attorney": [
        ("power of attorney", 5.0), ("attorney", 2.0), ("gpa", 3.0),
        ("मुख्तारनामा", 5.0), ("वसीयत", 1.0),
        ("first party", 2.0), ("second party", 2.0),
        ("article 48", 3.0), ("attorney holder", 3.0),
        ("is at liberty to sell", 2.0),
    ],
    "sale_deed": [
        ("sale deed", 5.0), ("बैनामा", 5.0), ("विक्रय विलेख", 5.0),
        ("conveyance deed", 4.0), ("vendor", 2.0), ("vendee", 2.0),
        ("purchaser", 1.5), ("consideration price", 2.0),
        ("sold and transferred", 2.0),
    ],
    "mutation_order": [
        ("mutation order", 5.0), ("नामांतरण", 4.0), ("दाखिल खारिज", 5.0),
        ("order description", 3.0), ("आदेश दिनांक", 2.0),
        ("तहसीलदार", 2.0), ("naib tehsildar", 2.0),
    ],
    "stamp_certificate": [
        ("india non judicial", 5.0), ("e-stamp", 4.0),
        ("certificate no", 2.0), ("stamp duty paid by", 3.0),
        ("unique doc. reference", 3.0), ("account reference", 2.0),
        ("shcilestamp", 3.0),
    ],
}

# A type must clear this to be claimed at all. Below it the answer is
# "unknown", which is a usable answer - the Record-of-Rights schema is then
# applied as the documented default and its misses are reported honestly.
MIN_SCORE = 4.0

# And it must beat the runner-up by this much. Land papers overlap heavily -
# a GPA executed on an e-Stamp carries BOTH sets of vocabulary, which is
# exactly the document that prompted this module - so a narrow win means
# "both apply", not "this one".
MIN_MARGIN = 2.0

# A stamp certificate is a CARRIER, not a kind of document.
#
# Indian instruments are executed ON e-Stamp paper, so the e-Stamp boilerplate
# ("India Non Judicial", "Certificate No.", "Stamp Duty Paid By") rides along
# with whatever the paper actually says. Measured on the real GPA: the carrier
# vocabulary scored 19.0 against the Power of Attorney's 17.0 and won - which
# would have applied a stamp-certificate schema to a document whose substance
# is an attorney grant.
#
# The document's own self-description is the better authority. This one prints
# "Description of Document : Article 48(c) Power of attorney - GPA" on its
# face. So when the carrier wins but a real instrument type also clears the
# bar, the instrument takes precedence and the carrier is reported separately
# as the medium.
CARRIER_TYPES = {"stamp_certificate"}

DISPLAY = {
    "record_of_rights": "Record of Rights / Khatauni",
    "power_of_attorney": "Power of Attorney (GPA)",
    "sale_deed": "Sale Deed",
    "mutation_order": "Mutation Order",
    "stamp_certificate": "Stamp Certificate",
    "unknown": "Unrecognised document type",
}

# Which of the 17 Record-of-Rights fields are even MEANINGFUL per type.
#
# This is what stops a GPA being blocked for missing a khasra number. The
# validator's required-field rule consults it, so "mandatory" becomes
# mandatory FOR THIS KIND OF DOCUMENT rather than mandatory in the abstract -
# the same reasoning that made khasra-or-survey an "at least one of" group
# for South Indian records.
APPLICABLE: Dict[str, Optional[set]] = {
    # None means "the full schema applies".
    "record_of_rights": None,
    # A GPA DOES cite the land record of the property it concerns - measured
    # on the real Delhi document: "out OF KHATONI NO.214/248" alongside
    # "PLOT NO.55" and "LAND AREA MEAS. 57 SQ.YDS.". Excluding the identifier
    # and area fields here was too aggressive; citing a khatauni by reference
    # is normal drafting, and a GPA that names one should have it captured
    # and shape-checked like any other.
    #
    # They are applicable, not required - the required flag still comes from
    # the FieldSpec, so a GPA that omits them is not blocked for it.
    "power_of_attorney": {"owner_name", "father_name", "village", "tehsil",
                          "district", "state", "registration_number",
                          "registration_date", "khasra_number",
                          "khata_number", "survey_number", "area", "share"},
    "sale_deed": {"owner_name", "father_name", "village", "tehsil", "district",
                  "state", "area", "khasra_number", "survey_number",
                  "registration_number", "registration_date", "share"},
    "mutation_order": {"khasra_number", "khata_number", "owner_name",
                       "father_name", "village", "tehsil", "district",
                       "mutation_number", "mutation_date", "area", "share"},
    "stamp_certificate": {"owner_name", "village", "district", "state",
                          "registration_number", "registration_date"},
    "unknown": None,
}


# APPLICABLE and REQUIRED are different questions, and conflating them broke
# the fix twice over.
#
# First pass excluded khasra/khata/area from a GPA entirely - which stopped it
# being blocked, but also meant the khatauni it DOES cite
# ("out OF KHATONI NO.214/248") could never be captured. Second pass added
# them back as applicable - which restored the citation but made them REQUIRED
# again, because the FieldSpec marks them mandatory and applicability was the
# only filter. The GPA went straight back to being blocked.
#
# So the two need separating. APPLICABLE answers "is this field worth reading
# on this kind of paper"; REQUIRED answers "must it be present for the record
# to be publishable". A GPA can cite a khatauni and is not defective for
# omitting one.
#
# None means "fall back to the FieldSpec's own required flag", which keeps
# Record-of-Rights behaviour exactly as documented.
REQUIRED_BY_TYPE: Dict[str, Optional[set]] = {
    "record_of_rights": None,
    # A GPA must identify who granted it and over what property. Everything
    # else it may cite or omit.
    "power_of_attorney": {"owner_name", "village"},
    "sale_deed": {"owner_name", "village", "area"},
    "mutation_order": {"owner_name", "village", "khasra_number"},
    "stamp_certificate": {"owner_name"},
    "unknown": None,
}


def required_fields(doc_type: str, spec_required) -> set:
    """
    Which fields are genuinely mandatory for this document type.

    `spec_required` is the set the FieldSpec declarations already mark
    required, used unchanged when a type has no override - so an
    unrecognised document is held to the documented default rather than
    quietly relaxed.
    """
    override = REQUIRED_BY_TYPE.get(doc_type, None)
    if override is None:
        return set(spec_required)
    return set(override)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).lower()


def score_all(text: str) -> Dict[str, float]:
    """Every type's score, so a caller can see the whole picture."""
    low = _normalise(text)
    out: Dict[str, float] = {}
    for name, signals in SIGNALS.items():
        total = 0.0
        for phrase, weight in signals:
            if phrase in low:
                total += weight
        out[name] = round(total, 2)
    return out


def detect(text_or_lines) -> dict:
    """
    Identify the document type from OCR text (a string, or objects with .text).

    Returns the verdict plus the evidence for it and the runner-up, because a
    type decision changes which fields are demanded of the document - and a
    reviewer told "this is a GPA" deserves to see why.
    """
    if isinstance(text_or_lines, str):
        text = text_or_lines
    else:
        text = "\n".join(getattr(l, "text", "") or ""
                         for l in (text_or_lines or []))

    scores = score_all(text)

    # Carriers are ranked SEPARATELY, never against instruments.
    #
    # Competing them failed twice on the same document: when the e-Stamp
    # boilerplate scored higher it won outright (19.0 vs 17.0), and once the
    # OCR improved and the two TIED at 17.0 the zero margin made the whole
    # thing "unknown". Both are wrong for one reason - the medium a document
    # is printed on is not a rival hypothesis about what the document IS, so
    # it has no business in the margin calculation.
    instruments = sorted(((n, sc) for n, sc in scores.items()
                          if n not in CARRIER_TYPES), key=lambda kv: -kv[1])
    carriers = sorted(((n, sc) for n, sc in scores.items()
                       if n in CARRIER_TYPES), key=lambda kv: -kv[1])

    best, best_score = instruments[0]
    runner, runner_score = (instruments[1] if len(instruments) > 1
                            else ("", 0.0))
    margin = round(best_score - runner_score, 2)

    carrier = carriers[0][0] if (carriers and carriers[0][1] >= MIN_SCORE) else None
    claimed = best_score >= MIN_SCORE and margin >= MIN_MARGIN
    verdict = best if claimed else "unknown"

    # A page that is ONLY the carrier - an e-Stamp cover sheet carrying no
    # instrument text yet - is worth naming as such rather than reported as
    # unrecognised.
    if not claimed and carrier:
        verdict, claimed, carrier = carrier, True, None

    low = _normalise(text)
    evidence = [phrase for phrase, _w in SIGNALS.get(verdict, []) if phrase in low][:6]

    return {
        "type": verdict,
        "display": DISPLAY.get(verdict, verdict),
        "score": best_score,
        "margin": margin,
        "runner_up": runner,
        "runner_up_score": runner_score,
        "scores": scores,
        "evidence": evidence,
        "claimed": claimed,
        # The paper it was executed on, when that differs from what it says.
        "carrier": carrier,
        "carrier_display": DISPLAY.get(carrier) if carrier else None,
        "reason": ("matched on " + ", ".join(evidence)) if claimed else
                  (f"best candidate '{best}' scored {best_score} with margin "
                   f"{margin}; needs {MIN_SCORE} and {MIN_MARGIN} - treating "
                   f"as unknown and applying the Record-of-Rights schema"),
    }


def applicable_fields(doc_type: str, all_keys) -> set:
    """
    Which field keys are meaningful for this document type.

    An unknown type gets the full schema rather than an empty one: reporting
    "khasra missing" on a paper we could not identify is honest, while
    reporting nothing at all would hide that the document was processed.
    """
    allowed = APPLICABLE.get(doc_type)
    if allowed is None:
        return set(all_keys)
    return {k for k in all_keys if k in allowed}
