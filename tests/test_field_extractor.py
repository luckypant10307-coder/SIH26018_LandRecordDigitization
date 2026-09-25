#!/usr/bin/env python3
"""
Unit tests for backend/field_extractor.py's multilingual support: Unicode
script detection and per-field label lexicons across Hindi/Marathi
(Devanagari), Bengali, Gurmukhi (Punjabi), Gujarati, Tamil, Telugu and
Kannada.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Run from anywhere with:
    python3 tests/test_field_extractor.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import dataclass
from typing import Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import field_extractor as fe  # noqa: E402


@dataclass
class FakeLine:
    """Minimal stand-in for ocr_engine.Line - extract_fields only reads these."""
    text: str
    confidence: float = 0.95
    page: int = 1
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)


def extract(*lines_text: str):
    fields = fe.extract_fields([FakeLine(t) for t in lines_text])
    return {f.key: f for f in fields}


class ScriptDetectionTests(unittest.TestCase):
    def test_devanagari(self):
        self.assertEqual(fe.detect_script("रामप्रसाद वर्मा"), "devanagari")

    def test_bengali(self):
        self.assertEqual(fe.detect_script("রামপ্রসাদ বর্মা"), "bengali")

    def test_gurmukhi(self):
        self.assertEqual(fe.detect_script("ਰਾਮ ਪ੍ਰਸਾਦ"), "gurmukhi")

    def test_gujarati(self):
        self.assertEqual(fe.detect_script("રામપ્રસાદ વર્મા"), "gujarati")

    def test_tamil(self):
        self.assertEqual(fe.detect_script("இராமபிரசாத் வர்மா"), "tamil")

    def test_telugu(self):
        self.assertEqual(fe.detect_script("రామ్ ప్రసాద్ వర్మ"), "telugu")

    def test_kannada(self):
        self.assertEqual(fe.detect_script("ರಾಮ್ ಪ್ರಸಾದ್ ವರ್ಮಾ"), "kannada")

    def test_latin(self):
        self.assertEqual(fe.detect_script("Ram Prasad Verma"), "latin")

    def test_mixed_bilingual_row(self):
        self.assertEqual(fe.detect_script("खाता संख्या / Khata Number : 1428"), "mixed")

    def test_unknown_for_no_letters(self):
        self.assertEqual(fe.detect_script("123 / - :"), "unknown")


class MultilingualExtractionTests(unittest.TestCase):
    """
    One line per language, each a realistic 'label : value' row, proving the
    new label lexicons actually drive extraction end to end - not just that
    the strings exist in a list.
    """

    def test_bengali_owner_and_village(self):
        fields = extract(
            "মালিকের নাম : সুনীল রায়",
            "গ্রাম : বরুণপুর",
        )
        self.assertEqual(fields["owner_name"].value, "সুনীল রায়")
        self.assertEqual(fields["owner_name"].script, "bengali")
        self.assertEqual(fields["village"].value, "বরুণপুর")

    def test_gurmukhi_owner_and_tehsil(self):
        fields = extract(
            "ਮਾਲਕ ਦਾ ਨਾਮ : ਹਰਪ੍ਰੀਤ ਸਿੰਘ",
            "ਤਹਿਸੀਲ : ਲੁਧਿਆਣਾ",
        )
        self.assertEqual(fields["owner_name"].value, "ਹਰਪ੍ਰੀਤ ਸਿੰਘ")
        self.assertEqual(fields["tehsil"].value, "ਲੁਧਿਆਣਾ")

    def test_gujarati_owner_and_district(self):
        fields = extract(
            "માલિકનું નામ : કિરણ પટેલ",
            "જિલ્લો : અમદાવાદ",
        )
        self.assertEqual(fields["owner_name"].value, "કિરણ પટેલ")
        self.assertEqual(fields["district"].value, "અમદાવાદ")

    def test_tamil_owner_and_survey_number(self):
        fields = extract(
            "உரிமையாளர் பெயர் : முருகன்",
            "சர்வே எண் : 142/2",
        )
        self.assertEqual(fields["owner_name"].value, "முருகன்")
        self.assertEqual(fields["survey_number"].value, "142/2")

    def test_telugu_owner_and_area(self):
        fields = extract(
            "యజమాని పేరు : వెంకటేశ్",
            "విస్తీర్ణం : 2 ఎకరం",
        )
        self.assertEqual(fields["owner_name"].value, "వెంకటేశ్")
        area = fields["area"]
        self.assertIsNotNone(area.value)
        self.assertAlmostEqual(area.extra["area"]["sqm"], 2 * 4046.86, places=1)

    def test_kannada_owner_and_land_classification(self):
        fields = extract(
            "ಮಾಲೀಕರ ಹೆಸರು : ಸುರೇಶ್",
            "ಭೂಮಿಯ ಪ್ರಕಾರ : ಕೃಷಿ",
        )
        self.assertEqual(fields["owner_name"].value, "ಸುರೇಶ್")
        self.assertEqual(fields["land_classification"].value, "agricultural")

    def test_kannada_guntha_area_unit(self):
        parsed = fe.parse_area("5 ಗುಂಠೆ")
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(parsed["sqm"], 5 * 101.17, places=1)
        self.assertTrue(parsed["regional_unit"])

    def test_tamil_cent_area_unit(self):
        parsed = fe.parse_area("10 சென்ட்")
        self.assertIsNotNone(parsed)
        self.assertAlmostEqual(parsed["sqm"], 10 * 40.47, places=1)

    def test_bengali_dag_number_maps_to_khasra(self):
        fields = extract("দাগ নম্বর : 512/1")
        self.assertEqual(fields["khasra_number"].value, "512/1")

    def test_punjabi_khasra_label(self):
        fields = extract("ਖਸਰਾ ਨੰਬਰ : 88")
        self.assertEqual(fields["khasra_number"].value, "88")

    def test_script_survives_the_persisted_dict_shape(self):
        # script rides inside extra_json (see ExtractedField.to_dict), the
        # channel db.py's insert_field actually persists - a top-level key
        # would be silently dropped by its fixed-column INSERT.
        fields = extract("গ্রাম : বরুণপুর")
        d = fields["village"].to_dict()
        self.assertNotIn("script", d)
        self.assertEqual(d["extra"]["script"], "bengali")



def _line(text, y=0):
    class L:
        def __init__(self):
            self.text = text
            self.confidence = 0.9
            self.page = 1
            self.bbox = (0, y, 900, y + 18)
    return L()


class StrictIdentifierTests(unittest.TestCase):
    """
    A shape violation on a parcel identifier means "not this field".

    Accepting it at 0.45 was how garbage reached records. Measured on a real
    Delhi GPA: the label "PLOT NO" matched 38 characters into the prose line
    "owner/s and in possession of PLOT NO.55, LAND AREA MEAS. 57 Song",
    everything after it became the value, and "5505.570" was stored as the
    khasra number - flagged merely low-confidence, so a reviewer saw a
    plausible parcel id rather than a blank.
    """

    def test_a_malformed_khasra_is_refused_not_downgraded(self):
        spec = fe.FIELD_BY_KEY["khasra_number"]
        value, conf, _extra, _notes = fe._validate_value(spec, "5505.570")
        self.assertIsNone(value)
        self.assertLess(conf, 0.1)

    def test_a_well_formed_khasra_still_passes(self):
        spec = fe.FIELD_BY_KEY["khasra_number"]
        for good in ("214/248", "55", "237/4"):
            value, conf, _e, _n = fe._validate_value(spec, good)
            self.assertEqual(value, good)
            self.assertGreater(conf, 0.9)

    def test_every_strict_identifier_refuses_a_bad_shape(self):
        for key in fe.STRICT_IDENTIFIERS:
            spec = fe.FIELD_BY_KEY[key]
            value, _c, _e, _n = fe._validate_value(spec, "0.3120")
            self.assertIsNone(value, key)

    def test_a_non_identifier_keeps_the_old_tolerance(self):
        """
        Only the fields downstream records KEY on are refused. Elsewhere a
        mangled reading still beats no reading, because nothing points at
        another person's land because of it.
        """
        spec = fe.FIELD_BY_KEY["mutation_number"]
        value, conf, _e, _n = fe._validate_value(spec, "4AF4")
        self.assertIsNotNone(value)


class ProseVersusRowTests(unittest.TestCase):
    """
    Deeds and court orders recite schema vocabulary in flowing prose.

    Label DEPTH cannot separate them, which is the subtle part: a real
    tabular row carries labels deep into the line -
    "khata no: 00620 plotno:50 area: 0.3120 hectare" has its area label at
    character 26 of 46 and is perfectly good. What distinguishes prose is
    that it is glued together by connectives.
    """

    PROSE = "| owner/s and in possession of plot no.55, land area meas. 57 song"
    ROW = "khata no: 00620 plotno:50 area: 0.3120 hectare"
    HINDI_PROSE = "आदेश दिनांक के अनुसार गाटा संख्या 484 रकवा का"
    HINDI_ROW = "खसरा नंबर : 50"

    def test_a_label_buried_in_prose_is_discounted(self):
        conf, _m, _e = fe._best_label_hit(self.PROSE,
                                          fe.FIELD_BY_KEY["khasra_number"])
        self.assertLess(conf, 0.55)

    def test_a_tabular_row_is_not_discounted(self):
        conf, _m, _e = fe._best_label_hit(self.ROW,
                                          fe.FIELD_BY_KEY["khata_number"])
        self.assertGreater(conf, 0.9)

    def test_a_label_deep_inside_a_real_row_survives(self):
        """The case a depth-based rule would have broken."""
        conf, _m, _e = fe._best_label_hit(self.ROW, fe.FIELD_BY_KEY["area"])
        self.assertGreater(conf, 0.9)

    def test_hindi_order_narrative_is_discounted_too(self):
        conf, _m, _e = fe._best_label_hit(self.HINDI_PROSE,
                                          fe.FIELD_BY_KEY["khasra_number"])
        self.assertLess(conf, 0.55)

    def test_a_devanagari_row_is_not_discounted(self):
        conf, _m, _e = fe._best_label_hit(self.HINDI_ROW,
                                          fe.FIELD_BY_KEY["khasra_number"])
        self.assertGreater(conf, 0.9)

    def test_a_structured_row_beats_prose_for_the_same_field(self):
        """Both present: the row must win."""
        fields = {f.key: f for f in fe.extract_fields(
            [_line(self.PROSE, 0), _line("khasra no: 237/4", 20)])}
        self.assertEqual(fields["khasra_number"].value, "237/4")

    def test_prose_is_used_when_nothing_structured_exists(self):
        """
        The discount is a PREFERENCE, not a ban. A deed describes its
        property in a sentence, so prose is the only place its area lives -
        banning it outright loses the value entirely.
        """
        fields = {f.key: f for f in fe.extract_fields(
            [_line("owner/s and in possession of land area meas. 57 sq.yds")])}
        self.assertIsNotNone(fields["area"].value)

    def test_a_prose_sourced_value_is_capped_and_labelled(self):
        fields = {f.key: f for f in fe.extract_fields(
            [_line("owner/s and in possession of land area meas. 57 sq.yds")])}
        area = fields["area"]
        self.assertLessEqual(area.confidence, fe.PROSE_CONFIDENCE_CAP)
        self.assertEqual(area.status, "needs_review")
        self.assertTrue(any("prose" in n for n in area.notes))


class AreaUnitTests(unittest.TestCase):
    """Square yards, and units separated from their number by OCR noise."""

    def test_square_yards_and_gaj_are_understood(self):
        """
        Gaj/square yards is THE unit for urban plots across North India, and
        it was missing - so every city property document reported
        AREA_UNIT_MISSING.
        """
        for probe in ("57 sq.yds.", "57 SQ.YDS", "57 square yards",
                      "57 gaj", "57 गज"):
            parsed = fe.parse_area(probe)
            self.assertEqual(parsed["unit"], "sqyd", probe)
            self.assertAlmostEqual(parsed["sqm"], 47.659, places=2)

    def test_a_square_yard_is_not_marked_regional(self):
        """Unlike bigha, a square yard does not vary by district, so no
        conversion caveat is warranted."""
        self.assertFalse(fe.parse_area("57 sq.yds")["regional_unit"])

    def test_one_ocr_artefact_between_number_and_unit_is_tolerated(self):
        """Measured: "LAND AREA MEAS. 57 SQ.YDS." read as "MEAS. 57 ong sq.yds"."""
        parsed = fe.parse_area("MEAS. 57 ong sq.yds")
        self.assertEqual(parsed["unit"], "sqyd")
        self.assertAlmostEqual(parsed["sqm"], 47.659, places=2)

    def test_the_tolerance_does_not_bridge_a_second_number(self):
        """"17.3 x 3 sq.yds" must attach the unit to the 3, not the 17.3."""
        self.assertEqual(fe.parse_area("17.3 x 3 sq.yds")["value"], 3.0)

    def test_the_tolerance_does_not_bridge_two_junk_tokens(self):
        self.assertIsNone(fe.parse_area("57 blah blah sq.yds")["unit"])

    def test_a_clean_parse_is_unaffected_by_the_fallback(self):
        """The fallback runs ONLY when the strict pass found nothing, so a
        document that already parses cannot be changed by it."""
        for probe, unit in (("1.4520 hectare", "hectare"), ("5 bigha", "bigha"),
                            ("57 sq.yds", "sqyd")):
            self.assertEqual(fe.parse_area(probe)["unit"], unit, probe)

    def test_a_composite_area_still_parses(self):
        parsed = fe.parse_area("2 बीघा 10 बिस्वा")
        self.assertEqual(parsed["unit"], "composite")
        self.assertGreater(parsed["sqm"], 0)

    def test_a_bare_number_still_reports_the_unit_as_missing(self):
        parsed = fe.parse_area("MEAS. 57")
        self.assertIsNone(parsed["unit"])
        self.assertTrue(parsed["unit_missing"])


if __name__ == "__main__":
    unittest.main()
