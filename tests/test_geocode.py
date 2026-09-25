#!/usr/bin/env python3
"""
Unit tests for backend/geocode.py (approximate geotagging from place names).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Pure lookup over bundled reference data, no optional dependencies, so every
test here always runs.

The tests are weighted towards two failure modes, because those are the ones
that actually cost something:

  1. A place that should resolve but does not. The real Delhi general power
     of attorney is the regression case - its village field reads
     "VILLAGE NARELA, SABOLI ROAD, DELHI" and the first implementation
     returned nothing at all for it, because gazetteer.normalise() does not
     case-fold and the all-caps token never met the table entry "Narela".

  2. A confident coordinate on the WRONG place. That is strictly worse than
     no coordinate, because a blank is visibly missing while a wrong pin
     looks like an answer. Hence the tests that a short or unknown token
     resolves to nothing, and that precision is always reported.

Run from anywhere with:
    python3 tests/test_geocode.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import geocode as gc  # noqa: E402


def _v(**kw):
    """Build an extractor-shaped field map."""
    return {k: {"value": v} for k, v in kw.items()}


class ReferenceDataTests(unittest.TestCase):
    """The bundled table has to actually load, or everything below is vacuous."""

    def test_the_index_loaded(self):
        self.assertTrue(gc.INDEX.loaded)

    def test_every_level_has_entries(self):
        counts = gc.describe()["counts"]
        for level in ("state", "district", "locality"):
            self.assertGreater(counts[level], 0, level)

    def test_all_states_and_union_territories_are_present(self):
        """A missing state means silent no-match for every record in it."""
        self.assertGreaterEqual(gc.describe()["counts"]["state"], 36)

    def test_delhi_is_present(self):
        """
        Delhi was the specific gap: admin_master.json covers 8 states and
        Delhi is not among them, so the real GPA could not resolve even to
        a state before this table existed.
        """
        match = gc.resolve(_v(state="Delhi"))
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "Delhi")

    def test_accuracy_is_declared_for_every_level(self):
        for level in ("state", "district", "locality"):
            self.assertGreater(gc.INDEX.accuracy[level], 0, level)

    def test_coordinates_are_inside_india(self):
        """A sign error or a transposed pair would put a record in the sea."""
        for level in gc.INDEX.by_level.values():
            for recs in level.values():
                for rec in recs:
                    self.assertTrue(6.0 <= rec["lat"] <= 37.5, rec)
                    self.assertTrue(68.0 <= rec["lon"] <= 97.5, rec)


class RealDocumentTests(unittest.TestCase):
    """The measured regression cases from the real Delhi GPA."""

    GPA_VILLAGE = "VILLAGE NARELA, SABOLI ROAD, DELHI"

    def test_the_real_gpa_village_field_resolves(self):
        match = gc.resolve(_v(village=self.GPA_VILLAGE))
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "Narela")

    def test_it_resolves_to_the_locality_not_the_state(self):
        """
        Both "NARELA" and "DELHI" are in that one string. The specific one
        has to win, or the coordinate is 300 km coarse instead of 5 km.
        """
        match = gc.resolve(_v(village=self.GPA_VILLAGE))
        self.assertEqual(match.level, "locality")
        self.assertEqual(match.accuracy_m, gc.INDEX.accuracy["locality"])

    def test_the_address_furniture_does_not_defeat_it(self):
        """
        "VILLAGE" and "ROAD" are structural words, not place names. Left in,
        the segment "VILLAGE NARELA" matches nothing.
        """
        for probe in ("VILLAGE NARELA", "VILL. BAWANA, DELHI",
                      "GRAM NARELA", "Narela Road, Delhi"):
            self.assertIsNotNone(gc.resolve(_v(village=probe)), probe)

    def test_an_all_caps_name_matches_a_title_case_table(self):
        """The exact bug that made the real GPA resolve to nothing."""
        for probe in ("NARELA", "Narela", "narela", "nArElA"):
            match = gc.resolve(_v(village=probe))
            self.assertIsNotNone(match, probe)
            self.assertEqual(match.name, "Narela", probe)

    def test_the_match_is_marked_exact(self):
        """An exact hit must not be reported as an inference."""
        self.assertTrue(gc.resolve(_v(village="NARELA")).exact)


class DevanagariTests(unittest.TestCase):
    """
    A Hindi record has to reach a Latin table.

    The romaniser doubles long vowels - नरेला becomes "narelaa" and भोपाल
    "bhopaala" - and phonetic_key alone bridged only 3 of 8 measured pairs,
    with दिल्ली -> "dillii" and लखनऊ -> "lakhanou" never reaching Delhi or
    Lucknow by any phonetic route. Explicit Devanagari aliases in the
    reference data are what carry those, so these tests guard the data as
    much as the code.
    """

    def test_a_devanagari_locality_resolves(self):
        match = gc.resolve(_v(village="ग्राम बवाना"))
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "Bawana")

    def test_a_devanagari_district_resolves(self):
        match = gc.resolve(_v(district="मुजफ्फरपुर"))
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "Muzaffarpur")

    def test_the_names_no_phonetic_key_can_bridge(self):
        """दिल्ली/Delhi and लखनऊ/Lucknow, carried by alias not by phonetics."""
        self.assertEqual(gc.resolve(_v(state="दिल्ली")).name, "Delhi")
        self.assertEqual(gc.resolve(_v(district="लखनऊ")).name, "Lucknow")

    def test_a_devanagari_state_resolves(self):
        for probe, expect in (("उत्तर प्रदेश", "Uttar Pradesh"),
                              ("मध्य प्रदेश", "Madhya Pradesh"),
                              ("राजस्थान", "Rajasthan")):
            match = gc.resolve(_v(state=probe))
            self.assertIsNotNone(match, probe)
            self.assertEqual(match.name, expect, probe)

    def test_devanagari_address_furniture_is_stripped(self):
        match = gc.resolve(_v(village="गांव नरेला, दिल्ली"))
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "Narela")


class HistoricalNameTests(unittest.TestCase):
    """
    Land records are written across decades and use whichever name was
    official at the time, so a 1990s deed says Allahabad and Bombay where a
    current table says Prayagraj and Mumbai. Refusing the old name would
    fail exactly the old documents that most need digitising.
    """

    def test_former_english_names_resolve(self):
        for probe, expect in (("Allahabad", "Prayagraj"),
                              ("Bombay", "Mumbai City"),
                              ("Bangalore", "Bengaluru Urban"),
                              ("Mysore", "Mysuru"),
                              ("Gulbarga", "Kalaburagi")):
            match = gc.resolve(_v(district=probe))
            self.assertIsNotNone(match, probe)
            self.assertEqual(match.name, expect, probe)


class RefusalTests(unittest.TestCase):
    """
    Where the module must return nothing.

    A wrong pin is worse than a blank, so these matter more than the
    positive cases.
    """

    def test_an_unknown_place_resolves_to_nothing(self):
        for probe in ("nowhere at all", "Zzzyx", "Atlantis",
                      "qwertyuiop", "Springfield"):
            self.assertIsNone(gc.resolve(_v(village=probe)), probe)

    def test_an_empty_or_missing_field_resolves_to_nothing(self):
        self.assertIsNone(gc.resolve(_v(village="")))
        self.assertIsNone(gc.resolve(_v(owner_name="SOMEONE")))
        self.assertIsNone(gc.resolve({}))

    def test_pure_address_furniture_resolves_to_nothing(self):
        """Strip the structural words and nothing is left to match."""
        for probe in ("VILLAGE", "TEHSIL DISTRICT", "ROAD, COLONY, SECTOR"):
            self.assertIsNone(gc.resolve(_v(village=probe)), probe)

    def test_a_non_string_value_is_survived(self):
        """Numeric or None values reach here from the extractor."""
        for bad in (None, 57, 3.5, [], {}):
            self.assertIsNone(gc.resolve({"village": {"value": bad}}), repr(bad))

    def test_a_short_token_is_not_fuzzy_matched(self):
        """
        Fuzzy matching a 3-letter token against 150 place names finds
        something every time, and it is never right.
        """
        for probe in ("Goa1", "Kot", "Pal", "Del"):
            match = gc.resolve(_v(village=probe))
            if match is not None:
                self.assertTrue(match.exact, probe)

    def test_missing_reference_data_degrades_quietly(self):
        """
        Approximate geotagging is an enrichment, so a missing or corrupt
        table must mean "no coordinate", never an exception that takes the
        upload path down with it.
        """
        original = gc._DATA
        try:
            gc._DATA = os.path.join(ROOT, "no_such_file_at_all.json")
            index = gc.PlaceIndex()
            self.assertFalse(index.loaded)
            self.assertEqual(index.lookup("Narela", "locality"), ([], False))
        finally:
            gc._DATA = original


class SpecificityTests(unittest.TestCase):
    """The most specific level named must win, and be labelled as such."""

    def test_a_locality_beats_a_district_and_a_state(self):
        match = gc.resolve(_v(village="Narela", district="North West Delhi",
                              state="Delhi"))
        self.assertEqual(match.level, "locality")
        self.assertEqual(match.name, "Narela")

    def test_a_district_beats_a_state(self):
        match = gc.resolve(_v(district="Bhopal", state="Madhya Pradesh"))
        self.assertEqual(match.level, "district")
        self.assertEqual(match.name, "Bhopal")

    def test_a_state_alone_still_resolves(self):
        match = gc.resolve(_v(state="Karnataka"))
        self.assertEqual(match.level, "state")
        self.assertEqual(match.accuracy_m, gc.INDEX.accuracy["state"])

    def test_a_multi_word_name_is_not_shadowed_by_its_last_word(self):
        """
        "New Delhi" must not be beaten by the bare "Delhi" inside it, which
        is why whole comma segments and longer n-grams are tried first.
        """
        match = gc.resolve(_v(district="New Delhi"))
        self.assertEqual(match.name, "New Delhi")

    def test_corroboration_is_reported_when_levels_agree(self):
        match = gc.resolve(_v(village="Narela", district="North West Delhi",
                              state="Delhi"))
        self.assertTrue(match.corroborated)

    def test_corroboration_is_not_claimed_from_a_single_name(self):
        self.assertFalse(gc.resolve(_v(village="Narela")).corroborated)


class ConflictTests(unittest.TestCase):
    """
    A record naming Narela and Kerala has something wrong with it.

    The stated state is the more reliable field so it wins the coordinate,
    but the disagreement is surfaced rather than hidden - silently emitting
    a confident Kerala pin would be the worst available outcome.
    """

    def test_a_contradicting_locality_is_reported(self):
        match = gc.resolve(_v(village="Narela", state="Kerala"))
        self.assertIsNotNone(match)
        self.assertTrue(match.conflicts)
        self.assertIn("Narela", match.conflicts[0])

    def test_the_stated_state_wins_the_coordinate(self):
        match = gc.resolve(_v(village="Narela", state="Kerala"))
        self.assertEqual(match.name, "Kerala")

    def test_an_agreeing_record_reports_no_conflict(self):
        match = gc.resolve(_v(village="Narela", state="Delhi"))
        self.assertEqual(match.conflicts, ())

    def test_conflicts_are_json_serialisable(self):
        """to_dict() crosses the API boundary, so no tuples may leak."""
        import json
        match = gc.resolve(_v(village="Narela", state="Kerala"))
        json.dumps(match.to_dict())


class PrecisionHonestyTests(unittest.TestCase):
    """
    The whole module is only defensible if it never lets an administrative
    coordinate pass for a survey one.
    """

    def test_every_match_declares_its_accuracy(self):
        for values in (_v(village="Narela"), _v(district="Bhopal"),
                       _v(state="Kerala")):
            match = gc.resolve(values)
            self.assertGreaterEqual(match.accuracy_m, 5000)

    def test_no_match_is_ever_parcel_precision(self):
        """
        A name can never yield metre-grade accuracy. If this ever fails,
        something has started claiming a precision it cannot have.
        """
        for values in (_v(village="Narela"), _v(district="Bhopal"),
                       _v(state="Kerala"), _v(village="ग्राम बवाना")):
            self.assertGreaterEqual(gc.resolve(values).accuracy_m, 5000)

    def test_a_coarser_level_is_never_more_accurate(self):
        acc = gc.INDEX.accuracy
        self.assertLess(acc["locality"], acc["district"])
        self.assertLess(acc["district"], acc["state"])

    def test_the_source_field_it_came_from_is_recorded(self):
        """A reviewer must be able to see which field produced the pin."""
        match = gc.resolve(_v(village="Narela"))
        self.assertEqual(match.matched_from, "village")
        self.assertTrue(match.matched_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
