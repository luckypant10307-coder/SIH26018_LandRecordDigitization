#!/usr/bin/env python3
"""
PostGIS spatial-store tests for backend/postgis.py.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

SKIPPED unless TEST_POSTGIS_URL points at a PostgreSQL server with the postgis
extension available, and which this suite may DROP AND RECREATE tables in. The
default has to be "skip" for the same reason test_postgres.py's does: the whole
suite must run on a laptop with nothing installed.

    docker run -d --name sihgis -e POSTGRES_PASSWORD=sih \\
        -e POSTGRES_DB=landrecords -p 55433:5432 postgis/postgis:16-3.4
    TEST_POSTGIS_URL=postgresql://postgres:sih@127.0.0.1:55433/landrecords \\
        python3 tests/test_postgis.py -v

WHAT IS TESTED, AND WHY THESE CASES

The interesting failures here are geometric, not SQL. Two dominate:

  * Adjacent parcels MUST NOT be reported as overlapping. Parcels in a village
    share edges by definition, so a naive ST_Intersects reports every
    neighbour in the sheet as a dispute and the feature becomes noise. The
    ST_Relate 'T********' pattern asks for shared INTERIOR, and
    test_adjacent_parcels_are_not_overlaps is what holds that line.

  * A parcel with no georeferencing MUST NOT be stored. A pixel polygon
    written as if it were lon/lat puts the parcel in the Gulf of Guinea, and
    a point lookup at 0,0 would then "find" a village in Uttar Pradesh.

Run from anywhere with:
    python3 tests/test_postgis.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import db as db_mod      # noqa: E402
import postgis           # noqa: E402

TEST_URL = os.environ.get("TEST_POSTGIS_URL")

# A 2x2 village block near Lucknow, ~110 m to a side.
LON, LAT, STEP = 80.9490, 26.8510, 0.0010


def box(lon, lat, w, h):
    return [(lon, lat), (lon + w, lat), (lon + w, lat + h), (lon, lat + h)]


class FakeParcel:
    """Only the three attributes postgis.store_parcels actually reads."""

    def __init__(self, parcel_id, khasra_number, geo_polygon):
        self.parcel_id = parcel_id
        self.khasra_number = khasra_number
        self.geo_polygon = geo_polygon


def village_of_four():
    return [
        FakeParcel(1, "213/1", box(LON,        LAT,        STEP, STEP)),
        FakeParcel(2, "213/2", box(LON + STEP, LAT,        STEP, STEP)),
        FakeParcel(3, "214/1", box(LON,        LAT + STEP, STEP, STEP)),
        FakeParcel(4, "214/2", box(LON + STEP, LAT + STEP, STEP, STEP)),
    ]


@unittest.skipUnless(TEST_URL, "TEST_POSTGIS_URL not set")
class PostGISTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.environ["DATABASE_URL"] = TEST_URL
        cls.db = db_mod.Database(os.path.join(ROOT, "storage", "_pgis_test.sqlite3"))
        if not postgis.available(cls.db):
            raise unittest.SkipTest(
                f"postgis unavailable: {postgis.status(cls.db).get('reason')}")

    def setUp(self):
        self.db.run("DROP TABLE IF EXISTS parcel_geometry")
        postgis.ensure_schema(self.db)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.db.run("DROP TABLE IF EXISTS parcel_geometry")
        except Exception:
            pass


class TestAvailability(PostGISTestCase):

    def test_available_is_true_here(self):
        self.assertTrue(postgis.available(self.db))

    def test_status_reports_a_version(self):
        status = postgis.status(self.db)
        self.assertTrue(status["available"])
        self.assertTrue(status["version"])

    def test_sqlite_is_never_reported_as_available(self):
        # The guard that keeps every one of these functions inert on the
        # laptop path. A false positive here would raise on a laptop demo.
        self.assertFalse(postgis.available(None))


class TestSchema(PostGISTestCase):

    def test_spatial_index_exists(self):
        # Without the GIST index the overlap query degrades to a scan, which
        # is the entire reason to use a spatial database at district scale.
        names = [r["indexname"] for r in
                 self.db.q("SELECT indexname FROM pg_indexes "
                           "WHERE tablename = 'parcel_geometry'")]
        self.assertIn("idx_parcel_geom", names)

    def test_geometry_column_is_wgs84(self):
        row = self.db.one("SELECT srid, type FROM geometry_columns "
                          "WHERE f_table_name = 'parcel_geometry'")
        self.assertEqual(row["srid"], 4326)
        self.assertEqual(row["type"].upper(), "POLYGON")


class TestStoring(PostGISTestCase):

    def test_stores_every_georeferenced_parcel(self):
        self.assertEqual(postgis.store_parcels(self.db, "v", village_of_four()), 4)

    def test_ungeoreferenced_parcel_is_skipped(self):
        # Storing this would place the parcel at 0,0 - in the Gulf of Guinea.
        stored = postgis.store_parcels(self.db, "v", [FakeParcel(9, "9/9", None)])
        self.assertEqual(stored, 0)

    def test_restoring_updates_rather_than_duplicating(self):
        postgis.store_parcels(self.db, "v", village_of_four())
        postgis.store_parcels(self.db, "v", village_of_four())
        self.assertEqual(
            self.db.one("SELECT COUNT(*) AS c FROM parcel_geometry")["c"], 4)

    def test_open_ring_is_closed_rather_than_rejected(self):
        ring = box(LON, LAT, STEP, STEP)            # 4 points, not closed
        self.assertTrue(postgis._ring_to_wkt(ring).endswith(
            f"{ring[0][0]} {ring[0][1]}))"))

    def test_degenerate_ring_is_refused(self):
        self.assertIsNone(postgis._ring_to_wkt([(1.0, 2.0), (1.0, 2.0)]))


class TestArea(PostGISTestCase):

    def test_geodesic_area_is_returned_in_square_metres(self):
        postgis.store_parcels(self.db, "v", village_of_four())
        areas = postgis.geodesic_areas(self.db, "v")
        self.assertEqual(len(areas), 4)
        # ~0.001 degree square near 26.85N is a little over a hectare.
        for m2 in areas.values():
            self.assertTrue(10_000 < m2 < 12_000, m2)

    def test_agrees_with_the_python_approximation_at_village_scale(self):
        # The Python path is an equirectangular approximation. This asserts
        # the two stay within 1% for a parcel of this size and latitude -
        # i.e. that the approximation is sound HERE, which is what justifies
        # keeping it as the no-database default.
        import cadastral
        parcels = village_of_four()
        postgis.store_parcels(self.db, "v", parcels)
        areas = postgis.geodesic_areas(self.db, "v")
        for p in parcels:
            approx = cadastral.polygon_area_m2(p.geo_polygon)
            exact = areas[p.parcel_id]
            self.assertLess(abs(exact - approx) / exact, 0.01)


class TestValidity(PostGISTestCase):

    def test_clean_map_reports_nothing(self):
        postgis.store_parcels(self.db, "v", village_of_four())
        self.assertEqual(postgis.validate_geometry(self.db, "v"), [])

    def test_self_intersection_is_caught_and_located(self):
        bow = FakeParcel(7, "BOW/1",
                         [(LON, LAT), (LON + STEP, LAT + STEP),
                          (LON + STEP, LAT), (LON, LAT + STEP)])
        postgis.store_parcels(self.db, "bow", [bow])
        issues = postgis.validate_geometry(self.db, "bow")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["rule"], "GEOMETRY_INVALID")
        self.assertEqual(issues[0]["severity"], "error")
        # The coordinates of the fault are the useful part of the message.
        self.assertIn("Self-intersection", issues[0]["message"])


class TestOverlaps(PostGISTestCase):

    def test_adjacent_parcels_are_not_overlaps(self):
        # THE case this module has to get right. Every parcel in a village
        # shares edges with its neighbours; reporting those as disputes would
        # flag the entire sheet and make the feature worthless.
        postgis.store_parcels(self.db, "v", village_of_four())
        self.assertEqual(postgis.find_overlaps(self.db, "v"), [])

    def test_genuine_overlap_is_reported_with_its_area(self):
        parcels = village_of_four()
        parcels.append(FakeParcel(5, "213/1-A",
                                  box(LON + STEP / 2, LAT, STEP, STEP)))
        postgis.store_parcels(self.db, "v", parcels)
        overlaps = postgis.find_overlaps(self.db, "v")
        self.assertTrue(overlaps)
        self.assertEqual(overlaps[0]["rule"], "PARCELS_OVERLAP")
        self.assertGreater(overlaps[0]["m2"], 0)

    def test_each_overlapping_pair_is_reported_once(self):
        # a<b in the join, so a pair must not appear as both (1,5) and (5,1).
        parcels = village_of_four()
        parcels.append(FakeParcel(5, "213/1-A",
                                  box(LON + STEP / 2, LAT, STEP, STEP)))
        postgis.store_parcels(self.db, "v", parcels)
        pairs = [tuple(sorted(o["parcels"])) for o in
                 postgis.find_overlaps(self.db, "v")]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_slivers_below_the_threshold_are_ignored(self):
        # Rounding in georeferencing produces overlaps a few centimetres
        # wide. Those are arithmetic, not disputes.
        #
        # The sliver has to be a THIN box straddling the shared edge. An
        # earlier version of this test used a full-width box offset by
        # 0.99999 STEP, which overlaps the neighbour almost entirely - it
        # was testing a total overlap and calling it a sliver.
        parcels = village_of_four()
        parcels.append(FakeParcel(6, "SLIVER",
                                  box(LON + STEP * 0.999, LAT,
                                      STEP * 0.002, STEP)))
        postgis.store_parcels(self.db, "v", parcels)
        # ~0.1 m x 110 m is about 12 m2 against each neighbour.
        self.assertEqual(postgis.find_overlaps(self.db, "v", min_area_m2=100.0), [])
        # ...and it IS reported once the threshold drops below it.
        self.assertTrue(postgis.find_overlaps(self.db, "v", min_area_m2=1.0))


class TestOverlappingClaims(PostGISTestCase):
    """The capability the Python path cannot provide at all."""

    def _two_documents_on_the_same_ground(self):
        parcels = village_of_four()
        parcels.append(FakeParcel(5, "213/1-A",
                                  box(LON + STEP / 2, LAT, STEP, STEP)))
        postgis.store_parcels(self.db, "v", parcels)
        self.db.run("UPDATE parcel_geometry SET document_id = 101 "
                    "WHERE map_id='v' AND parcel_id=1")
        self.db.run("UPDATE parcel_geometry SET document_id = 202 "
                    "WHERE map_id='v' AND parcel_id=5")

    def test_two_documents_claiming_the_same_land_are_found(self):
        self._two_documents_on_the_same_ground()
        claims = postgis.find_overlapping_claims(self.db)
        self.assertTrue(claims)
        self.assertEqual(claims[0]["rule"], "OVERLAPPING_CLAIM")
        self.assertEqual(claims[0]["severity"], "error")
        self.assertEqual(sorted(claims[0]["documents"]), [101, 202])

    def test_unlinked_parcels_are_not_a_claim(self):
        # Overlapping geometry with no documents attached is a survey problem
        # for find_overlaps, not a dispute between two owners.
        parcels = village_of_four()
        parcels.append(FakeParcel(5, "213/1-A",
                                  box(LON + STEP / 2, LAT, STEP, STEP)))
        postgis.store_parcels(self.db, "v", parcels)
        self.assertEqual(postgis.find_overlapping_claims(self.db), [])

    def test_one_document_over_two_parcels_is_not_a_dispute(self):
        # A single record legitimately covering adjoining parcels must not be
        # reported as being in dispute with itself.
        self._two_documents_on_the_same_ground()
        self.db.run("UPDATE parcel_geometry SET document_id = 101 "
                    "WHERE map_id='v' AND parcel_id=5")
        self.assertEqual(postgis.find_overlapping_claims(self.db), [])


class TestPointLookup(PostGISTestCase):

    def test_finds_the_parcel_containing_a_point(self):
        postgis.store_parcels(self.db, "v", village_of_four(), village="Narharpur")
        hit = postgis.parcel_containing(self.db, LAT + 0.0002, LON + 0.0002)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["khasra_number"], "213/1")

    def test_a_point_on_no_parcel_returns_nothing(self):
        # 0,0 specifically: it is where an ungeoreferenced parcel would land.
        postgis.store_parcels(self.db, "v", village_of_four())
        self.assertIsNone(postgis.parcel_containing(self.db, 0.0, 0.0))


class TestLinking(PostGISTestCase):

    def test_link_document_attaches_a_record(self):
        postgis.store_parcels(self.db, "v", village_of_four())
        self.assertTrue(postgis.link_document(self.db, "v", 2, 303))
        row = self.db.one("SELECT document_id FROM parcel_geometry "
                          "WHERE map_id='v' AND parcel_id=2")
        self.assertEqual(row["document_id"], 303)

    def test_linking_an_absent_parcel_reports_failure(self):
        self.assertFalse(postgis.link_document(self.db, "v", 999, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
