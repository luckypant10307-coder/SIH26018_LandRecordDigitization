"""
PostGIS spatial store for parcel geometry.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT REPLACE

topology.py already checks ring validity, overlaps, containment and vertex
snapping in pure Python, and it stays: it is the path that works on SQLite, on
a laptop, with no database server, which is the promise the rest of this
project rests on. This module is what becomes available when a deployment has
a real spatial database, and it is not a reimplementation - it does four
things Python does worse or cannot do at all.

  1. GEODESIC AREA. cadastral.polygon_area_m2() projects equirectangularly
     about the ring's own latitude and applies the shoelace formula. That is
     honest for a parcel a few hundred metres across and its error grows with
     size and latitude. ST_Area(geography) is the real thing on the ellipsoid,
     so the two can be COMPARED - and a disagreement between them is itself a
     signal that a parcel is larger or further north than the approximation
     was meant for.

  2. RIGOROUS VALIDITY. ST_IsValidReason applies the OGC rules and names the
     failure and its location: "Self-intersection at 80.95 26.85". Python's
     checks catch the cases they were written for; this catches the ones
     nobody thought of.

  3. OVERLAP DETECTION THAT SCALES. find_overlaps() compares every ring to
     every other, which is O(n^2) - fine for the 14 parcels of a demo sheet
     and hopeless for a district's tens of thousands. A GIST index turns the
     same question into an indexed lookup.

  4. OVERLAPPING CLAIMS, which is the one that matters for land records and
     which the system cannot currently answer at all. Two DIFFERENT documents
     whose parcels intersect on the ground is either a survey error or a
     genuine boundary dispute, and it is invisible to a per-document
     validator. This is what a revenue office actually needs from a spatial
     database, and it is why "PostGIS" belonged on the architecture diagram.

ENABLED ONLY WHEN IT IS REALLY THERE

Requires DATABASE_URL to point at Postgres AND the postgis extension to be
installed. Absent either, every function here returns empty and the Python
topology path is unaffected - the system does not degrade, it simply does not
gain the extra checks, and available() says so.

The geometry column is SRID 4326 (WGS84 lon/lat), matching what georeference.py
produces and what RFC 7946 GeoJSON requires, so nothing is reprojected on the
way in. Areas are computed by casting to `geography`, which is what makes
ST_Area return square metres on the ellipsoid rather than square degrees.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS parcel_geometry (
    id              BIGSERIAL PRIMARY KEY,
    map_id          TEXT NOT NULL,
    parcel_id       INTEGER NOT NULL,
    khasra_number   TEXT,
    village         TEXT,
    district        TEXT,
    document_id     BIGINT,
    is_demo         BOOLEAN DEFAULT FALSE,
    geom            geometry(Polygon, 4326) NOT NULL,
    UNIQUE (map_id, parcel_id)
);
CREATE INDEX IF NOT EXISTS idx_parcel_geom ON parcel_geometry USING GIST (geom);
CREATE INDEX IF NOT EXISTS idx_parcel_khasra ON parcel_geometry (khasra_number);
CREATE INDEX IF NOT EXISTS idx_parcel_document ON parcel_geometry (document_id);
"""


def available(db) -> bool:
    """True only on Postgres with the postgis extension actually installed."""
    if db is None or not getattr(db, "is_postgres", False):
        return False
    try:
        row = db.one("SELECT installed_version FROM pg_available_extensions "
                     "WHERE name = 'postgis'")
        return bool(row and row["installed_version"])
    except Exception:
        return False


def status(db) -> dict:
    """Honest capability report for run.py --check."""
    if db is None or not getattr(db, "is_postgres", False):
        return {"available": False,
                "reason": "not running on PostgreSQL (DATABASE_URL unset)"}
    try:
        row = db.one("SELECT default_version, installed_version "
                     "FROM pg_available_extensions WHERE name = 'postgis'")
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if not row:
        return {"available": False,
                "reason": "the postgis extension is not available in this server"}
    if not row["installed_version"]:
        return {"available": False,
                "reason": (f"postgis {row['default_version']} is available but not "
                           f"installed - run: CREATE EXTENSION postgis;")}
    return {"available": True, "version": row["installed_version"]}


def ensure_schema(db) -> bool:
    """
    Create the extension and the geometry table. Returns whether it worked.

    CREATE EXTENSION needs privileges an application role may not have, so a
    failure here is reported rather than raised: the system must still run.
    """
    if db is None or not getattr(db, "is_postgres", False):
        return False
    try:
        db.run("CREATE EXTENSION IF NOT EXISTS postgis")
        for statement in [s.strip() for s in SCHEMA.split(";") if s.strip()]:
            db.run(statement)
        return True
    except Exception:
        return False


def _ring_to_wkt(ring: Sequence) -> Optional[str]:
    """
    A lon/lat ring as WKT POLYGON.

    The ring is closed explicitly if the caller did not: PostGIS rejects an
    unclosed polygon outright, and a vectoriser that emits an open ring is a
    common enough mistake to be worth absorbing here rather than failing the
    whole import.
    """
    points = [(float(x), float(y)) for x, y in ring if x is not None and y is not None]
    if len(points) < 3:
        return None
    if points[0] != points[-1]:
        points.append(points[0])
    inner = ", ".join(f"{x} {y}" for x, y in points)
    return f"POLYGON(({inner}))"


def store_parcels(db, map_id: str, parcels: Sequence, village: Optional[str] = None,
                  district: Optional[str] = None, is_demo: bool = False) -> int:
    """
    Persist a map's parcels as real geometry. Returns how many were stored.

    Only parcels that have been georeferenced are stored - a pixel polygon has
    no place on the earth, and storing one as if it did would put a parcel off
    the coast of Africa at 0,0.
    """
    if not available(db) or not ensure_schema(db):
        return 0
    stored = 0
    for parcel in parcels:
        ring = getattr(parcel, "geo_polygon", None)
        if not ring:
            continue                      # not georeferenced: nothing to store
        wkt = _ring_to_wkt(ring)
        if not wkt:
            continue
        try:
            db.run(
                "INSERT INTO parcel_geometry "
                "(map_id, parcel_id, khasra_number, village, district, is_demo, geom) "
                "VALUES (?,?,?,?,?,?, ST_GeomFromText(?, 4326)) "
                "ON CONFLICT (map_id, parcel_id) DO UPDATE SET "
                "  khasra_number = EXCLUDED.khasra_number, "
                "  village = EXCLUDED.village, district = EXCLUDED.district, "
                "  is_demo = EXCLUDED.is_demo, geom = EXCLUDED.geom",
                (map_id, getattr(parcel, "parcel_id", 0),
                 getattr(parcel, "khasra_number", None),
                 village, district, is_demo, wkt))
            stored += 1
        except Exception:
            continue                      # one bad ring must not abort the map
    return stored


def validate_geometry(db, map_id: str) -> List[dict]:
    """
    OGC validity for every stored parcel, with the reason and the place.

    ST_IsValidReason names the failure - "Self-intersection at 80.95 26.85" -
    which is what a reviewer needs in order to fix a boundary, rather than a
    bare "invalid".
    """
    if not available(db):
        return []
    try:
        rows = db.q(
            "SELECT parcel_id, khasra_number, ST_IsValidReason(geom) AS reason "
            "FROM parcel_geometry WHERE map_id = ? AND NOT ST_IsValid(geom)",
            (map_id,))
    except Exception:
        return []
    return [{
        "rule": "GEOMETRY_INVALID", "severity": "error",
        "parcel_id": r["parcel_id"], "khasra_number": r["khasra_number"],
        "message": (f"Parcel {r['parcel_id']} "
                    f"(khasra {r['khasra_number'] or '-'}) is not a valid "
                    f"polygon: {r['reason']}."),
        "suggestion": "Correct the boundary on the source map and re-import.",
    } for r in rows]


def geodesic_areas(db, map_id: str) -> Dict[int, float]:
    """
    True ellipsoidal area per parcel, in square metres.

    Casting to `geography` is what makes ST_Area return metres rather than
    square degrees, which is a number with no physical meaning.
    """
    if not available(db):
        return {}
    try:
        rows = db.q("SELECT parcel_id, ST_Area(geom::geography) AS m2 "
                    "FROM parcel_geometry WHERE map_id = ?", (map_id,))
    except Exception:
        return {}
    return {int(r["parcel_id"]): float(r["m2"]) for r in rows}


def find_overlaps(db, map_id: str, min_area_m2: float = 1.0) -> List[dict]:
    """
    Parcels on the same map whose interiors intersect.

    ST_Relate with the 'T********' pattern asks for a shared INTERIOR, which
    is the right question: two parcels sharing only a boundary line are
    neighbours, not an overlap, and a naive ST_Intersects would report every
    adjacent pair in the village.

    The GIST index does the pruning, so this stays usable at district scale
    where the pairwise Python version does not.
    """
    if not available(db):
        return []
    try:
        rows = db.q(
            "SELECT a.parcel_id AS a_id, b.parcel_id AS b_id, "
            "       a.khasra_number AS a_khasra, b.khasra_number AS b_khasra, "
            "       ST_Area(ST_Intersection(a.geom, b.geom)::geography) AS m2 "
            "FROM parcel_geometry a JOIN parcel_geometry b "
            "  ON a.map_id = b.map_id AND a.parcel_id < b.parcel_id "
            "WHERE a.map_id = ? AND ST_Relate(a.geom, b.geom, 'T********') "
            "  AND ST_Area(ST_Intersection(a.geom, b.geom)::geography) >= ?",
            (map_id, min_area_m2))
    except Exception:
        return []
    # m2 is carried as a field, not only inside the sentence: a caller
    # ranking disputes by severity, or filtering the worst ones, must not
    # have to parse prose to get a number the query already returned.
    return [{
        "rule": "PARCELS_OVERLAP", "severity": "warning",
        "parcels": [r["a_id"], r["b_id"]],
        "m2": round(float(r["m2"]), 1),
        "message": (f"Parcels {r['a_id']} (khasra {r['a_khasra'] or '-'}) and "
                    f"{r['b_id']} (khasra {r['b_khasra'] or '-'}) overlap by "
                    f"{r['m2']:.1f} m2 of shared interior."),
        "suggestion": "Two parcels cannot occupy the same ground - check the survey.",
    } for r in rows]


def find_overlapping_claims(db, min_area_m2: float = 1.0) -> List[dict]:
    """
    Two DIFFERENT documents whose parcels intersect on the ground.

    This is the question a per-document validator cannot ask and a revenue
    office most needs answered: it is either a survey error or a live boundary
    dispute, and neither is visible when records are checked one at a time.

    Deliberately NOT restricted to one map - a dispute that matters is one
    that crosses sheets.
    """
    if not available(db):
        return []
    try:
        rows = db.q(
            "SELECT a.document_id AS doc_a, b.document_id AS doc_b, "
            "       a.khasra_number AS khasra_a, b.khasra_number AS khasra_b, "
            "       ST_Area(ST_Intersection(a.geom, b.geom)::geography) AS m2 "
            "FROM parcel_geometry a JOIN parcel_geometry b "
            "  ON a.document_id < b.document_id "
            "WHERE a.document_id IS NOT NULL AND b.document_id IS NOT NULL "
            "  AND ST_Relate(a.geom, b.geom, 'T********') "
            "  AND ST_Area(ST_Intersection(a.geom, b.geom)::geography) >= ?",
            (min_area_m2,))
    except Exception:
        return []
    return [{
        "rule": "OVERLAPPING_CLAIM", "severity": "error",
        "documents": [r["doc_a"], r["doc_b"]],
        "m2": round(float(r["m2"]), 1),
        "message": (f"Documents #{r['doc_a']} (khasra {r['khasra_a'] or '-'}) and "
                    f"#{r['doc_b']} (khasra {r['khasra_b'] or '-'}) claim land that "
                    f"overlaps by {r['m2']:.1f} m2."),
        "suggestion": ("Escalate to the revenue authority: this is a survey error "
                       "or a boundary dispute, not a data-entry problem."),
    } for r in rows]


def parcel_containing(db, lat: float, lon: float) -> Optional[dict]:
    """
    Which parcel contains this point, using the spatial index.

    Useful for the reverse lookup a field officer actually performs: standing
    somewhere, which khasra is this?
    """
    if not available(db):
        return None
    try:
        row = db.one(
            "SELECT map_id, parcel_id, khasra_number, village, district, is_demo, "
            "       ST_Area(geom::geography) AS m2 "
            "FROM parcel_geometry "
            "WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint(?, ?), 4326)) LIMIT 1",
            (lon, lat))
    except Exception:
        return None
    return dict(row) if row else None


def link_document(db, map_id: str, parcel_id: int, document_id: int) -> bool:
    """
    Attach a digitised record to the ground it describes.

    Returns whether a parcel was actually matched. The UPDATE succeeding is
    not the same fact: naming a parcel that does not exist on this map
    affects no rows and raises nothing, so returning True on "no exception"
    would report a record as placed on the earth when nothing was linked -
    and the caller would have no way to discover that.
    """
    if not available(db):
        return False
    try:
        # Existence is checked separately rather than read from the UPDATE,
        # because Database.run() does not return a rowcount: on Postgres it
        # returns the new id for an INSERT and 0 for everything else, so
        # `run(UPDATE...) > 0` is false even when rows were changed.
        if not db.one("SELECT 1 AS present FROM parcel_geometry "
                      "WHERE map_id = ? AND parcel_id = ?", (map_id, parcel_id)):
            return False
        db.run("UPDATE parcel_geometry SET document_id = ? "
               "WHERE map_id = ? AND parcel_id = ?",
               (document_id, map_id, parcel_id))
        return True
    except Exception:
        return False
