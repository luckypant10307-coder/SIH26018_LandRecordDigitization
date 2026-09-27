#!/usr/bin/env python3
"""
PostgreSQL backend tests for backend/db.py.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

SKIPPED unless TEST_DATABASE_URL points at a Postgres this suite may DROP AND
RECREATE. That default matters: the project's promise is that the whole test
suite runs on a laptop with nothing installed, so needing a database server
would have to be opt-in or the promise is broken.

    docker run -d --name sihpg -e POSTGRES_PASSWORD=sih \
        -e POSTGRES_DB=landrecords -p 55432:5432 postgres:16-alpine
    TEST_DATABASE_URL=postgresql://postgres:sih@127.0.0.1:55432/landrecords \
        python3 tests/test_postgres.py -v

What is tested is the dialect boundary, because that is where the engines can
silently disagree: placeholder translation, the upsert whose SQLite form has
different semantics, GROUP BY strictness, keyword-quoting, and the advisory
lock that replaces a process lock the moment writers can live in different
processes.

Run from anywhere with:
    python3 tests/test_postgres.py -v
"""

from __future__ import annotations

import os
import sys
import threading
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import db as db_mod  # noqa: E402

TEST_URL = os.environ.get("TEST_DATABASE_URL")
OPERATOR = {"id": 1, "username": "operator1", "role": "operator"}


def fresh() -> "db_mod.Database":
    """A Database on an empty schema. Drops whatever was there."""
    import psycopg
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute("DROP SCHEMA public CASCADE")
        conn.execute("CREATE SCHEMA public")
    return db_mod.Database(url=TEST_URL)


@unittest.skipUnless(TEST_URL, "set TEST_DATABASE_URL to run the Postgres tests")
class PostgresTestCase(unittest.TestCase):

    def setUp(self):
        self.db = fresh()

    def tearDown(self):
        try:
            self.db._conn.close()
        except Exception:
            pass


class TestEngineSelection(PostgresTestCase):

    def test_a_url_selects_postgres(self):
        self.assertTrue(self.db.is_postgres)

    def test_no_url_still_gives_sqlite(self):
        """The zero-install default must survive the Postgres work."""
        import tempfile
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            local = db_mod.Database(os.path.join(d, "t.db"))
            self.assertFalse(local.is_postgres)
            local._conn.close()


class TestSchemaAndSeed(PostgresTestCase):

    def test_schema_applies_and_users_are_seeded(self):
        self.assertEqual(len(self.db.list_users()), 4)

    def test_bigserial_replaced_autoincrement(self):
        """AUTOINCREMENT is SQLite-only; ids must still be generated."""
        first = self.db.run(
            "INSERT INTO users (username, full_name, role, office, created_at) "
            "VALUES (?,?,?,?,?)", ("x1", "X", "operator", "O", "2026-01-01"))
        second = self.db.run(
            "INSERT INTO users (username, full_name, role, office, created_at) "
            "VALUES (?,?,?,?,?)", ("x2", "X", "operator", "O", "2026-01-01"))
        self.assertTrue(first > 0 and second > first)

    def test_rows_index_by_column_name(self):
        """
        The rest of the codebase reads row["col"]. sqlite3.Row and psycopg's
        dict_row both allow it, which is why no caller needed changing.
        """
        row = self.db.one("SELECT username, role FROM users ORDER BY id LIMIT 1")
        self.assertEqual(row["username"], "operator1")


class TestPlaceholderTranslation(PostgresTestCase):

    def test_question_marks_become_percent_s(self):
        row = self.db.one("SELECT * FROM users WHERE username = ?", ("admin1",))
        self.assertEqual(row["role"], "admin")

    def test_a_question_mark_inside_a_literal_is_not_a_placeholder(self):
        """
        Rewriting every '?' blindly would corrupt a query containing one in a
        string. Only text outside quotes is translated.
        """
        row = self.db.one("SELECT '?' AS q, username FROM users LIMIT 1")
        self.assertEqual(row["q"], "?")


class TestUpsert(PostgresTestCase):

    def _doc(self) -> int:
        return self.db.insert_document(
            filename="f.pdf", stored_path="/x/f.pdf", preview_path=None,
            sha256="a" * 64, file_size=1, mime="application/pdf",
            uploaded_by=1, ocr_engine="pdf_text_layer", page_count=1,
            mean_ocr_conf=0.9, legibility=90.0, quality_json="{}",
            warnings_json="[]", full_text="x", status="needs_review",
            decision="needs_review", trust_score=50.0, error_count=0,
            warning_count=0, signature="sig", issues_json="[]",
            summary_json="{}", processing_ms=1)

    def test_reinserting_a_field_replaces_rather_than_duplicates(self):
        doc = self._doc()
        for value in ("first", "second"):
            self.db.insert_field(doc, {"key": "village", "display": "Village",
                                       "value": value, "confidence": 0.9})
        rows = self.db.q("SELECT value FROM fields WHERE document_id = ? "
                         "AND field_key = ?", (doc, "village"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], "second")

    def test_reingest_clears_the_human_correction_attribution(self):
        """
        SQLite's INSERT OR REPLACE deletes the row, so corrected_by/at reset.
        ON CONFLICT would have preserved them, so they are cleared explicitly -
        both engines must agree that a re-read field is a machine value again.
        """
        doc = self._doc()
        self.db.insert_field(doc, {"key": "village", "display": "Village",
                                   "value": "A", "confidence": 0.9})
        self.db.run("UPDATE fields SET corrected_by = ?, corrected_at = ? "
                    "WHERE document_id = ? AND field_key = ?",
                    (2, "2026-01-01", doc, "village"))
        self.db.insert_field(doc, {"key": "village", "display": "Village",
                                   "value": "B", "confidence": 0.9})
        row = self.db.one("SELECT corrected_by, corrected_at FROM fields "
                          "WHERE document_id = ? AND field_key = ?",
                          (doc, "village"))
        self.assertIsNone(row["corrected_by"])
        self.assertIsNone(row["corrected_at"])


class TestStatsPortability(PostgresTestCase):
    """
    stats() is where the dialects diverge most: ROUND's argument types, GROUP BY
    strictness, and "day" being a Postgres keyword. It must simply run.
    """

    def test_stats_runs_on_an_empty_database(self):
        s = self.db.stats()
        self.assertEqual(s["documents_total"], 0)

    def test_stats_runs_with_data(self):
        self.db.audit(OPERATOR, "document_ingested", 1)
        s = self.db.stats()
        self.assertIn("by_district", s)
        self.assertIn("daily_volume", s)
        self.assertIn("field_accuracy", s)


class TestAuditChain(PostgresTestCase):

    def test_chain_verifies_on_postgres(self):
        for i in range(5):
            self.db.audit(OPERATOR, "document_ingested", i)
        result = self.db.verify_audit_chain()
        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["entries"], 5)

    def test_tampering_is_caught(self):
        self.db.audit(OPERATOR, "document_ingested", 1)
        self.db.audit(OPERATOR, "field_corrected", 1, "owner_name", "A", "B")
        rows = self.db.q("SELECT id FROM audit_log ORDER BY id")
        self.db.run("UPDATE audit_log SET new_value = ? WHERE id = ?",
                    ("FORGED", rows[1]["id"]))
        result = self.db.verify_audit_chain()
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "content_altered")

    def test_concurrent_writers_on_separate_connections_do_not_fork_the_chain(self):
        """
        THE reason Postgres needed more than the existing process lock. Separate
        connections are exactly what a second server instance or a psql session
        is, and two of them reading the same tip would fork the chain and make
        an untampered log fail verification permanently. The advisory lock in
        audit() serialises them.
        """
        writers = [db_mod.Database(url=TEST_URL) for _ in range(4)]
        errors = []

        def append(handle, which):
            for i in range(10):
                try:
                    handle.audit(OPERATOR, "document_ingested", which * 100 + i)
                except Exception as exc:            # pragma: no cover
                    errors.append(repr(exc))

        threads = [threading.Thread(target=append, args=(writers[i], i))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for handle in writers:
            handle._conn.close()

        self.assertEqual(errors, [])
        result = self.db.verify_audit_chain()
        self.assertEqual(result["entries"], 40)
        self.assertTrue(result["ok"], result["message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
