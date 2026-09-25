#!/usr/bin/env python3
"""
Unit tests for the audit tamper-evidence chain (backend/db.py).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

The chain exists to make tampering with a land record's history detectable, so
these tests are written from the attacker's side: each one performs the edit a
dishonest operator would actually attempt, then asserts that verification
catches it AND names the right row.

A test that only checks "an untouched log verifies" would pass against an
implementation that always returns ok, which is why the majority of what
follows is deliberate corruption.

Run from anywhere with:
    python3 tests/test_audit_chain.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import db as db_mod  # noqa: E402


OPERATOR = {"id": 1, "username": "operator1", "role": "operator"}
VERIFIER = {"id": 2, "username": "verifier1", "role": "verifier"}


class ChainTestCase(unittest.TestCase):
    """A throwaway database per test, with a short audit history in it."""

    def setUp(self):
        # Windows will not unlink an open SQLite file, so the connection is
        # closed in tearDown before the directory goes.
        self._dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.db = db_mod.Database(os.path.join(self._dir.name, "test.db"))
        # _seed_users() writes no audit rows, so the log starts empty and the
        # ids below are predictable.
        self.db.audit(OPERATOR, "document_ingested", 1, detail="engine=pdf_text_layer")
        self.db.audit(VERIFIER, "field_corrected", 1, "owner_name",
                      "सुनीता देवा", "सुनीता देवी")
        self.db.audit(VERIFIER, "document_approved", 1, detail="decision=auto_approved")

    def tearDown(self):
        try:
            self.db._conn.close()
        except Exception:
            pass
        self._dir.cleanup()

    def _raw(self, sql, params=()):
        """Write straight past Database.audit(), the way tampering would."""
        with db_mod._LOCK:
            self.db._conn.execute(sql, params)
            self.db._conn.commit()


class TestHonestLog(ChainTestCase):

    def test_an_untampered_log_verifies(self):
        result = self.db.verify_audit_chain()
        self.assertTrue(result["ok"], result["message"])
        self.assertEqual(result["entries"], 3)
        self.assertEqual(result["verified"], 3)
        self.assertIsNone(result["broken_at"])

    def test_every_row_links_to_the_one_before_it(self):
        rows = self.db.q("SELECT * FROM audit_log ORDER BY id")
        self.assertIsNone(rows[0]["prev_hash"])
        for earlier, later in zip(rows, rows[1:]):
            self.assertEqual(later["prev_hash"], earlier["entry_hash"])

    def test_hashes_are_distinct_per_row(self):
        hashes = [r["entry_hash"] for r in
                  self.db.q("SELECT entry_hash FROM audit_log ORDER BY id")]
        self.assertEqual(len(hashes), len(set(hashes)))
        for h in hashes:
            self.assertEqual(len(h), 64)

    def test_an_empty_log_verifies(self):
        self._raw("DELETE FROM audit_log")
        result = self.db.verify_audit_chain()
        self.assertTrue(result["ok"])
        self.assertEqual(result["entries"], 0)


class TestTampering(ChainTestCase):

    def test_editing_an_owner_name_in_place_is_caught(self):
        """The attack this whole feature exists for: quietly rewrite who owns it."""
        self._raw("UPDATE audit_log SET new_value = ? WHERE id = 2", ("रीता देवी",))
        result = self.db.verify_audit_chain()
        self.assertFalse(result["ok"])
        self.assertEqual(result["broken_at"], 2)
        self.assertEqual(result["reason"], "content_altered")

    def test_deleting_a_row_is_caught(self):
        """Removing the correction so the record looks clean from the start."""
        self._raw("DELETE FROM audit_log WHERE id = 2")
        result = self.db.verify_audit_chain()
        self.assertFalse(result["ok"])
        self.assertEqual(result["broken_at"], 3)
        self.assertEqual(result["reason"], "broken_link")

    def test_changing_who_did_it_is_caught(self):
        self._raw("UPDATE audit_log SET username = ?, user_id = ? WHERE id = 3",
                  ("operator1", 1))
        result = self.db.verify_audit_chain()
        self.assertFalse(result["ok"])
        self.assertEqual(result["broken_at"], 3)
        self.assertEqual(result["reason"], "content_altered")

    def test_backdating_an_entry_is_caught(self):
        self._raw("UPDATE audit_log SET at = ? WHERE id = 1", ("2019-01-01T09:00:00",))
        result = self.db.verify_audit_chain()
        self.assertFalse(result["ok"])
        self.assertEqual(result["broken_at"], 1)

    def test_tampering_reports_the_first_bad_row_not_the_last(self):
        """Everything after the first break is unverifiable, so that is the answer."""
        self._raw("UPDATE audit_log SET detail = 'x' WHERE id = 1")
        self._raw("UPDATE audit_log SET detail = 'y' WHERE id = 3")
        self.assertEqual(self.db.verify_audit_chain()["broken_at"], 1)

    def test_a_forged_row_appended_by_hand_is_caught(self):
        """An insert that bypasses audit() cannot produce a valid hash."""
        self._raw("INSERT INTO audit_log (at, username, role, action, document_id,"
                  " prev_hash, entry_hash) VALUES (?,?,?,?,?,?,?)",
                  ("2026-01-01T00:00:00", "admin1", "admin", "document_approved", 9,
                   "deadbeef" * 8, "cafebabe" * 8))
        result = self.db.verify_audit_chain()
        self.assertFalse(result["ok"])
        self.assertEqual(result["broken_at"], 4)
        self.assertEqual(result["reason"], "broken_link")


class TestAppendAfterTampering(ChainTestCase):

    def test_a_later_honest_entry_does_not_repair_a_break(self):
        """
        Tampering must stay visible. If a normal append silently re-sealed the
        chain, an attacker could edit a row and then wait for ordinary traffic
        to cover it.
        """
        self._raw("UPDATE audit_log SET new_value = 'रीता देवी' WHERE id = 2")
        self.db.audit(OPERATOR, "document_ingested", 2, detail="engine=tesseract")
        result = self.db.verify_audit_chain()
        self.assertFalse(result["ok"])
        self.assertEqual(result["broken_at"], 2)


class TestMigration(ChainTestCase):

    def test_rows_written_before_the_chain_existed_are_sealed(self):
        """
        An installation upgrading from the pre-chain schema has audit rows with
        no hashes. They must be sealed in id order, and the result must verify.
        """
        self._raw("UPDATE audit_log SET prev_hash = NULL, entry_hash = NULL")
        sealed = self.db._seal_unchained()
        self.assertEqual(sealed, 3)
        self.assertTrue(self.db.verify_audit_chain()["ok"])

    def test_sealing_does_not_recompute_rows_that_already_have_hashes(self):
        """Re-sealing must never quietly repair a chain verification should fail."""
        before = [r["entry_hash"] for r in
                  self.db.q("SELECT entry_hash FROM audit_log ORDER BY id")]
        self._raw("UPDATE audit_log SET new_value = 'रीता देवी' WHERE id = 2")
        self.assertEqual(self.db._seal_unchained(), 0)
        after = [r["entry_hash"] for r in
                 self.db.q("SELECT entry_hash FROM audit_log ORDER BY id")]
        self.assertEqual(before, after)
        self.assertFalse(self.db.verify_audit_chain()["ok"])

    def test_migration_is_idempotent(self):
        self.assertEqual(self.db._migrate_audit_chain(), 0)
        self.assertTrue(self.db.verify_audit_chain()["ok"])


class TestDigest(unittest.TestCase):

    def test_devanagari_content_digests_identically_across_calls(self):
        row = {"at": "2026-01-01T00:00:00", "user_id": 2, "username": "verifier1",
               "role": "verifier", "action": "field_corrected", "document_id": 1,
               "field_key": "owner_name", "old_value": "सुनीता देवा",
               "new_value": "सुनीता देवी", "detail": None}
        self.assertEqual(db_mod._chain_digest(None, row),
                         db_mod._chain_digest(None, row))

    def test_a_different_predecessor_gives_a_different_hash(self):
        row = {"at": "2026-01-01T00:00:00", "user_id": 1, "username": "operator1",
               "role": "operator", "action": "document_ingested", "document_id": 1,
               "field_key": None, "old_value": None, "new_value": None,
               "detail": None}
        self.assertNotEqual(db_mod._chain_digest(None, row),
                            db_mod._chain_digest("a" * 64, row))

    def test_the_first_row_is_anchored_to_the_genesis_constant(self):
        row = {"at": "2026-01-01T00:00:00", "user_id": 1, "username": "operator1",
               "role": "operator", "action": "document_ingested", "document_id": 1,
               "field_key": None, "old_value": None, "new_value": None,
               "detail": None}
        self.assertEqual(db_mod._chain_digest(None, row),
                         db_mod._chain_digest(db_mod.GENESIS_HASH, row))


if __name__ == "__main__":
    unittest.main(verbosity=2)
