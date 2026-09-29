"""
Persistence layer (SQLite, standard library only).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Covers the storage requirements named in the problem statement:
  * secure document repository with metadata management
  * audit trails (every field change is recorded, nothing is overwritten silently)
  * role-based access control
  * the correction corpus that feeds the learning loop

SQLite is used deliberately: the demo must run offline with no server to
install. The schema is plain relational SQL, so migrating to PostgreSQL for a
state-scale deployment is a connection-string change, not a rewrite.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(_HERE, "..", "storage", "landrecords.db")

_LOCK = threading.Lock()


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Users and roles ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT UNIQUE NOT NULL,
    full_name   TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('operator','verifier','admin','auditor')),
    office      TEXT,
    created_at  TEXT NOT NULL
);

-- Documents --------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    stored_path     TEXT NOT NULL,
    preview_path    TEXT,
    sha256          TEXT NOT NULL,
    file_size       INTEGER NOT NULL,
    mime            TEXT,
    uploaded_by     INTEGER REFERENCES users(id),
    uploaded_at     TEXT NOT NULL,
    -- extraction metadata
    ocr_engine      TEXT,
    page_count      INTEGER,
    mean_ocr_conf   REAL,
    legibility      REAL,
    quality_json    TEXT,
    warnings_json   TEXT,
    full_text       TEXT,
    -- validation outcome
    status          TEXT NOT NULL DEFAULT 'processing'
                    CHECK (status IN ('processing','needs_review','blocked',
                                      'auto_approved','approved','rejected')),
    decision        TEXT,
    trust_score     REAL,
    error_count     INTEGER DEFAULT 0,
    warning_count   INTEGER DEFAULT 0,
    signature       TEXT,
    issues_json     TEXT,
    summary_json    TEXT,
    -- Where this record is on the earth, as structured data.
    --
    -- The geotag was previously computed and then thrown away: it reached the
    -- issue list as a human-readable sentence and nothing else, so the map had
    -- no coordinate to plot and the API returned no position. A sentence is
    -- not a location.
    geotag_json     TEXT,
    -- verification
    verified_by     INTEGER REFERENCES users(id),
    verified_at     TEXT,
    reject_reason   TEXT,
    processing_ms   INTEGER
);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_signature ON documents(signature);
CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);

-- Extracted fields -------------------------------------------------------
CREATE TABLE IF NOT EXISTS fields (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    field_key       TEXT NOT NULL,
    display         TEXT,
    -- machine output (never mutated after ingestion)
    ai_value        TEXT,
    ai_confidence   REAL,
    conf_label      REAL,
    conf_pattern    REAL,
    conf_ocr        REAL,
    -- current accepted value (may be human-corrected)
    value           TEXT,
    status          TEXT NOT NULL DEFAULT 'extracted'
                    CHECK (status IN ('extracted','missing','needs_review',
                                      'confirmed','corrected')),
    page            INTEGER,
    bbox_json       TEXT,
    source_line     TEXT,
    notes_json      TEXT,
    extra_json      TEXT,
    corrected_by    INTEGER REFERENCES users(id),
    corrected_at    TEXT,
    UNIQUE (document_id, field_key)
);
CREATE INDEX IF NOT EXISTS idx_fields_doc ON fields(document_id);

-- Immutable audit trail --------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    user_id     INTEGER REFERENCES users(id),
    username    TEXT,
    role        TEXT,
    action      TEXT NOT NULL,
    document_id INTEGER,
    field_key   TEXT,
    old_value   TEXT,
    new_value   TEXT,
    detail      TEXT,
    -- Tamper-evidence chain. Each row carries the hash of the row before it,
    -- so editing or deleting any past entry invalidates every entry after it.
    -- See _chain_digest() and verify_audit_chain().
    prev_hash   TEXT,
    entry_hash  TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_doc ON audit_log(document_id);
CREATE INDEX IF NOT EXISTS idx_audit_at ON audit_log(at);

-- Learning corpus: every human correction is a training signal ----------
CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    at              TEXT NOT NULL,
    document_id     INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    field_key       TEXT NOT NULL,
    ai_value        TEXT,
    human_value     TEXT,
    ai_confidence   REAL,
    ocr_engine      TEXT,
    source_line     TEXT
);
CREATE INDEX IF NOT EXISTS idx_corrections_field ON corrections(field_key);
"""


def _now() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------
# Audit tamper-evidence chain
# --------------------------------------------------------------------------
#
# WHAT THIS DOES, AND WHAT IT DOES NOT
#
# Each audit row stores the SHA-256 of (the previous row's hash + this row's
# own content). Editing an old entry, or deleting one, changes that row's hash
# and breaks the link every later row depends on, so verify_audit_chain()
# localises the tampering to the first row that no longer agrees.
#
# This is deliberately NOT a blockchain, and claiming it were one would be the
# kind of overclaim the rest of this system avoids. There is no distributed
# consensus and no external notary: an attacker with write access to the file
# can recompute the whole chain from the edited row onward and it will verify
# clean. What the chain buys is that tampering can no longer be SILENT - it
# must be deliberate and complete, and a copy of the tip hash held anywhere
# outside this database (a nightly export, a printout, a second office) turns
# even that into a detectable change. That is the honest claim: tamper-EVIDENT,
# not tamper-proof.
#
# Rows written before the chain existed are sealed on first migration. Sealing
# proves nothing about what happened to them beforehand; it only establishes
# the baseline that everything after it is measured against.

GENESIS_HASH = "0" * 64

# Arbitrary but fixed: every appender must ask for the SAME advisory lock key
# or the serialisation it provides is worthless. Scoped to the audit chain
# alone so it never contends with ordinary document writes.
_AUDIT_LOCK_KEY = 0x5A1D_AD17

# The row fields the hash covers. `id` is excluded because SQLite assigns it
# after the digest must already exist; row deletion is still caught, because
# the following row's prev_hash then matches no surviving predecessor.
_CHAINED_FIELDS = ("at", "user_id", "username", "role", "action",
                   "document_id", "field_key", "old_value", "new_value",
                   "detail")


def _chain_digest(prev_hash: Optional[str], row: Any) -> str:
    """
    The hash of one audit entry, bound to its predecessor.

    Serialisation is canonical (sorted keys, fixed separators, UTF-8) so the
    same row always digests identically - a Devanagari owner name must not
    hash differently just because it was read back through a different code
    path.
    """
    payload = json.dumps({k: row[k] for k in _CHAINED_FIELDS},
                         sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    seed = (prev_hash or GENESIS_HASH) + "\n" + payload
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


class Database:
    """
    SQLite by default, PostgreSQL when DATABASE_URL is set.

    WHY BOTH, RATHER THAN JUST MOVING TO POSTGRES

    The project's standing promise is that `python3 run.py` works on a demo
    laptop with nothing installed and no database server running - that is why
    the backend is standard library only, and it is the property that makes the
    offline fallback real rather than aspirational. Making Postgres mandatory
    would trade that away.

    So the engine is selected by environment: no DATABASE_URL and it is the
    same single-file SQLite as before; set one and the identical schema and
    queries run on Postgres, which is what a revenue department actually
    deploys - concurrent writers, real backups, point-in-time recovery, and
    room for PostGIS on the parcel geometry.

        DATABASE_URL=postgresql://user:pass@host:5432/landrecords

    WHAT DIFFERS, AND WHERE IT IS HANDLED

    Queries are written once, in SQLite's dialect, and translated centrally in
    q/one/run - so no caller anywhere else in the codebase knows which engine
    it is talking to:

      * placeholders   ? -> %s                     (_translate)
      * autoincrement  INTEGER PRIMARY KEY AUTOINCREMENT -> BIGSERIAL PRIMARY KEY
      * upsert         INSERT OR REPLACE -> INSERT ... ON CONFLICT DO UPDATE
      * pragmas        dropped; WAL and foreign_keys are SQLite concepts
      * lastrowid      -> RETURNING id
      * rows           sqlite3.Row and psycopg's dict rows both index by name,
                       which is why the rest of this file needed no changes

    THE ONE REAL CORRECTNESS DIFFERENCE

    The audit chain's single-writer guarantee came from a process-wide lock
    over a single SQLite connection. That is no longer sufficient with Postgres,
    where other processes - a second server instance, a psql session - can write
    concurrently and would fork the chain. On Postgres the append therefore takes
    a transaction-scoped advisory lock, which serialises writers across
    connections and is released automatically at commit. See audit().
    """

    def __init__(self, path: str = DEFAULT_DB, url: Optional[str] = None):
        self.url = url or os.environ.get("DATABASE_URL") or None
        self.is_postgres = bool(self.url)
        self.path = os.path.abspath(path)

        if self.is_postgres:
            import psycopg
            from psycopg.rows import dict_row
            self._conn = psycopg.connect(self.url, row_factory=dict_row,
                                         autocommit=False)
        else:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

        with _LOCK:
            self._apply_schema()
            self._conn.commit()
        self._migrate_audit_chain()
        self._migrate_geotag_column()
        self._seed_users()

    def _migrate_geotag_column(self) -> None:
        """
        Add documents.geotag_json to a database created before it existed.

        CREATE TABLE IF NOT EXISTS leaves an existing table untouched, so
        without this an installation with history would fail every insert on a
        missing column. Existing rows keep a NULL geotag, which is truthful -
        those documents were processed before the position was retained, and
        re-deriving one now would invent a provenance they never had. They
        pick one up on re-validation.
        """
        with _LOCK:
            if self.is_postgres:
                cur = self._conn.cursor()
                cur.execute("ALTER TABLE documents "
                            "ADD COLUMN IF NOT EXISTS geotag_json TEXT")
                cur.close()
            else:
                existing = {r["name"] for r in
                            self._conn.execute("PRAGMA table_info(documents)")}
                if "geotag_json" not in existing:
                    self._conn.execute(
                        "ALTER TABLE documents ADD COLUMN geotag_json TEXT")
            self._conn.commit()

    # -- dialect ----------------------------------------------------------
    def _translate(self, sql: str) -> str:
        """
        SQLite-dialect SQL to whatever this connection speaks.

        Only the placeholder style differs in the statements this codebase
        issues at runtime; schema-level differences are handled in
        _apply_schema, which runs once. Splitting on "'" and rewriting only
        the even-indexed segments keeps a literal question mark inside a
        string from being mangled into a placeholder.
        """
        if not self.is_postgres:
            return sql
        parts = sql.split("'")
        for i in range(0, len(parts), 2):
            parts[i] = parts[i].replace("?", "%s")
        return "'".join(parts)

    def _apply_schema(self) -> None:
        sql = SCHEMA
        if self.is_postgres:
            sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT",
                              "BIGSERIAL PRIMARY KEY")
            # PRAGMAs are SQLite's own knobs: WAL is its journal mode, and
            # Postgres enforces foreign keys unconditionally.
            sql = "\n".join(line for line in sql.splitlines()
                            if not line.strip().upper().startswith("PRAGMA"))
            cur = self._conn.cursor()
            # psycopg refuses multiple statements in one execute only for
            # prepared queries; a plain execute of the whole script is fine,
            # but splitting keeps the error message pointed at one statement.
            for statement in [s.strip() for s in sql.split(";") if s.strip()]:
                cur.execute(statement)
            cur.close()
        else:
            self._conn.executescript(sql)

    # -- schema migration -------------------------------------------------
    def _migrate_audit_chain(self) -> int:
        """
        Add the chain columns to a database created before they existed, then
        seal any unchained rows. Returns how many rows were sealed.

        CREATE TABLE IF NOT EXISTS silently leaves an existing table alone, so
        an installation that already has an audit history needs the columns
        added explicitly or every later append would fail on a missing column.
        """
        if self.is_postgres:
            # information_schema is the portable equivalent of PRAGMA
            # table_info, and ADD COLUMN IF NOT EXISTS makes the whole thing
            # idempotent without a pre-check.
            with _LOCK:
                cur = self._conn.cursor()
                for column in ("prev_hash", "entry_hash"):
                    cur.execute("ALTER TABLE audit_log "
                                f"ADD COLUMN IF NOT EXISTS {column} TEXT")
                cur.close()
                self._conn.commit()
            return self._seal_unchained()

        with _LOCK:
            existing = {r["name"] for r in
                        self._conn.execute("PRAGMA table_info(audit_log)")}
            added = False
            for column in ("prev_hash", "entry_hash"):
                if column not in existing:
                    self._conn.execute(
                        f"ALTER TABLE audit_log ADD COLUMN {column} TEXT")
                    added = True
            if added:
                self._conn.commit()
        return self._seal_unchained()

    def _seal_unchained(self) -> int:
        """
        Give every row that has no hash yet its place in the chain.

        Runs in id order so the links are built in the order the entries were
        written. A row that is already sealed is never recomputed - doing so
        would quietly repair a chain that verification is supposed to report
        as broken.
        """
        with _LOCK:
            cur = self._execute("SELECT * FROM audit_log ORDER BY id", ())
            rows = cur.fetchall()
            if self.is_postgres:
                cur.close()
            previous = None
            updates = []
            for row in rows:
                if row["entry_hash"]:
                    previous = row["entry_hash"]
                    continue
                digest = _chain_digest(previous, row)
                updates.append((previous, digest, row["id"]))
                previous = digest
            for prev_hash, entry_hash, row_id in updates:
                upd = self._execute(
                    "UPDATE audit_log SET prev_hash = ?, entry_hash = ? WHERE id = ?",
                    (prev_hash, entry_hash, row_id))
                if self.is_postgres:
                    upd.close()
            self._conn.commit()
            return len(updates)

    # -- helpers ----------------------------------------------------------
    #
    # Every query in this codebase goes through these three, which is what
    # makes one dialect enough: callers write SQLite SQL and never learn which
    # engine answered. Rows index by column name on both engines
    # (sqlite3.Row and psycopg's dict_row), so nothing downstream changed.

    def _execute(self, sql: str, params: tuple):
        sql = self._translate(sql)
        if self.is_postgres:
            cur = self._conn.cursor()
            cur.execute(sql, params)
            return cur
        return self._conn.execute(sql, params)

    def q(self, sql: str, params: tuple = ()) -> List[Any]:
        with _LOCK:
            cur = self._execute(sql, params)
            rows = cur.fetchall()
            if self.is_postgres:
                cur.close()
                # A read inside Postgres still opens a transaction; leaving it
                # idle-in-transaction would hold locks and block the next
                # writer, so close it here.
                self._conn.commit()
            return rows

    def one(self, sql: str, params: tuple = ()) -> Optional[Any]:
        rows = self.q(sql, params)
        return rows[0] if rows else None

    def run(self, sql: str, params: tuple = ()) -> int:
        with _LOCK:
            if self.is_postgres:
                # Postgres has no lastrowid. Asking for the key back is the
                # portable equivalent, and only an INSERT into a table with an
                # id has one to return.
                statement = self._translate(sql)
                wants_id = (statement.lstrip()[:6].upper() == "INSERT"
                            and "RETURNING" not in statement.upper())
                if wants_id:
                    statement += " RETURNING id"
                cur = self._conn.cursor()
                try:
                    cur.execute(statement, params)
                    row = cur.fetchone() if wants_id else None
                except Exception:
                    self._conn.rollback()
                    raise
                finally:
                    cur.close()
                self._conn.commit()
                return int(row["id"]) if row else 0
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid

    # -- users / RBAC -----------------------------------------------------
    def _seed_users(self) -> None:
        if self.one("SELECT id FROM users LIMIT 1"):
            return
        demo = [
            ("operator1", "Data Entry Operator", "operator", "Tehsil Office, Sadar"),
            ("verifier1", "Revenue Inspector", "verifier", "Tehsil Office, Sadar"),
            ("admin1", "District Land Records Officer", "admin", "Collectorate"),
            ("auditor1", "State Audit Cell", "auditor", "DILRMP State Cell"),
        ]
        for username, name, role, office in demo:
            self.run(
                "INSERT INTO users (username, full_name, role, office, created_at) "
                "VALUES (?,?,?,?,?)", (username, name, role, office, _now()))

    def get_user(self, username: str) -> Optional[sqlite3.Row]:
        return self.one("SELECT * FROM users WHERE username = ?", (username,))

    def ensure_user(self, username: str, full_name: str, role: str,
                    office: Optional[str] = None) -> sqlite3.Row:
        """
        Find a user, creating them if this is their first sign-in.

        Written as select-then-write rather than an upsert because SQLite's
        INSERT OR REPLACE and PostgreSQL's ON CONFLICT DO UPDATE are not the
        same statement and not the same semantics - REPLACE deletes the row
        and reinserts it, which would issue a new id and orphan every audit
        entry pointing at the old one. Two portable statements beat one
        dialect-specific statement that silently rewrites history.

        The role is refreshed on every sign-in so that a change made in the
        identity provider takes effect here without a manual edit.
        """
        row = self.get_user(username)
        if row is None:
            self.run(
                "INSERT INTO users (username, full_name, role, office, created_at) "
                "VALUES (?,?,?,?,?)",
                (username, full_name or username, role, office, _now()))
            return self.get_user(username)
        if row["role"] != role:
            self.run("UPDATE users SET role = ? WHERE username = ?",
                     (role, username))
            return self.get_user(username)
        return row

    def list_users(self) -> List[dict]:
        return [dict(r) for r in self.q("SELECT * FROM users ORDER BY id")]

    # -- audit ------------------------------------------------------------
    def audit(self, user: Optional[dict], action: str, document_id: Optional[int] = None,
              field_key: Optional[str] = None, old_value: Any = None,
              new_value: Any = None, detail: Optional[str] = None) -> None:
        entry = {
            "at": _now(),
            "user_id": (user or {}).get("id"),
            "username": (user or {}).get("username"),
            "role": (user or {}).get("role"),
            "action": action,
            "document_id": document_id,
            "field_key": field_key,
            "old_value": None if old_value is None else str(old_value),
            "new_value": None if new_value is None else str(new_value),
            "detail": detail,
        }
        # Reading the chain tip and appending to it must be ONE atomic step.
        # Two threads that both read the same tip would write two rows whose
        # prev_hash points at the same predecessor, forking the chain and
        # making an untampered log fail verification ever after. The server is
        # threaded, so this is a real race, not a theoretical one.
        insert = ("INSERT INTO audit_log (at, user_id, username, role, action,"
                  " document_id, field_key, old_value, new_value, detail,"
                  " prev_hash, entry_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)")
        with _LOCK:
            cur = self._conn.cursor() if self.is_postgres else self._conn
            if self.is_postgres:
                # The in-process lock above only serialises THIS process. On
                # Postgres a second server instance, a migration script or a
                # psql session can append concurrently, and two writers reading
                # the same tip would fork the chain - making an untampered log
                # fail verification permanently. A transaction-scoped advisory
                # lock serialises appenders across every connection and is
                # released automatically on commit, so a crashed writer cannot
                # wedge the audit trail.
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (_AUDIT_LOCK_KEY,))
            tip = self._execute(
                "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1",
                ()).fetchone()
            prev_hash = tip["entry_hash"] if tip else None
            entry_hash = _chain_digest(prev_hash, entry)
            params = tuple(entry[k] for k in _CHAINED_FIELDS) + (prev_hash, entry_hash)
            if self.is_postgres:
                cur.execute(self._translate(insert), params)
                cur.close()
            else:
                self._conn.execute(insert, params)
            self._conn.commit()

    def verify_audit_chain(self, limit: Optional[int] = None) -> dict:
        """
        Walk the audit log and confirm every entry still hashes to what the
        next entry expects.

        Reports the FIRST row that disagrees rather than a count, because that
        row is where the history stops being trustworthy - everything after it
        is unverifiable regardless of whether it was itself touched.
        """
        rows = self.q("SELECT * FROM audit_log ORDER BY id"
                      + (" LIMIT ?" if limit else ""),
                      (limit,) if limit else ())
        previous = None
        for row in rows:
            if (row["prev_hash"] or None) != previous:
                return {
                    "ok": False,
                    "entries": len(rows),
                    "verified": 0,
                    "broken_at": row["id"],
                    "reason": "broken_link",
                    "message": (f"Audit entry {row['id']} expects a different "
                                f"predecessor than the entry before it. A row "
                                f"was most likely deleted or reordered."),
                }
            expected = _chain_digest(previous, row)
            if row["entry_hash"] != expected:
                return {
                    "ok": False,
                    "entries": len(rows),
                    "verified": 0,
                    "broken_at": row["id"],
                    "reason": "content_altered",
                    "message": (f"Audit entry {row['id']} ('{row['action']}') no "
                                f"longer matches its recorded hash. Its content "
                                f"was changed after it was written."),
                }
            previous = row["entry_hash"]
        return {
            "ok": True,
            "entries": len(rows),
            "verified": len(rows),
            "broken_at": None,
            "reason": None,
            "tip": previous,
            "message": (f"All {len(rows)} audit entries verify against the chain."
                        if rows else "The audit log is empty."),
        }

    def audit_trail(self, document_id: Optional[int] = None, limit: int = 200) -> List[dict]:
        if document_id is not None:
            rows = self.q("SELECT * FROM audit_log WHERE document_id = ? "
                          "ORDER BY id DESC LIMIT ?", (document_id, limit))
        else:
            rows = self.q("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # -- documents --------------------------------------------------------
    @staticmethod
    def file_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def find_by_hash(self, sha: str) -> Optional[dict]:
        row = self.one("SELECT id, filename FROM documents WHERE sha256 = ?", (sha,))
        return dict(row) if row else None

    def insert_document(self, **kw) -> int:
        cols = ("filename", "stored_path", "preview_path", "sha256", "file_size", "mime",
                "uploaded_by", "ocr_engine", "page_count", "mean_ocr_conf", "legibility",
                "quality_json", "warnings_json", "full_text", "status", "decision",
                "trust_score", "error_count", "warning_count", "signature",
                "issues_json", "summary_json", "geotag_json", "processing_ms")
        values = [kw.get(c) for c in cols]
        placeholders = ",".join("?" for _ in cols)
        return self.run(
            f"INSERT INTO documents ({','.join(cols)}, uploaded_at) "
            f"VALUES ({placeholders}, ?)", tuple(values) + (_now(),))

    # Columns written by insert_field, in order. Named once because the
    # Postgres upsert has to list them again in its DO UPDATE clause.
    _FIELD_COLUMNS = ("document_id", "field_key", "display", "ai_value",
                      "ai_confidence", "conf_label", "conf_pattern", "conf_ocr",
                      "value", "status", "page", "bbox_json", "source_line",
                      "notes_json", "extra_json")

    def insert_field(self, document_id: int, f: dict) -> None:
        cb = f.get("confidence_breakdown") or {}
        params = (document_id, f["key"], f.get("display"), f.get("value"),
                  f.get("confidence"), cb.get("label"), cb.get("pattern"), cb.get("ocr"),
                  f.get("value"), f.get("status", "extracted"), f.get("page"),
                  json.dumps(f.get("bbox") or []), f.get("source_line"),
                  json.dumps(f.get("notes") or []), json.dumps(f.get("extra") or {}))
        columns = ", ".join(self._FIELD_COLUMNS)
        marks = ",".join("?" * len(self._FIELD_COLUMNS))

        if not self.is_postgres:
            self.run(f"INSERT OR REPLACE INTO fields ({columns}) VALUES ({marks})",
                     params)
            return

        # SQLite's INSERT OR REPLACE deletes the old row and inserts a new one,
        # so columns it does NOT name - corrected_by, corrected_at - go back to
        # their defaults. ON CONFLICT DO UPDATE would instead leave them
        # standing. They are reset explicitly here so re-ingesting a document
        # means the same thing on both engines: a field re-read from the page
        # is a machine value again, and no longer carries the attribution of a
        # human who corrected the version it replaced.
        assignments = ", ".join(f"{c} = EXCLUDED.{c}"
                                for c in self._FIELD_COLUMNS
                                if c not in ("document_id", "field_key"))
        self.run(
            f"INSERT INTO fields ({columns}) VALUES ({marks}) "
            f"ON CONFLICT (document_id, field_key) DO UPDATE SET {assignments}, "
            "corrected_by = NULL, corrected_at = NULL",
            params)

    def get_document(self, document_id: int) -> Optional[dict]:
        row = self.one("SELECT d.*, u.full_name AS uploader_name, "
                       "v.full_name AS verifier_name FROM documents d "
                       "LEFT JOIN users u ON u.id = d.uploaded_by "
                       "LEFT JOIN users v ON v.id = d.verified_by "
                       "WHERE d.id = ?", (document_id,))
        if not row:
            return None
        doc = dict(row)
        doc["quality"] = json.loads(doc.pop("quality_json") or "{}")
        doc["warnings"] = json.loads(doc.pop("warnings_json") or "[]")
        doc["issues"] = json.loads(doc.pop("issues_json") or "[]")
        doc["summary"] = json.loads(doc.pop("summary_json") or "{}")
        # None, not {} - "we could not place this record" and "it sits
        # at 0,0" are different claims, and the UI must be able to tell
        # them apart.
        doc["geotag"] = json.loads(doc.pop("geotag_json") or "null")
        doc["fields"] = self.get_fields(document_id)
        return doc

    def get_fields(self, document_id: int) -> List[dict]:
        rows = self.q("SELECT f.*, u.full_name AS corrector_name FROM fields f "
                      "LEFT JOIN users u ON u.id = f.corrected_by "
                      "WHERE f.document_id = ? ORDER BY f.id", (document_id,))
        out = []
        for r in rows:
            d = dict(r)
            d["bbox"] = json.loads(d.pop("bbox_json") or "[]")
            d["notes"] = json.loads(d.pop("notes_json") or "[]")
            d["extra"] = json.loads(d.pop("extra_json") or "{}")
            out.append(d)
        return out

    def field_values_map(self, document_id: int) -> Dict[str, dict]:
        """Shape expected by validator.validate()."""
        out: Dict[str, dict] = {}
        for f in self.get_fields(document_id):
            out[f["field_key"]] = {
                "value": f["value"],
                "confidence": 1.0 if f["status"] in ("confirmed", "corrected")
                              else (f["ai_confidence"] or 0.0),
                "extra": f["extra"],
            }
        return out

    def list_documents(self, status: Optional[str] = None, search: Optional[str] = None,
                       limit: int = 100, offset: int = 0) -> List[dict]:
        sql = ("SELECT d.id, d.filename, d.status, d.decision, d.trust_score, "
               "d.error_count, d.warning_count, d.uploaded_at, d.ocr_engine, "
               "d.legibility, d.mean_ocr_conf, d.page_count, d.signature, "
               "d.summary_json, u.full_name AS uploader_name, "
               "(SELECT value FROM fields WHERE document_id = d.id AND field_key='owner_name') AS owner_name, "
               "(SELECT value FROM fields WHERE document_id = d.id AND field_key='village') AS village, "
               "(SELECT value FROM fields WHERE document_id = d.id AND field_key='district') AS district, "
               "(SELECT value FROM fields WHERE document_id = d.id AND field_key='khasra_number') AS khasra_number, "
               "(SELECT value FROM fields WHERE document_id = d.id AND field_key='land_classification') AS land_classification, "
               "(SELECT value FROM fields WHERE document_id = d.id AND field_key='area') AS area "
               "FROM documents d LEFT JOIN users u ON u.id = d.uploaded_by WHERE 1=1")
        params: List[Any] = []
        if status and status != "all":
            sql += " AND d.status = ?"
            params.append(status)
        if search:
            sql += (" AND (d.filename LIKE ? OR d.full_text LIKE ? OR d.id IN "
                    "(SELECT document_id FROM fields WHERE value LIKE ?))")
            like = f"%{search}%"
            params.extend([like, like, like])
        sql += " ORDER BY d.id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        out = []
        for r in self.q(sql, tuple(params)):
            d = dict(r)
            d["summary"] = json.loads(d.pop("summary_json") or "{}")
            out.append(d)
        return out

    def existing_signatures(self, exclude_id: Optional[int] = None) -> List[dict]:
        """Feeds duplicate detection."""
        sql = ("SELECT d.id AS document_id, d.filename, d.signature, "
               "(SELECT value FROM fields WHERE document_id = d.id AND field_key='owner_name') AS owner_name "
               "FROM documents d WHERE d.signature IS NOT NULL")
        params: tuple = ()
        if exclude_id is not None:
            sql += " AND d.id != ?"
            params = (exclude_id,)
        return [dict(r) for r in self.q(sql, params)]

    # -- verification -----------------------------------------------------
    def update_field(self, document_id: int, field_key: str, new_value: Optional[str],
                     user: dict, confirm_only: bool = False) -> dict:
        row = self.one("SELECT * FROM fields WHERE document_id = ? AND field_key = ?",
                       (document_id, field_key))
        if not row:
            raise KeyError(f"Unknown field '{field_key}' on document {document_id}")
        old = row["value"]
        status = "confirmed" if confirm_only else (
            "confirmed" if (old or "") == (new_value or "") else "corrected")

        self.run("UPDATE fields SET value = ?, status = ?, corrected_by = ?, "
                 "corrected_at = ? WHERE id = ?",
                 (old if confirm_only else new_value, status, user.get("id"), _now(), row["id"]))

        self.audit(user, "field_confirmed" if status == "confirmed" else "field_corrected",
                   document_id, field_key, old,
                   old if confirm_only else new_value)

        if status == "corrected":
            doc = self.one("SELECT ocr_engine FROM documents WHERE id = ?", (document_id,))
            self.run(
                "INSERT INTO corrections (at, document_id, field_key, ai_value,"
                " human_value, ai_confidence, ocr_engine, source_line) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (_now(), document_id, field_key, row["ai_value"], new_value,
                 row["ai_confidence"], (doc or {})["ocr_engine"] if doc else None,
                 row["source_line"]))
        return {"field_key": field_key, "old_value": old,
                "new_value": old if confirm_only else new_value, "status": status}

    def set_document_validation(self, document_id: int, result: dict) -> None:
        # The geotag is rewritten here too, not only at ingestion. Revalidation
        # re-runs the geo pass, so correcting a misread village moves the
        # record to where it actually is - leaving the old position would make
        # the map disagree with the corrected record.
        self.run("UPDATE documents SET decision = ?, trust_score = ?, error_count = ?,"
                 " warning_count = ?, signature = ?, issues_json = ?, status = ?,"
                 " geotag_json = ? "
                 "WHERE id = ?",
                 (result["decision"], result["trust_score"], result["error_count"],
                  result["warning_count"], result["signature"],
                  json.dumps(result["issues"], ensure_ascii=False),
                  result["decision"] if result["decision"] != "auto_approved"
                  else "auto_approved",
                  (json.dumps(result["geotag"], ensure_ascii=False)
                   if result.get("geotag") else None),
                  document_id))

    def approve_document(self, document_id: int, user: dict) -> None:
        self.run("UPDATE documents SET status = 'approved', verified_by = ?, "
                 "verified_at = ? WHERE id = ?", (user["id"], _now(), document_id))
        self.audit(user, "document_approved", document_id,
                   detail="Record approved for publication to LRMS.")

    def reject_document(self, document_id: int, user: dict, reason: str) -> None:
        self.run("UPDATE documents SET status = 'rejected', verified_by = ?, "
                 "verified_at = ?, reject_reason = ? WHERE id = ?",
                 (user["id"], _now(), reason, document_id))
        self.audit(user, "document_rejected", document_id, detail=reason)

    # -- analytics --------------------------------------------------------
    def stats(self) -> dict:
        total = (self.one("SELECT COUNT(*) c FROM documents") or {"c": 0})["c"]

        by_status = {r["status"]: r["c"] for r in self.q(
            "SELECT status, COUNT(*) c FROM documents GROUP BY status")}

        avg = self.one("SELECT AVG(trust_score) t, AVG(legibility) l, "
                       "AVG(mean_ocr_conf) o, AVG(processing_ms) p FROM documents")

        by_district = [dict(r) for r in self.q(
            "SELECT COALESCE(f.value,'Unassigned') district, COUNT(*) c, "
            "ROUND(CAST(AVG(d.trust_score) AS NUMERIC),1) avg_trust FROM documents d "
            "LEFT JOIN fields f ON f.document_id = d.id AND f.field_key='district' "
            "GROUP BY district ORDER BY c DESC LIMIT 12")]

        by_engine = [dict(r) for r in self.q(
            "SELECT COALESCE(ocr_engine,'unknown') engine, COUNT(*) c, "
            "ROUND(CAST(AVG(trust_score) AS NUMERIC),1) avg_trust FROM documents GROUP BY engine")]

        # Field-level accuracy: how often the machine value survived review.
        field_acc = [dict(r) for r in self.q(
            "SELECT field_key, display, COUNT(*) total, "
            "SUM(CASE WHEN status='corrected' THEN 1 ELSE 0 END) corrected, "
            "SUM(CASE WHEN status='confirmed' THEN 1 ELSE 0 END) confirmed, "
            "SUM(CASE WHEN value IS NULL THEN 1 ELSE 0 END) missing, "
            "ROUND(CAST(AVG(ai_confidence) AS NUMERIC),4) avg_conf FROM fields "
            # `display` is grouped as well as selected: SQLite would happily
            # pick an arbitrary value for it, Postgres refuses. Grouping both
            # is correct on either engine - display is functionally dependent
            # on field_key, so this cannot split a row.
            "GROUP BY field_key, display ORDER BY corrected DESC")]
        for row in field_acc:
            reviewed = (row["corrected"] or 0) + (row["confirmed"] or 0)
            row["reviewed"] = reviewed
            row["precision"] = round(1.0 - (row["corrected"] or 0) / reviewed, 4) if reviewed else None

        issue_freq = {}
        for row in self.q("SELECT issues_json FROM documents WHERE issues_json IS NOT NULL"):
            for issue in json.loads(row["issues_json"] or "[]"):
                key = issue.get("rule", "UNKNOWN")
                issue_freq.setdefault(key, {"rule": key, "count": 0,
                                            "severity": issue.get("severity")})
                issue_freq[key]["count"] += 1
        top_issues = sorted(issue_freq.values(), key=lambda x: -x["count"])[:10]

        daily = [dict(r) for r in self.q(
            # "day" is a keyword to Postgres, so the alias is quoted - which
            # SQLite also accepts, and which keeps the result key unchanged for
            # the dashboard.
            'SELECT substr(uploaded_at,1,10) AS "day", COUNT(*) c FROM documents '
            'GROUP BY "day" ORDER BY "day" DESC LIMIT 14')]

        corrections_total = (self.one("SELECT COUNT(*) c FROM corrections") or {"c": 0})["c"]
        fields_reviewed = (self.one("SELECT COUNT(*) c FROM fields WHERE status IN "
                                    "('confirmed','corrected')") or {"c": 0})["c"]

        pending = (by_status.get("needs_review", 0) + by_status.get("blocked", 0))

        return {
            "documents_total": total,
            "by_status": by_status,
            "pending_verification": pending,
            "avg_trust_score": round(avg["t"], 1) if avg and avg["t"] is not None else None,
            "avg_legibility": round(avg["l"], 1) if avg and avg["l"] is not None else None,
            "avg_ocr_confidence": round(avg["o"], 4) if avg and avg["o"] is not None else None,
            "avg_processing_ms": int(avg["p"]) if avg and avg["p"] is not None else None,
            "by_district": by_district,
            "by_engine": by_engine,
            "field_accuracy": field_acc,
            "top_issues": top_issues,
            "daily_volume": list(reversed(daily)),
            "corrections_total": corrections_total,
            "fields_reviewed": fields_reviewed,
            "extraction_precision": round(1.0 - corrections_total / fields_reviewed, 4)
                                    if fields_reviewed else None,
        }

    def learning_signals(self, limit: int = 50) -> List[dict]:
        """
        The correction corpus. This is what an incremental retraining job would
        consume: (ai_value -> human_value) pairs per field, with the confidence
        the model had when it was wrong.
        """
        return [dict(r) for r in self.q(
            "SELECT field_key, ai_value, human_value, ai_confidence, ocr_engine, "
            "source_line, at FROM corrections ORDER BY id DESC LIMIT ?", (limit,))]

    def close(self) -> None:
        with _LOCK:
            self._conn.close()
