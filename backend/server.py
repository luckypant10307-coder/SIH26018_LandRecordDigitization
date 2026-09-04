"""
HTTP API + ingestion pipeline.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

Standard library only (http.server + sqlite3). No pip install, no build step,
no internet. Start it with `python3 run.py` and it serves both the JSON API and
the front-end.

Role-based access control is enforced on every mutating endpoint:

  operator -> upload documents, correct fields
  verifier -> everything an operator can do, plus approve / reject records
  admin    -> everything, plus retraining and export
  auditor  -> read-only (can read the audit trail, cannot change anything)
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import shutil
import sys
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import learning
import ocr_engine
import validator as validator_mod
from db import Database
from field_extractor import (
    FIELD_BY_KEY, FIELD_SPECS, extract_fields, fields_to_dict, summarise,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, ".."))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
STORAGE_DIR = os.path.join(ROOT, "storage")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
WORK_DIR = os.path.join(STORAGE_DIR, "work")
SAMPLES_DIR = os.path.join(ROOT, "samples")

for _d in (STORAGE_DIR, UPLOAD_DIR, WORK_DIR):
    os.makedirs(_d, exist_ok=True)

DB = Database(os.path.join(STORAGE_DIR, "landrecords.db"))

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
ALLOWED_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".txt"}

ROLE_RIGHTS = {
    "operator": {"upload", "correct"},
    "verifier": {"upload", "correct", "approve", "reject", "revalidate"},
    "admin": {"upload", "correct", "approve", "reject", "revalidate", "retrain", "export", "purge"},
    "auditor": set(),
}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------
# Multipart parsing (the stdlib `cgi` module was removed in Python 3.13)
# --------------------------------------------------------------------------

def parse_multipart(body: bytes, content_type: str) -> List[dict]:
    """Minimal but correct multipart/form-data parser. Returns part dicts."""
    m = re.search(r'boundary="?([^";]+)"?', content_type or "", re.I)
    if not m:
        raise ApiError(400, "Malformed upload: no multipart boundary.")
    boundary = m.group(1).encode()
    delim = b"--" + boundary

    parts: List[dict] = []
    for raw in body.split(delim):
        if raw in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        raw = raw.lstrip(b"\r\n")
        if raw.startswith(b"--"):
            break
        head, _, payload = raw.partition(b"\r\n\r\n")
        if not _:
            continue
        headers: Dict[str, str] = {}
        for line in head.decode("utf-8", "replace").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
        disp = headers.get("content-disposition", "")
        name = re.search(r'name="([^"]*)"', disp)
        filename = re.search(r'filename="([^"]*)"', disp)
        parts.append({
            "name": name.group(1) if name else None,
            "filename": filename.group(1) if filename else None,
            "content_type": headers.get("content-type"),
            "data": payload[:-2] if payload.endswith(b"\r\n") else payload,
        })
    return parts


def safe_filename(name: str) -> str:
    name = os.path.basename(name or "document")
    name = re.sub(r"[^A-Za-z0-9._\u0900-\u097f -]", "_", name).strip() or "document"
    return name[:150]


# --------------------------------------------------------------------------
# Ingestion pipeline
# --------------------------------------------------------------------------

def process_document(stored_path: str, original_name: str, user: dict) -> dict:
    """
    The full pipeline for one document:
      extract text -> extract fields -> apply learned model -> validate
      -> persist -> route to the right queue.
    """
    started = time.time()

    sha = Database.file_hash(stored_path)
    duplicate_file = DB.find_by_hash(sha)

    doc_work = os.path.join(WORK_DIR, sha[:16])
    os.makedirs(doc_work, exist_ok=True)

    extraction = ocr_engine.extract(stored_path, work_dir=doc_work)
    fields = extract_fields(extraction.lines)

    learned = learning.apply_model(fields)

    values = {}
    for f in fields:
        d = f.to_dict()
        values[f.key] = {"value": d["value"], "confidence": d["confidence"],
                         "extra": d["extra"]}

    result = validator_mod.validate(values, existing=DB.existing_signatures())

    if duplicate_file:
        result["issues"].insert(0, {
            "rule": "FILE_ALREADY_UPLOADED", "severity": "warning", "field": None,
            "message": f"An identical file was already uploaded as document "
                       f"#{duplicate_file['id']} ({duplicate_file['filename']}).",
            "suggestion": "Check whether this is a redundant re-scan.",
        })
        result["warning_count"] += 1
        if result["decision"] == "auto_approved":
            result["decision"] = "needs_review"

    summary = summarise(fields)
    preview = extraction.render_path
    if preview and os.path.exists(preview):
        dest = os.path.join(doc_work, "preview" + os.path.splitext(preview)[1])
        if os.path.abspath(preview) != os.path.abspath(dest):
            try:
                shutil.copyfile(preview, dest)
                preview = dest
            except Exception:
                pass

    elapsed_ms = int((time.time() - started) * 1000)

    doc_id = DB.insert_document(
        filename=original_name,
        stored_path=stored_path,
        preview_path=preview,
        sha256=sha,
        file_size=os.path.getsize(stored_path),
        mime=mimetypes.guess_type(original_name)[0],
        uploaded_by=user.get("id"),
        ocr_engine=extraction.engine,
        page_count=extraction.page_count,
        mean_ocr_conf=extraction.mean_confidence,
        legibility=(extraction.quality or {}).get("legibility_score"),
        quality_json=json.dumps(extraction.quality, ensure_ascii=False),
        warnings_json=json.dumps(extraction.warnings, ensure_ascii=False),
        full_text=extraction.full_text[:200000],
        status=result["decision"],
        decision=result["decision"],
        trust_score=result["trust_score"],
        error_count=result["error_count"],
        warning_count=result["warning_count"],
        signature=result["signature"],
        issues_json=json.dumps(result["issues"], ensure_ascii=False),
        summary_json=json.dumps(summary, ensure_ascii=False),
        processing_ms=elapsed_ms,
    )

    for f in fields:
        DB.insert_field(doc_id, f.to_dict())

    DB.audit(user, "document_ingested", doc_id,
             detail=f"engine={extraction.engine}; decision={result['decision']}; "
                    f"errors={result['error_count']}; warnings={result['warning_count']}; "
                    f"{elapsed_ms}ms")
    for adj in learned:
        DB.audit(user, "learned_adjustment", doc_id, adj.get("field_key"),
                 adj.get("from"), adj.get("to"), detail=adj.get("type"))

    return {
        "document_id": doc_id,
        "filename": original_name,
        "engine": extraction.engine,
        "decision": result["decision"],
        "trust_score": result["trust_score"],
        "error_count": result["error_count"],
        "warning_count": result["warning_count"],
        "summary": summary,
        "warnings": extraction.warnings,
        "learned_adjustments": learned,
        "processing_ms": elapsed_ms,
    }


def revalidate(document_id: int, user: dict) -> dict:
    """Re-run the rule engine after human corrections."""
    values = DB.field_values_map(document_id)
    if not values:
        raise ApiError(404, f"Document {document_id} not found.")
    result = validator_mod.validate(
        values, existing=DB.existing_signatures(exclude_id=document_id))
    DB.set_document_validation(document_id, result)
    DB.audit(user, "document_revalidated", document_id,
             detail=f"decision={result['decision']}; errors={result['error_count']}")
    return result


# --------------------------------------------------------------------------
# Request handler
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "LRDVS/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing -----------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("  %s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, status: int, body: bytes, content_type: str,
              extra_headers: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data: Any, status: int = 200) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            raise ApiError(413, "Upload exceeds the 40 MB limit.")
        return self.rfile.read(length) if length else b""

    def _json_body(self) -> dict:
        raw = self._read_body()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            raise ApiError(400, "Request body is not valid JSON.")

    # -- auth ---------------------------------------------------------
    def _current_user(self) -> dict:
        username = self.headers.get("X-User") or "operator1"
        row = DB.get_user(username)
        if not row:
            raise ApiError(401, f"Unknown user '{username}'.")
        return dict(row)

    def _require(self, right: str) -> dict:
        user = self._current_user()
        if right not in ROLE_RIGHTS.get(user["role"], set()):
            raise ApiError(403, f"Role '{user['role']}' is not permitted to {right}. "
                                f"This action is restricted by role-based access control.")
        return user

    # -- routing ------------------------------------------------------
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)

        try:
            if not path.startswith("/api"):
                return self._serve_static(path)
            handler = self._route(method, path)
            if handler is None:
                raise ApiError(404, f"No API route for {method} {path}.")
            handler(query)
        except ApiError as exc:
            self._json({"error": exc.message, "status": exc.status}, exc.status)
        except BrokenPipeError:
            pass
        except Exception as exc:
            traceback.print_exc()
            self._json({"error": f"Internal error: {exc}", "status": 500}, 500)

    def _route(self, method: str, path: str):
        doc_match = re.match(r"^/api/documents/(\d+)$", path)
        sub_match = re.match(r"^/api/documents/(\d+)/([a-z]+)$", path)

        if method == "GET":
            if path == "/api/session":
                return self.api_session
            if path == "/api/documents":
                return self.api_list_documents
            if path == "/api/stats":
                return self.api_stats
            if path == "/api/audit":
                return self.api_audit
            if path == "/api/learning":
                return self.api_learning
            if path == "/api/schema":
                return self.api_schema
            if path == "/api/export/csv":
                return self.api_export_csv
            if path == "/api/export/json":
                return self.api_export_json
            if doc_match:
                return lambda q: self.api_get_document(int(doc_match.group(1)), q)
            if sub_match and sub_match.group(2) == "preview":
                return lambda q: self.api_preview(int(sub_match.group(1)), q)
            if sub_match and sub_match.group(2) == "text":
                return lambda q: self.api_text(int(sub_match.group(1)), q)

        if method == "POST":
            if path == "/api/upload":
                return self.api_upload
            if path == "/api/seed":
                return self.api_seed
            if path == "/api/learning/retrain":
                return self.api_retrain
            if path == "/api/reset":
                return self.api_reset
            if sub_match:
                doc_id, action = int(sub_match.group(1)), sub_match.group(2)
                if action == "fields":
                    return lambda q: self.api_update_field(doc_id, q)
                if action == "approve":
                    return lambda q: self.api_approve(doc_id, q)
                if action == "reject":
                    return lambda q: self.api_reject(doc_id, q)
                if action == "revalidate":
                    return lambda q: self.api_revalidate(doc_id, q)
        return None

    # -- static -------------------------------------------------------
    def _serve_static(self, path: str) -> None:
        rel = "index.html" if path == "/" else path.lstrip("/")
        target = os.path.abspath(os.path.join(FRONTEND_DIR, rel))
        if not target.startswith(os.path.abspath(FRONTEND_DIR)):
            raise ApiError(403, "Path traversal blocked.")
        if not os.path.isfile(target):
            raise ApiError(404, f"Not found: {rel}")
        ctype = mimetypes.guess_type(target)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(target, "rb") as fh:
            self._send(200, fh.read(), ctype)

    # -- API: reads ---------------------------------------------------
    def api_session(self, query: dict) -> None:
        user = self._current_user()
        self._json({
            "user": user,
            "rights": sorted(ROLE_RIGHTS.get(user["role"], set())),
            "users": DB.list_users(),
            "capabilities": ocr_engine.capabilities(),
            "admin_master_loaded": validator_mod._MASTER.loaded,
            "samples_available": sorted(
                f for f in os.listdir(SAMPLES_DIR)
                if os.path.splitext(f)[1].lower() in ALLOWED_EXT
            ) if os.path.isdir(SAMPLES_DIR) else [],
        })

    def api_schema(self, query: dict) -> None:
        self._json({"fields": [
            {"key": s.key, "display": s.display, "kind": s.kind,
             "required": s.required, "labels": s.labels}
            for s in FIELD_SPECS
        ]})

    def api_list_documents(self, query: dict) -> None:
        status = (query.get("status") or ["all"])[0]
        search = (query.get("search") or [None])[0]
        limit = min(500, int((query.get("limit") or [100])[0]))
        offset = int((query.get("offset") or [0])[0])
        self._json({"documents": DB.list_documents(status, search, limit, offset)})

    def api_get_document(self, doc_id: int, query: dict) -> None:
        doc = DB.get_document(doc_id)
        if not doc:
            raise ApiError(404, f"Document {doc_id} not found.")
        doc.pop("full_text", None)
        doc["audit"] = DB.audit_trail(doc_id, limit=60)
        self._json(doc)

    def api_text(self, doc_id: int, query: dict) -> None:
        row = DB.one("SELECT full_text FROM documents WHERE id = ?", (doc_id,))
        if not row:
            raise ApiError(404, f"Document {doc_id} not found.")
        self._json({"document_id": doc_id, "full_text": row["full_text"] or ""})

    def api_preview(self, doc_id: int, query: dict) -> None:
        row = DB.one("SELECT preview_path, stored_path FROM documents WHERE id = ?",
                     (doc_id,))
        if not row:
            raise ApiError(404, f"Document {doc_id} not found.")
        path = row["preview_path"] or row["stored_path"]
        if not path or not os.path.isfile(path):
            raise ApiError(404, "No preview available for this document.")
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def api_stats(self, query: dict) -> None:
        self._json(DB.stats())

    def api_audit(self, query: dict) -> None:
        doc_id = query.get("document_id")
        limit = min(500, int((query.get("limit") or [150])[0]))
        self._json({"entries": DB.audit_trail(
            int(doc_id[0]) if doc_id else None, limit)})

    def api_learning(self, query: dict) -> None:
        model = learning.load_model()
        self._json({
            "model": model,
            "recent_corrections": DB.learning_signals(limit=40),
            "thresholds": {
                "min_alias_support": learning.MIN_ALIAS_SUPPORT,
                "min_confusion_support": learning.MIN_CONFUSION_SUPPORT,
                "min_calibration_sample": learning.MIN_CALIBRATION_SAMPLE,
            },
        })

    # -- API: writes --------------------------------------------------
    def api_upload(self, query: dict) -> None:
        user = self._require("upload")
        ctype = self.headers.get("Content-Type") or ""
        if "multipart/form-data" not in ctype.lower():
            raise ApiError(400, "Upload must be multipart/form-data.")

        parts = parse_multipart(self._read_body(), ctype)
        files = [p for p in parts if p.get("filename")]
        if not files:
            raise ApiError(400, "No file was included in the upload.")

        results, errors = [], []
        for part in files:
            name = safe_filename(part["filename"])
            ext = os.path.splitext(name)[1].lower()
            if ext not in ALLOWED_EXT:
                errors.append({"filename": name,
                               "error": f"Unsupported file type '{ext}'."})
                continue
            if not part["data"]:
                errors.append({"filename": name, "error": "File was empty."})
                continue

            stored = os.path.join(UPLOAD_DIR, f"{int(time.time() * 1000)}_{name}")
            with open(stored, "wb") as fh:
                fh.write(part["data"])
            try:
                results.append(process_document(stored, name, user))
            except Exception as exc:
                traceback.print_exc()
                errors.append({"filename": name, "error": str(exc)})

        self._json({"processed": results, "errors": errors,
                    "count": len(results)}, 200 if results else 400)

    def api_seed(self, query: dict) -> None:
        """Ingest the bundled sample records - the one-click demo path."""
        user = self._require("upload")
        if not os.path.isdir(SAMPLES_DIR):
            raise ApiError(404, "No samples directory found.")
        names = sorted(f for f in os.listdir(SAMPLES_DIR)
                       if os.path.splitext(f)[1].lower() in ALLOWED_EXT)
        if not names:
            raise ApiError(404, "No sample documents found. Run tools/make_samples.py.")

        results, errors = [], []
        for name in names:
            src = os.path.join(SAMPLES_DIR, name)
            stored = os.path.join(UPLOAD_DIR, f"{int(time.time() * 1000)}_{name}")
            shutil.copyfile(src, stored)
            try:
                results.append(process_document(stored, name, user))
            except Exception as exc:
                traceback.print_exc()
                errors.append({"filename": name, "error": str(exc)})
        self._json({"processed": results, "errors": errors, "count": len(results)})

    def api_update_field(self, doc_id: int, query: dict) -> None:
        user = self._require("correct")
        body = self._json_body()
        field_key = body.get("field_key")
        if field_key not in FIELD_BY_KEY:
            raise ApiError(400, f"Unknown field '{field_key}'.")

        confirm = bool(body.get("confirm"))
        new_value = body.get("value")
        if new_value is not None:
            new_value = str(new_value).strip() or None

        try:
            change = DB.update_field(doc_id, field_key, new_value, user, confirm_only=confirm)
        except KeyError as exc:
            raise ApiError(404, str(exc))

        result = revalidate(doc_id, user)
        self._json({"change": change, "validation": result,
                    "fields": DB.get_fields(doc_id)})

    def api_revalidate(self, doc_id: int, query: dict) -> None:
        user = self._require("revalidate")
        self._json({"validation": revalidate(doc_id, user)})

    def api_approve(self, doc_id: int, query: dict) -> None:
        user = self._require("approve")
        doc = DB.get_document(doc_id)
        if not doc:
            raise ApiError(404, f"Document {doc_id} not found.")

        values = DB.field_values_map(doc_id)
        result = validator_mod.validate(
            values, existing=DB.existing_signatures(exclude_id=doc_id))
        if result["error_count"] > 0:
            DB.set_document_validation(doc_id, result)
            raise ApiError(409, f"Cannot approve: {result['error_count']} blocking "
                                f"error(s) remain. Resolve them first.")

        DB.approve_document(doc_id, user)
        model = learning.retrain(DB)
        self._json({"status": "approved", "document_id": doc_id,
                    "learning": {"samples": model.get("samples"),
                                 "active_rules": model.get("active_rules")}})

    def api_reject(self, doc_id: int, query: dict) -> None:
        user = self._require("reject")
        body = self._json_body()
        reason = (body.get("reason") or "").strip()
        if len(reason) < 4:
            raise ApiError(400, "A rejection reason is required for the audit trail.")
        if not DB.get_document(doc_id):
            raise ApiError(404, f"Document {doc_id} not found.")
        DB.reject_document(doc_id, user, reason)
        self._json({"status": "rejected", "document_id": doc_id})

    def api_retrain(self, query: dict) -> None:
        user = self._require("retrain")
        model = learning.retrain(DB)
        DB.audit(user, "model_retrained",
                 detail=f"samples={model['samples']}; rules={model['active_rules']}")
        self._json({"model": model})

    def api_reset(self, query: dict) -> None:
        """Clear all records. Admin only - used to re-run a clean demo."""
        user = self._require("purge")
        for table in ("corrections", "fields", "audit_log", "documents"):
            DB.run(f"DELETE FROM {table}")
        learning.save_model({"version": 1, "samples": 0, "confusions": [],
                             "aliases": [], "calibration": [], "active_rules": 0})
        DB.audit(user, "system_reset", detail="All documents and audit history cleared.")
        self._json({"status": "reset"})

    # -- API: export --------------------------------------------------
    def api_export_csv(self, query: dict) -> None:
        self._require("export")
        import csv
        keys = [s.key for s in FIELD_SPECS]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["document_id", "filename", "status", "trust_score",
                         "ocr_engine", "uploaded_at"] + keys)

        for row in DB.list_documents(limit=5000):
            fields = {f["field_key"]: f["value"] for f in DB.get_fields(row["id"])}
            writer.writerow([row["id"], row["filename"], row["status"],
                             row["trust_score"], row["ocr_engine"], row["uploaded_at"]]
                            + [fields.get(k, "") or "" for k in keys])

        self._send(200, buf.getvalue().encode("utf-8-sig"), "text/csv; charset=utf-8",
                   {"Content-Disposition": 'attachment; filename="land_records_export.csv"'})

    def api_export_json(self, query: dict) -> None:
        self._require("export")
        payload = []
        for row in DB.list_documents(limit=5000):
            doc = DB.get_document(row["id"]) or {}
            doc.pop("full_text", None)
            payload.append(doc)
        body = json.dumps({"records": payload}, ensure_ascii=False, indent=2, default=str)
        self._send(200, body.encode("utf-8"), "application/json; charset=utf-8",
                   {"Content-Disposition": 'attachment; filename="land_records_export.json"'})


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    caps = ocr_engine.capabilities()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print("=" * 72)
    print("  Intelligent Land Record Digitization and Validation System")
    print("  SIH 2026 | PS 26018 | Ministry of Rural Development (DoLR)")
    print("=" * 72)
    print(f"  Server      : http://{host}:{port}")
    print(f"  PDF text    : {'yes' if caps['pdf_text_layer'] else 'no'}")
    print(f"  Preprocess  : {'yes (OpenCV)' if caps['image_preprocessing'] else 'no'}")
    print(f"  Tesseract   : {'yes -> ' + ', '.join(caps['tesseract_languages'][:8]) if caps['tesseract'] else 'NOT INSTALLED (scanned images queue for manual entry)'}")
    print(f"  Admin master: {'loaded' if validator_mod._MASTER.loaded else 'missing'}")
    print("=" * 72)
    print("  Press Ctrl+C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    serve(port=p)
