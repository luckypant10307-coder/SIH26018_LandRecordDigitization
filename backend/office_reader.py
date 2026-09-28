"""
Office and web document readers, standard library only.
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

WHY THIS EXISTS

Before this, an upload had to be a PDF, an image or plain text. A revenue
office does not only hold scans: a tehsil clerk's khatauni extract is very
often a .docx, a district's parcel list is an .xlsx, and offices running
LibreOffice - which most Indian government installations do - produce .odt and
.ods. Refusing those forced the one thing this project exists to avoid,
re-typing a record by hand.

WHY NOT DOCLING, WHICH IS THE OBVIOUS ANSWER

Docling handles all of these and more, and it is genuinely good. It also pulls
easyocr, ONNX Runtime and rapidocr as declared dependencies - an entire second
OCR stack beside the Tesseract pipeline this project already measures at 99.3%
precision, and enough weight to put the deployment image past a free tier's
limit. It solves layout analysis for complex PDFs, which is a problem this
project has already solved by other means.

The formats added here do not need any of that. DOCX, XLSX, PPTX, ODT and ODS
are ZIP archives full of XML, and they carry REAL TEXT rather than pixels -
there is nothing to recognise, only something to read. `zipfile` and
`xml.etree` are in the standard library, so this costs no dependency, no
download and no deployment weight, and it works on a laptop with no network.

The structuring is unchanged: text comes out of here and
field_extractor.py's label-anchored pass turns it into the 17 fields, exactly
as it does for a PDF text layer. That is the part that actually matters, and
it is already built.

WHAT IS DELIBERATELY NOT CLAIMED

A spreadsheet is not a document. Cells are emitted row by row as
"header: value" pairs where a header row can be identified, because that is
the shape the label-anchored extractor reads. A genuinely tabular parcel
register with fifty rows will extract the FIRST record and flag the rest as
unread rather than silently merging fifty parcels into one - see
MULTI_ROW_WARNING.
"""

from __future__ import annotations

import csv
import io
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from typing import List, Optional, Tuple

# Extensions this module can read, mapped to the engine name recorded on the
# document so the dashboard can report what actually read it.
READABLE = {
    ".docx": "word_docx",
    ".odt": "opendocument_text",
    ".xlsx": "excel_xlsx",
    ".ods": "opendocument_sheet",
    ".pptx": "powerpoint_pptx",
    ".csv": "delimited_text",
    ".tsv": "delimited_text",
    ".html": "html",
    ".htm": "html",
    ".rtf": "rich_text",
}

MULTI_ROW_WARNING = (
    "This file holds {rows} data rows. A land record is one parcel, so only "
    "the first row was extracted into the 17 fields; the remaining {rest} "
    "were read but not structured. Split the file, or upload one record per "
    "file, to digitise them all.")

# Office XML namespaces. Declared rather than matched loosely because a
# namespace collision would silently pull the wrong element.
_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
}


def can_read(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in READABLE


def engine_for(path: str) -> Optional[str]:
    return READABLE.get(os.path.splitext(path)[1].lower())


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Collapse the whitespace Office XML scatters through a run of text."""
    return re.sub(r"[ \t\u00a0]+", " ", (text or "").replace("\r", " ")).strip()


def _zip_xml(path: str, member: str):
    """Parse one XML member of a ZIP container, or None when it is absent."""
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as handle:
                return ET.parse(handle).getroot()
    except Exception:
        return None


def _rows_to_lines(rows: List[List[str]]) -> Tuple[List[str], Optional[str]]:
    """
    Turn tabular rows into label-anchored lines the field extractor can read.

    A spreadsheet's first row is nearly always a header. Pairing each header
    with its cell reproduces the "Khasra No : 213/1" shape the extractor was
    built for, which is why a table can be read at all without a table model.

    Returns (lines, warning). The warning is populated when more than one data
    row exists, because a land record is ONE parcel and silently merging fifty
    of them would be a confident wrong answer.
    """
    rows = [r for r in rows if any(_clean(c) for c in r)]
    if not rows:
        return [], None
    if len(rows) == 1:
        return [" ".join(_clean(c) for c in rows[0] if _clean(c))], None

    header = [_clean(c) for c in rows[0]]
    body = rows[1:]
    lines: List[str] = []
    first = body[0]
    for index, cell in enumerate(first):
        value = _clean(cell)
        if not value:
            continue
        label = header[index] if index < len(header) else ""
        lines.append(f"{label} : {value}" if label else value)

    warning = None
    if len(body) > 1:
        warning = MULTI_ROW_WARNING.format(rows=len(body), rest=len(body) - 1)
        # The unstructured remainder is still emitted so the raw text of the
        # document is complete and searchable, and so a reviewer can see what
        # was not structured rather than having to guess.
        for row in body[1:]:
            joined = " | ".join(_clean(c) for c in row if _clean(c))
            if joined:
                lines.append(joined)
    return lines, warning


# --------------------------------------------------------------------------
# Word / OpenDocument text
# --------------------------------------------------------------------------

def _read_docx(path: str) -> Tuple[List[str], Optional[str]]:
    root = _zip_xml(path, "word/document.xml")
    if root is None:
        return [], None
    lines: List[str] = []
    for paragraph in root.iter(f"{{{_NS['w']}}}p"):
        text = "".join(node.text or "" for node in
                       paragraph.iter(f"{{{_NS['w']}}}t"))
        text = _clean(text)
        if text:
            lines.append(text)
    # A khatauni in Word is usually a TABLE, and a table cell's paragraphs are
    # already captured above - but flattened, losing the row pairing. Rebuild
    # the rows so a label in column 1 stays attached to its value in column 2.
    for table in root.iter(f"{{{_NS['w']}}}tbl"):
        for row in table.iter(f"{{{_NS['w']}}}tr"):
            cells = []
            for cell in row.iter(f"{{{_NS['w']}}}tc"):
                cells.append(_clean("".join(
                    node.text or "" for node in cell.iter(f"{{{_NS['w']}}}t"))))
            cells = [c for c in cells if c]
            if len(cells) >= 2:
                lines.append(f"{cells[0]} : {' '.join(cells[1:])}")
    return lines, None


def _read_odt(path: str) -> Tuple[List[str], Optional[str]]:
    root = _zip_xml(path, "content.xml")
    if root is None:
        return [], None
    lines = []
    for tag in ("p", "h"):
        for node in root.iter(f"{{{_NS['text']}}}{tag}"):
            text = _clean("".join(node.itertext()))
            if text:
                lines.append(text)
    for row in root.iter(f"{{{_NS['table']}}}table-row"):
        cells = [_clean("".join(c.itertext()))
                 for c in row.iter(f"{{{_NS['table']}}}table-cell")]
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            lines.append(f"{cells[0]} : {' '.join(cells[1:])}")
    return lines, None


# --------------------------------------------------------------------------
# Spreadsheets
# --------------------------------------------------------------------------

def _read_xlsx(path: str) -> Tuple[List[str], Optional[str]]:
    """
    Read the first worksheet without openpyxl.

    xlsx stores strings in a shared table and references them by index from
    each cell, so the table has to be read first or every text cell comes back
    as a number.
    """
    shared: List[str] = []
    shared_root = _zip_xml(path, "xl/sharedStrings.xml")
    if shared_root is not None:
        for item in shared_root.iter(f"{{{_NS['s']}}}si"):
            shared.append(_clean("".join(item.itertext())))

    sheet = _zip_xml(path, "xl/worksheets/sheet1.xml")
    if sheet is None:
        return [], None

    rows: List[List[str]] = []
    for row in sheet.iter(f"{{{_NS['s']}}}row"):
        cells: List[str] = []
        for cell in row.iter(f"{{{_NS['s']}}}c"):
            value_node = cell.find(f"{{{_NS['s']}}}v")
            inline = cell.find(f"{{{_NS['s']}}}is")
            if inline is not None:
                cells.append(_clean("".join(inline.itertext())))
            elif value_node is None or value_node.text is None:
                cells.append("")
            elif cell.get("t") == "s":
                index = int(value_node.text)
                cells.append(shared[index] if 0 <= index < len(shared) else "")
            else:
                cells.append(_clean(value_node.text))
        rows.append(cells)
    return _rows_to_lines(rows)


def _read_ods(path: str) -> Tuple[List[str], Optional[str]]:
    root = _zip_xml(path, "content.xml")
    if root is None:
        return [], None
    rows = []
    for row in root.iter(f"{{{_NS['table']}}}table-row"):
        cells = [_clean("".join(c.itertext()))
                 for c in row.iter(f"{{{_NS['table']}}}table-cell")]
        rows.append(cells)
    return _rows_to_lines(rows)


# --------------------------------------------------------------------------
# Slides, delimited text, HTML, RTF
# --------------------------------------------------------------------------

def _read_pptx(path: str) -> Tuple[List[str], Optional[str]]:
    lines: List[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            slides = sorted(n for n in archive.namelist()
                            if re.match(r"ppt/slides/slide\d+\.xml$", n))
            for name in slides:
                with archive.open(name) as handle:
                    root = ET.parse(handle).getroot()
                for node in root.iter(f"{{{_NS['a']}}}p"):
                    text = _clean("".join(t.text or "" for t in
                                          node.iter(f"{{{_NS['a']}}}t")))
                    if text:
                        lines.append(text)
    except Exception:
        return [], None
    return lines, None


def _read_delimited(path: str) -> Tuple[List[str], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            sample = handle.read(8192)
            handle.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            except Exception:
                dialect = csv.excel
            rows = [row for row in csv.reader(handle, dialect)]
    except Exception:
        return [], None
    return _rows_to_lines(rows)


class _TextHTML(HTMLParser):
    """Strip markup, keeping block boundaries so labels stay on their own line."""

    BLOCKS = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")
        elif tag == "td" or tag == "th":
            self.parts.append(" : ")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _read_html(path: str) -> Tuple[List[str], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            parser = _TextHTML()
            parser.feed(handle.read())
    except Exception:
        return [], None
    text = "".join(parser.parts)
    lines = [_clean(t).strip(" :") for t in text.splitlines()]
    return [t for t in lines if t], None


def _read_rtf(path: str) -> Tuple[List[str], Optional[str]]:
    """
    Minimal RTF de-control. Not a parser - RTF is a large format and this
    reads only the plain text a revenue office's exported record contains.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            body = handle.read()
    except Exception:
        return [], None
    body = re.sub(r"\\par[d]?", "\n", body)
    body = re.sub(r"\\'([0-9a-fA-F]{2})",
                  lambda m: bytes([int(m.group(1), 16)]).decode("cp1252", "replace"),
                  body)
    body = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", body)
    body = body.replace("{", "").replace("}", "")
    return [_clean(t) for t in body.splitlines() if _clean(t)], None


_READERS = {
    ".docx": _read_docx, ".odt": _read_odt,
    ".xlsx": _read_xlsx, ".ods": _read_ods,
    ".pptx": _read_pptx,
    ".csv": _read_delimited, ".tsv": _read_delimited,
    ".html": _read_html, ".htm": _read_html,
    ".rtf": _read_rtf,
}


def read(path: str) -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    Extract text lines from an office or web document.

    Returns (lines, engine, warning). Never raises: a malformed or
    password-protected file yields no lines and the caller reports it as
    unreadable, which routes the document to manual entry rather than
    recording an empty success.
    """
    extension = os.path.splitext(path)[1].lower()
    reader = _READERS.get(extension)
    if reader is None:
        return [], None, None
    try:
        lines, warning = reader(path)
    except Exception as exc:
        return [], READABLE.get(extension), f"{type(exc).__name__}: {exc}"
    return lines, READABLE.get(extension), warning


def capabilities() -> dict:
    return {
        "formats": sorted(READABLE),
        "count": len(READABLE),
        "dependency_free": True,
    }
