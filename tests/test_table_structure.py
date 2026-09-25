#!/usr/bin/env python3
"""
Unit tests for backend/table_structure.py (PP-StructureV3 table/layout
recognition).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

PaddleOCR/PP-StructureV3 is deliberately NOT exercised end-to-end in this
suite, even when installed: unlike scikit-learn or spaCy, constructing the
real pipeline downloads/loads hundreds of MB of model weights and takes
seconds-to-tens-of-seconds per page on CPU. Running that on every test-suite
invocation would defeat the project's "the test suite is always fast" habit.
What IS tested here, unconditionally: the module never raises regardless of
whether the library is present, and its defensive parsing of the PP-StructureV3
result object (which this project does not control the exact shape of) handles
both the documented attribute-style access and a plain-dict fallback.

Run from anywhere with:
    python3 tests/test_table_structure.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import table_structure as ts  # noqa: E402


class _FakeRes:
    """Stand-in for a PP-StructureV3 result object, attribute-style access."""
    def __init__(self, markdown=None, layout_det_res=None):
        self.markdown = markdown
        self.layout_det_res = layout_det_res


class _FakeBox:
    def __init__(self, label):
        self.label = label


class MarkdownExtractionTests(unittest.TestCase):
    def test_plain_string_markdown(self):
        res = _FakeRes(markdown="# Doc\n<table><tr><td>1</td></tr></table>")
        self.assertIn("<table>", ts._markdown_text(res))

    def test_dict_markdown_texts_key(self):
        res = _FakeRes(markdown={"markdown_texts": "hello world"})
        self.assertEqual(ts._markdown_text(res), "hello world")

    def test_dict_markdown_key_fallback(self):
        res = _FakeRes(markdown={"markdown": "fallback text"})
        self.assertEqual(ts._markdown_text(res), "fallback text")

    def test_missing_markdown_is_empty_string(self):
        res = _FakeRes(markdown=None)
        self.assertEqual(ts._markdown_text(res), "")

    def test_no_markdown_attribute_at_all(self):
        class Bare:
            pass
        self.assertEqual(ts._markdown_text(Bare()), "")


class LayoutLabelCountTests(unittest.TestCase):
    def test_attribute_style_boxes(self):
        res = _FakeRes(layout_det_res=type("L", (), {"boxes": [_FakeBox("table"), _FakeBox("text")]})())
        counts = ts._layout_label_counts(res)
        self.assertEqual(counts.get("table"), 1)
        self.assertEqual(counts.get("text"), 1)

    def test_dict_style_boxes(self):
        res = _FakeRes(layout_det_res={"boxes": [{"label": "table"}, {"label": "table"}, {"label": "seal"}]})
        counts = ts._layout_label_counts(res)
        self.assertEqual(counts.get("table"), 2)
        self.assertEqual(counts.get("seal"), 1)

    def test_missing_layout_det_res_is_empty(self):
        res = _FakeRes(layout_det_res=None)
        self.assertEqual(ts._layout_label_counts(res), {})

    def test_boxes_without_label_are_skipped(self):
        res = _FakeRes(layout_det_res={"boxes": [{}, {"label": "text"}]})
        counts = ts._layout_label_counts(res)
        self.assertEqual(counts, {"text": 1})


class DegradationTests(unittest.TestCase):
    """Must pass regardless of whether paddleocr is actually installed."""

    def test_analyze_never_raises_on_a_nonexistent_path(self):
        with tempfile.TemporaryDirectory() as work_dir:
            result = ts.analyze("Z:\\does\\not\\exist.png", work_dir)
            self.assertIsInstance(result.available, bool)

    def test_unavailable_flag_matches_import_probe(self):
        # Whatever this machine's real state is, the module's own flag must
        # agree with a fresh import attempt - no drift between the two.
        try:
            __import__("paddleocr")
            expected = True
        except Exception:
            expected = False
        self.assertEqual(ts.PADDLE_STRUCTURE_AVAILABLE, expected)

    def test_result_when_unavailable_reports_no_tables(self):
        if ts.PADDLE_STRUCTURE_AVAILABLE:
            self.skipTest("paddleocr is installed on this machine - "
                          "unavailable-path assertions do not apply")
        with tempfile.TemporaryDirectory() as work_dir:
            result = ts.analyze(os.path.join(work_dir, "whatever.png"), work_dir)
            self.assertFalse(result.available)
            self.assertEqual(result.table_count, 0)
            self.assertEqual(result.tables_html, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
