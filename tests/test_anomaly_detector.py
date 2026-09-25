#!/usr/bin/env python3
"""
Unit tests for backend/anomaly_detector.py (one-class ML anomaly baseline).
Intelligent Land Record Digitization and Validation System - SIH 2026, PS 26018.

scikit-learn is an optional dependency for this project, following the same
pattern as fact_checker.py. Tests split into a degradation group (must pass
with or without scikit-learn) and a training/scoring group (skipped, not
failed, when scikit-learn is absent).

Run from anywhere with:
    python3 tests/test_anomaly_detector.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

import anomaly_detector as ad  # noqa: E402


def _normal_document(i=0):
    """
    A typical clean, approved record - the shape db.get_document() returns.
    Deliberately varies *every* dimension a little, not just one: verified
    that IsolationForest cannot build a meaningful decision boundary at all
    when most training features are an exact constant across every sample
    (every point - including a wildly different one - scored identically),
    which is a real sensitivity of one-class anomaly detection worth being
    aware of, not a bug in anomaly_detector.py - see its module docstring.
    Real approved documents naturally vary like this; a fixture that does
    not would test something IsolationForest cannot actually do.
    """
    return {
        "trust_score": 90.0 + (i % 8),
        "error_count": 0,
        "warning_count": i % 3,
        "summary": {"mean_field_confidence": 0.90 + (i % 7) * 0.01,
                   "completeness": 0.88 + (i % 5) * 0.02},
        "quality": {"legibility_score": 92.0 + (i % 8)},
        "issues": [
            {"rule": "SEAL_DETECTED", "severity": "info"},
            {"rule": "SIGNATURE_DETECTED", "severity": "info"},
            {"rule": "STAMP_PAPER_DETECTED", "severity": "info"},
        ],
    }


def _weird_document():
    """Low confidence, low completeness, no seal/signature, many errors."""
    return {
        "trust_score": 12.0,
        "error_count": 6,
        "warning_count": 8,
        "summary": {"mean_field_confidence": 0.31, "completeness": 0.35},
        "quality": {"legibility_score": 40.0},
        "issues": [],
    }


class FeatureExtractionTests(unittest.TestCase):
    def test_feature_vector_length_matches_names(self):
        features = ad.extract_features(_normal_document())
        self.assertEqual(len(features), len(ad.FEATURE_NAMES))

    def test_missing_fields_default_sensibly(self):
        features = ad.extract_features({})
        self.assertEqual(len(features), len(ad.FEATURE_NAMES))
        # legibility_score defaults to 100 (a clean digital PDF has no
        # image-quality metrics at all, and that should not look "bad")
        legibility_idx = ad.FEATURE_NAMES.index("legibility_score")
        self.assertEqual(features[legibility_idx], 100.0)

    def test_seal_signature_stamp_flags_from_issue_rules(self):
        doc = _normal_document()
        features = ad.extract_features(doc)
        for name in ("has_seal", "has_signature", "has_stamp_paper"):
            self.assertEqual(features[ad.FEATURE_NAMES.index(name)], 1.0)


class DegradationTests(unittest.TestCase):
    """Must hold whether or not scikit-learn is installed."""

    def test_load_model_missing_file_returns_none(self):
        self.assertIsNone(ad.load_model(path="definitely_does_not_exist.pkl"))

    @unittest.skipIf(ad.SKLEARN_AVAILABLE, "this asserts the no-sklearn degraded path")
    def test_train_without_sklearn_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            ad.train([[0.0] * len(ad.FEATURE_NAMES)] * 20)

    def test_retrain_without_sklearn_never_trains(self):
        # Without sklearn, retrain() must return before even counting
        # documents - sample_count is only meaningful once sklearn is
        # available (see the gated test below for that case).
        class FakeDB:
            def list_documents(self, status=None, limit=None):
                return [{"id": 1}, {"id": 2}]

            def get_document(self, doc_id):
                return _normal_document(doc_id)

        result = ad.retrain(FakeDB(), path="unused_never_written.pkl")
        self.assertFalse(result["trained"])
        self.assertFalse(os.path.exists("unused_never_written.pkl"))


@unittest.skipUnless(ad.SKLEARN_AVAILABLE, "scikit-learn not installed")
class TrainingAndScoringTests(unittest.TestCase):
    def test_retrain_with_too_few_documents_reports_the_real_count(self):
        class FakeDB:
            def list_documents(self, status=None, limit=None):
                return [{"id": 1}, {"id": 2}]

            def get_document(self, doc_id):
                return _normal_document(doc_id)

        result = ad.retrain(FakeDB(), path="unused_never_written.pkl")
        self.assertFalse(result["trained"])
        self.assertEqual(result["sample_count"], 2)
        self.assertFalse(os.path.exists("unused_never_written.pkl"))

    def test_train_requires_minimum_samples(self):
        too_few = [ad.extract_features(_normal_document(i)) for i in range(ad.MIN_TRAINING_SAMPLES - 1)]
        with self.assertRaises(ValueError):
            ad.train(too_few)

    def test_normal_documents_score_as_normal(self):
        # contamination=0.1 means IsolationForest always flags roughly the
        # most-extreme 10% of ANY scored set, training data included - index
        # 0 happens to sit at a corner of this fixture's cyclic pattern
        # (i%8, i%3, i%7, i%5 all zero at once) and is one of those, so a
        # comfortably mid-pack index is used instead. That is the parameter
        # working as documented, not a flaky index - see the module
        # docstring's note on contamination's real effect on training data.
        rows = [ad.extract_features(_normal_document(i)) for i in range(60)]
        model = ad.train(rows)
        is_anomaly, _ = ad.score(model, ad.extract_features(_normal_document(10)))
        self.assertFalse(is_anomaly)

    def test_wildly_different_document_scores_as_anomalous(self):
        rows = [ad.extract_features(_normal_document(i)) for i in range(60)]
        model = ad.train(rows)
        is_anomaly, _ = ad.score(model, ad.extract_features(_weird_document()))
        self.assertTrue(is_anomaly)

    def test_save_and_load_round_trip(self):
        rows = [ad.extract_features(_normal_document(i)) for i in range(60)]
        model = ad.train(rows)
        path = "test_anomaly_model.pkl"
        try:
            ad.save_model(model, path=path)
            loaded = ad.load_model(path=path)
            self.assertIsNotNone(loaded)
            is_anomaly, _ = ad.score(loaded, ad.extract_features(_weird_document()))
            self.assertTrue(is_anomaly)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_retrain_end_to_end_with_fake_db(self):
        class FakeDB:
            def list_documents(self, status=None, limit=None):
                return [{"id": i} for i in range(60)]

            def get_document(self, doc_id):
                return _normal_document(doc_id)

        path = "test_retrain_model.pkl"
        try:
            result = ad.retrain(FakeDB(), path=path)
            self.assertTrue(result["trained"])
            self.assertEqual(result["sample_count"], 60)
            self.assertTrue(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
