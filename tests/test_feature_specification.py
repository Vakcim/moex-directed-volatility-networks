from __future__ import annotations

import unittest

from train_directed_har_benchmark_v01 import (
    DYNAMIC_V10,
    DYNAMIC_V101,
    FEATURE_SETS,
    FEATURE_SETS_V101,
    stable_json_sha256,
)


class FeatureSpecificationTests(unittest.TestCase):
    def test_v101_dynamic_constructs_are_unique(self) -> None:
        self.assertEqual(len(DYNAMIC_V101), 3)
        self.assertEqual(len(DYNAMIC_V101), len(set(DYNAMIC_V101)))
        self.assertNotIn("incoming_edge_turnover", DYNAMIC_V101)
        self.assertNotIn("g_change", DYNAMIC_V101)
        self.assertIn("incoming_jaccard_distance", DYNAMIC_V101)
        self.assertIn("weighted_incoming_edge_change", DYNAMIC_V101)

    def test_historical_v10_schema_is_preserved(self) -> None:
        self.assertIn("incoming_edge_turnover", DYNAMIC_V10)
        self.assertIn("g_change", DYNAMIC_V10)
        self.assertEqual(
            len(FEATURE_SETS["M7_catboost_directed_dynamic"]),
            len(FEATURE_SETS_V101["M7_catboost_directed_dynamic"]) + 2,
        )
        self.assertNotEqual(
            stable_json_sha256(FEATURE_SETS),
            stable_json_sha256(FEATURE_SETS_V101),
        )
