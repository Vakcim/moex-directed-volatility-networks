from __future__ import annotations

import pandas as pd
import unittest

from train_directed_har_benchmark_v01 import block_segment, train_rows


class TemporalGuardTests(unittest.TestCase):
    def test_training_history_is_restricted_to_score_segment(self) -> None:
        data = pd.DataFrame({
            "TRADEDATE": pd.to_datetime(["2021-01-01", "2023-01-01", "2023-01-02"]),
            "segment_id": [1, 2, 2],
            "split": ["train", "train", "test"],
        })
        train = train_rows(data, pd.Timestamp("2023-01-02"), "development", segment_id=2)
        self.assertEqual(train["segment_id"].tolist(), [2])
        self.assertLess(train["TRADEDATE"].max(), pd.Timestamp("2023-01-02"))

    def test_scoring_block_cannot_cross_segments(self) -> None:
        score = pd.DataFrame({"segment_id": [1, 2]})
        with self.assertRaisesRegex(RuntimeError, "spans temporal segments"):
            block_segment(score, "test block")
