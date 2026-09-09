from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from _parquet_stub import parquet_as_pickle
from download_imoex_10m_v01 import checkpoint_reusable


class DownloaderCheckpointTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, begin: str) -> None:
        pd.DataFrame({
            "begin": pd.to_datetime([begin]),
            "open": [100.0],
            "close": [101.0],
        }).to_parquet(path, index=False)

    def test_final_requested_month_refreshes_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory, parquet_as_pickle():
            path = Path(directory) / "2026-09.parquet"
            self._write(path, "2026-09-01 10:00")
            self.assertFalse(checkpoint_reusable(
                path,
                pd.Timestamp("2026-09-01"),
                pd.Timestamp("2026-09-09"),
                is_final_requested_month=True,
                reuse_final_month=False,
            ))
            self.assertTrue(checkpoint_reusable(
                path,
                pd.Timestamp("2026-09-01"),
                pd.Timestamp("2026-09-09"),
                is_final_requested_month=True,
                reuse_final_month=True,
            ))

    def test_closed_nonfinal_month_reuses_valid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory, parquet_as_pickle():
            path = Path(directory) / "2026-08.parquet"
            self._write(path, "2026-08-03 10:00")
            self.assertTrue(checkpoint_reusable(
                path,
                pd.Timestamp("2026-08-01"),
                pd.Timestamp("2026-08-31"),
                is_final_requested_month=False,
                reuse_final_month=False,
            ))

    def test_narrow_old_checkpoint_is_not_reused_for_broader_month(self) -> None:
        with tempfile.TemporaryDirectory() as directory, parquet_as_pickle():
            path = Path(directory) / "2026-08.parquet"
            self._write(path, "2026-08-10 10:00")
            self.assertFalse(checkpoint_reusable(
                path,
                pd.Timestamp("2026-08-01"),
                pd.Timestamp("2026-08-31"),
                is_final_requested_month=False,
                reuse_final_month=False,
                expected_dates=pd.DatetimeIndex(["2026-08-03", "2026-08-31"]),
            ))
