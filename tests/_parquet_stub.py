from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd


def _read_pickle(path, columns=None, **_kwargs):
    frame = pd.read_pickle(path)
    return frame if columns is None else frame[list(columns)]


def _write_pickle(self, path, index=False, **_kwargs):
    frame = self if index else self.reset_index(drop=True)
    frame.to_pickle(path)


@contextmanager
def parquet_as_pickle():
    """Exercise parquet-facing logic without requiring pyarrow in unit tests."""
    with patch.object(pd, "read_parquet", side_effect=_read_pickle), patch.object(
        pd.DataFrame, "to_parquet", new=_write_pickle
    ):
        yield

