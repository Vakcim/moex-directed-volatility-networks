from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from _parquet_stub import parquet_as_pickle
from v10_integrity import (
    checkpoint_decision,
    create_once_json,
    resolve_frozen_source,
)


def keys(*rows: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["TRADEDATE", "SECID"])


class IntegrityTests(unittest.TestCase):
    def test_checkpoint_exact_keys_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory, parquet_as_pickle():
            path = Path(directory) / "part.parquet"
            expected = keys(("2026-08-11", "SBER"), ("2026-08-11", "GAZP"))
            expected.assign(pred_rv=1.0).to_parquet(path, index=False)
            self.assertEqual(checkpoint_decision(path, expected), "skip")

    def test_partial_final_block_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, parquet_as_pickle():
            path = Path(directory) / "part.parquet"
            expected = keys(("2026-08-11", "SBER"), ("2026-08-12", "SBER"))
            expected.iloc[:1].assign(pred_rv=1.0).to_parquet(path, index=False)
            self.assertEqual(checkpoint_decision(path, expected), "recompute")

    def test_checkpoint_metadata_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory, parquet_as_pickle():
            path = Path(directory) / "part.parquet"
            expected = keys(("2026-08-11", "SBER"))
            expected.assign(pred_rv=1.0, spec_version="v10").to_parquet(
                path, index=False
            )
            with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                checkpoint_decision(
                    path,
                    expected,
                    expected_metadata={"spec_version": "v10.1"},
                )

    def test_incompatible_checkpoint_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory, parquet_as_pickle():
            path = Path(directory) / "part.parquet"
            expected = keys(("2026-08-11", "SBER"))
            keys(("2026-08-11", "GAZP")).assign(pred_rv=1.0).to_parquet(path, index=False)
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                checkpoint_decision(path, expected)

    def test_frozen_json_is_create_once_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            self.assertTrue(create_once_json(path, {"alpha": 100}))
            self.assertFalse(create_once_json(path, {"alpha": 100}))
            with self.assertRaisesRegex(RuntimeError, "Never overwrite"):
                create_once_json(path, {"alpha": 10})
            self.assertEqual(json.loads(path.read_text()), {"alpha": 100})

    def test_legacy_absolute_source_path_can_be_relocated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data_v02"
            target = root / "graphs_v10" / "graph_design.json"
            target.parent.mkdir(parents=True)
            target.write_text("{}")
            recorded = str(Path(directory) / "retired-location" / "data_v02" / "graphs_v10" / "graph_design.json")
            self.assertEqual(resolve_frozen_source(root, recorded), target)
