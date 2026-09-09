from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from _parquet_stub import parquet_as_pickle
from train_directed_har_benchmark_v01 import FEATURE_SETS, assemble


def write_parts(root: Path, rows: pd.DataFrame, incomplete_model: str | None) -> None:
    parts = root / "benchmark_v10_directed_confirmatory/prediction_parts/period=confirmatory"
    for model in FEATURE_SETS:
        frame = rows.copy()
        if model == incomplete_model:
            frame = frame.iloc[:-1].copy()
        frame["model"] = model
        frame["pred_rv"] = 1.0
        target = parts / f"model={model}" / "block=0001.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)


class AssemblyCompletenessTests(unittest.TestCase):
    @staticmethod
    def _inputs(root: Path) -> tuple[pd.DataFrame, argparse.Namespace]:
        rows = pd.DataFrame({
            "TRADEDATE": pd.to_datetime(["2026-08-11", "2026-08-11"]),
            "SECID": ["SBER", "GAZP"],
        })
        data = rows.assign(split="test", y_rv=1.0)
        args = argparse.Namespace(
            root=root,
            variant="raw",
            spec_version="v10",
            period="confirmatory",
        )
        return data, args

    def test_assembly_compares_core_models_to_expected_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory, parquet_as_pickle():
            root = Path(directory)
            data, args = self._inputs(root)
            rows = data[["TRADEDATE", "SECID"]]
            write_parts(root, rows, incomplete_model="M7_catboost_directed_dynamic")
            with self.assertRaisesRegex(RuntimeError, "Incomplete or foreign"):
                assemble(args, data)

    def test_sealed_assembly_contains_no_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory, parquet_as_pickle():
            root = Path(directory)
            data, args = self._inputs(root)
            rows = data[["TRADEDATE", "SECID"]]
            write_parts(root, rows, incomplete_model=None)
            with redirect_stdout(io.StringIO()):
                assemble(args, data)
            assembled = pd.read_parquet(
                root / "benchmark_v10_directed_confirmatory/predictions_sealed.parquet"
            )
            self.assertNotIn("y_rv", assembled)
            self.assertNotIn("qlike", assembled)
