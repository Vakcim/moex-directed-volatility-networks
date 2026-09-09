"""Verify that the stress-MAD rebuild changed no forecasting predictors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STRESS_COLUMNS = ["stress_z", "stress_threshold80", "stress_indicator"]
KEYS = ["TRADEDATE", "SECID"]


def normalized(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["TRADEDATE"] = pd.to_datetime(frame["TRADEDATE"]).dt.normalize()
    frame["SECID"] = frame["SECID"].astype(str)
    return frame.sort_values(KEYS).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_v02"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_data = args.root / "model_data"
    old_path = model_data / "directed_graph_features.parquet.pre_stress_mad_fix"
    new_path = model_data / "directed_graph_features.parquet"
    old = normalized(old_path)
    new = normalized(new_path)

    same_columns = list(old.columns) == list(new.columns)
    same_keys = old[KEYS].equals(new[KEYS])
    nonstress = [column for column in old.columns if column not in STRESS_COLUMNS]
    nonstress_identical = same_columns and same_keys and old[nonstress].equals(new[nonstress])

    predictions = pd.read_parquet(
        args.root
        / "benchmark_v10_directed_development"
        / "predictions_development.parquet",
        columns=["TRADEDATE", "model"],
    )
    prediction_dates = pd.to_datetime(
        predictions.loc[
            predictions["model"] == "M7_catboost_directed_dynamic", "TRADEDATE"
        ]
    ).dt.normalize().drop_duplicates()

    def finite_dates(frame: pd.DataFrame) -> int:
        by_date = frame[frame["TRADEDATE"].isin(prediction_dates)].groupby(
            "TRADEDATE", as_index=False
        )["stress_z"].first()
        return int(np.isfinite(by_date["stress_z"]).sum())

    audit = pd.DataFrame([
        {"metric": "same_columns", "value": bool(same_columns), "passed": bool(same_columns)},
        {"metric": "same_date_secid_keys", "value": bool(same_keys), "passed": bool(same_keys)},
        {
            "metric": "nonstress_columns_identical",
            "value": bool(nonstress_identical),
            "passed": bool(nonstress_identical),
        },
        {
            "metric": "old_finite_development_stress_dates",
            "value": finite_dates(old),
            "passed": True,
        },
        {
            "metric": "new_finite_development_stress_dates",
            "value": finite_dates(new),
            "passed": True,
        },
    ])
    output = args.root / "model_audit" / "stress_mad_fix_audit.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output, index=False)
    print(audit.to_string(index=False))
    if not (same_columns and same_keys and nonstress_identical):
        raise RuntimeError("Stress rebuild changed keys or non-stress model inputs")


if __name__ == "__main__":
    main()
