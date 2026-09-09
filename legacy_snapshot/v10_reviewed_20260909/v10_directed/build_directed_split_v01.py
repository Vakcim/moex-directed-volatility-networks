"""Freeze a coverage-only v10 split for the directed HAR benchmark.

The legacy v09 split predates the 504+126-date directed-graph warm-up.  On the
directed common sample it can therefore contain no usable training dates before
the first validation origin.  This script changes only calendar roles.  It
reads no target values, predictions, or losses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


DEVELOPMENT_END = pd.Timestamp("2026-08-10")
DEFAULT_VALIDATION_DATES = 60
MIN_WARMUP_DATES = 15
MIN_WARMUP_SECIDS = 30
MIN_DEVELOPMENT_DATES = 120


def find_legacy_split(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for rel in [
        "benchmark_v05_paired_graph_history/split_dates.csv",
        "benchmark_v05_dynamic_history_placebos/split_dates.csv",
        "benchmark_v04_placebo/split_dates.csv",
        "benchmark_v03_residual/split_dates.csv",
        "benchmark_v02_dynamic/split_dates.csv",
        "benchmark_v01/split_dates.csv",
    ]:
        path = root / rel
        if path.exists():
            return path
    raise FileNotFoundError("No legacy frozen split_dates.csv; pass --base-split")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_coverage(path: Path) -> pd.DataFrame:
    columns = ["TRADEDATE", "SECID", "directed_sample_valid"]
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns)
    return pd.read_parquet(path, columns=columns)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data_v02"))
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--base-split", type=Path, default=None)
    p.add_argument("--validation-dates", type=int, default=DEFAULT_VALIDATION_DATES)
    p.add_argument("--min-warmup-dates", type=int, default=MIN_WARMUP_DATES)
    p.add_argument("--min-warmup-secids", type=int, default=MIN_WARMUP_SECIDS)
    p.add_argument("--min-development-dates", type=int, default=MIN_DEVELOPMENT_DATES)
    a = p.parse_args()

    feature_path = a.features or (a.root / "model_data/directed_graph_features.parquet")
    base_path = find_legacy_split(a.root, a.base_split)
    # Deliberately read only dates and the precomputed coverage flag.
    features = read_coverage(feature_path)
    features["TRADEDATE"] = pd.to_datetime(features["TRADEDATE"]).dt.normalize()
    features["directed_sample_valid"] = (
        features["directed_sample_valid"].astype("boolean").fillna(False).astype(bool)
    )
    if features.duplicated(["TRADEDATE", "SECID"]).any():
        raise RuntimeError("Duplicate directed feature keys")
    by_date = features.groupby("TRADEDATE", as_index=False).agg(
        valid_rows=("directed_sample_valid", "sum")
    )
    by_date["directed_valid"] = by_date["valid_rows"] >= 30

    base = pd.read_csv(base_path)
    if not {"TRADEDATE", "split"}.issubset(base.columns):
        raise RuntimeError("Legacy split must contain TRADEDATE and split")
    base = base[["TRADEDATE", "split"]].copy()
    base["TRADEDATE"] = pd.to_datetime(base["TRADEDATE"]).dt.normalize()
    base["split"] = base["split"].astype(str).str.lower()
    if base.duplicated("TRADEDATE").any():
        raise RuntimeError("Duplicate dates in legacy split")

    panel = by_date.merge(base, on="TRADEDATE", how="left", validate="one_to_one")
    old_test_valid = panel.loc[
        panel["directed_valid"] & (panel["split"] == "test"), "TRADEDATE"
    ].sort_values()
    if len(old_test_valid) < a.validation_dates + a.min_development_dates:
        raise RuntimeError(
            "Not enough legacy-test directed-valid dates for the frozen v10 split: "
            f"available={len(old_test_valid)}, validation={a.validation_dates}, "
            f"minimum_development={a.min_development_dates}"
        )

    val_dates = old_test_valid.iloc[: a.validation_dates]
    first_val = pd.Timestamp(val_dates.iloc[0])
    last_val = pd.Timestamp(val_dates.iloc[-1])
    warmup = panel.loc[
        panel["directed_valid"] & (panel["TRADEDATE"] < first_val), "TRADEDATE"
    ].sort_values()
    development = panel.loc[
        panel["directed_valid"]
        & (panel["TRADEDATE"] > last_val)
        & (panel["TRADEDATE"] <= DEVELOPMENT_END),
        "TRADEDATE",
    ].sort_values()
    if len(warmup) < a.min_warmup_dates:
        raise RuntimeError(
            f"Directed warm-up has {len(warmup)} valid dates; "
            f"minimum is {a.min_warmup_dates}"
        )
    warmup_secids = features.loc[
        features["directed_sample_valid"] & (features["TRADEDATE"] < first_val),
        "SECID",
    ].nunique()
    if warmup_secids < a.min_warmup_secids:
        raise RuntimeError(
            f"Directed warm-up has {warmup_secids} SECIDs; "
            f"minimum is {a.min_warmup_secids}"
        )
    if len(development) < a.min_development_dates:
        raise RuntimeError(
            f"Directed development has {len(development)} valid dates; "
            f"minimum is {a.min_development_dates}"
        )

    split = panel[["TRADEDATE"]].copy()
    split["split"] = "train"
    split.loc[split["TRADEDATE"].between(first_val, last_val), "split"] = "val"
    split.loc[
        (split["TRADEDATE"] > last_val)
        & (split["TRADEDATE"] <= DEVELOPMENT_END),
        "split",
    ] = "test"
    split.loc[split["TRADEDATE"] > DEVELOPMENT_END, "split"] = "confirmatory"

    out_dir = a.root / "model_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "directed_split_dates_v10.csv"
    audit_path = out_dir / "directed_split_audit_v10.csv"
    design_path = out_dir / "directed_split_design_v10.json"
    split.to_csv(split_path, index=False)

    merged = panel[["TRADEDATE", "directed_valid", "valid_rows"]].merge(
        split, on="TRADEDATE", how="left", validate="one_to_one"
    )
    audit = (
        merged.groupby("split", as_index=False)
        .agg(
            all_dates=("TRADEDATE", "nunique"),
            valid_dates=("directed_valid", "sum"),
            valid_rows=("valid_rows", lambda s: int(s[merged.loc[s.index, "directed_valid"]].sum())),
            first_date=("TRADEDATE", "min"),
            last_date=("TRADEDATE", "max"),
        )
    )
    audit.to_csv(audit_path, index=False)
    payload = {
        "amendment": "V10 Amendment 03",
        "construction": "calendar-and-coverage-only",
        "target_prediction_or_loss_columns_read": False,
        "base_split": str(base_path),
        "base_split_sha256": sha256(base_path),
        "feature_file": str(feature_path),
        "feature_file_sha256": sha256(feature_path),
        "minimum_labels_per_valid_date": 30,
        "warmup_valid_dates": int(len(warmup)),
        "warmup_secids": int(warmup_secids),
        "validation_valid_dates": int(len(val_dates)),
        "development_valid_dates": int(len(development)),
        "first_validation_origin": str(first_val.date()),
        "last_validation_origin": str(last_val.date()),
        "development_end": str(DEVELOPMENT_END.date()),
        "split_sha256": sha256(split_path),
    }
    design_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"saved split -> {split_path}")
    print(audit.to_string(index=False))
    print(
        "directed-valid roles: "
        f"warmup={len(warmup)}, validation={len(val_dates)}, "
        f"development={len(development)}, warmup_secids={warmup_secids}"
    )


if __name__ == "__main__":
    main()
