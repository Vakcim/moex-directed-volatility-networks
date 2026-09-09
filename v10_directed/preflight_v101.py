"""Outcome-blind readiness checks for prospective v10.1.

This script reads only feature aliases, split labels, frozen design metadata and
prediction schemas/keys.  It never reads target or loss columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_directed_har_benchmark_v01 import (
    FEATURE_SETS_V101,
    confirmatory_dir,
    development_dir,
    frozen_parameters,
)
from v10_integrity import stable_json_sha256, validate_reproducibility_manifest
ALIASES = [
    ("incoming_jaccard_distance", "incoming_edge_turnover"),
    ("weighted_incoming_edge_change", "g_change"),
]
PREFIXES = ["pearson", "linear", "catboost"]
FORBIDDEN_OUTCOME_COLUMNS = {
    "y_rv",
    "y_log_rv",
    "qlike",
    "squared_log_error",
    "absolute_log_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_v02"))
    parser.add_argument(
        "--expect",
        choices=["selection", "freeze", "holdout"],
        default="selection",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def check_aliases(root: Path) -> dict:
    path = root / "model_data/directed_graph_features.parquet"
    columns = [f"{prefix}_{suffix}" for prefix in PREFIXES for pair in ALIASES for suffix in pair]
    frame = pd.read_parquet(path, columns=columns)
    rows = []
    for prefix in PREFIXES:
        for left_suffix, right_suffix in ALIASES:
            left, right = f"{prefix}_{left_suffix}", f"{prefix}_{right_suffix}"
            finite = np.isfinite(frame[left]) & np.isfinite(frame[right])
            equal = bool(np.allclose(frame.loc[finite, left], frame.loc[finite, right], rtol=0, atol=0))
            same_missing = bool(frame[left].isna().equals(frame[right].isna()))
            rows.append({
                "left": left,
                "right": right,
                "finite_rows": int(finite.sum()),
                "exact_equal": equal and same_missing,
            })
    return {"path": str(path), "comparisons": rows, "passed": all(r["exact_equal"] for r in rows)}


def check_selection(root: Path) -> dict:
    dev = development_dir(root, "raw", "v10.1")
    path = dev / "hyperparameter_selection.csv"
    if not path.exists():
        return {"path": str(path), "passed": False, "reason": "missing"}
    table = pd.read_csv(
        path, usecols=["model", "selected", "feature_schema_sha256"]
    )
    selected = table[table["selected"].astype(str).str.lower().isin(["true", "1"])]
    models = set(selected["model"].astype(str))
    expected = set(FEATURE_SETS_V101)
    expected_hash = stable_json_sha256(FEATURE_SETS_V101)
    observed_hashes = set(table["feature_schema_sha256"].astype(str))
    return {
        "path": str(path),
        "selected_models": sorted(models),
        "expected_models": sorted(expected),
        "feature_schema_sha256": sorted(observed_hashes),
        "passed": (
            models == expected
            and len(selected) == len(expected)
            and observed_hashes == {expected_hash}
        ),
    }


def check_split(root: Path) -> dict:
    path = root / "model_audit/directed_split_dates_v10.csv"
    feature_path = root / "model_data/directed_graph_features.parquet"
    split = pd.read_csv(path, usecols=["TRADEDATE", "split"])
    split["TRADEDATE"] = pd.to_datetime(
        split["TRADEDATE"], errors="coerce"
    ).dt.normalize()
    split["split"] = split["split"].astype(str).str.lower()
    if split["TRADEDATE"].isna().any() or split.duplicated("TRADEDATE").any():
        raise RuntimeError("Invalid or duplicate dates in directed split")

    # Amendment 03 assigns a calendar interval to validation, so that interval
    # can contain more than 60 dates.  The prespecified count is 60
    # directed-valid origins, defined without outcomes as at least 30 rows with
    # the precomputed coverage flag.  Repeat the split-builder rule here rather
    # than counting every calendar date carrying the inherited label.
    coverage = pd.read_parquet(
        feature_path,
        columns=["TRADEDATE", "SECID", "directed_sample_valid"],
    )
    coverage["TRADEDATE"] = pd.to_datetime(
        coverage["TRADEDATE"], errors="coerce"
    ).dt.normalize()
    if coverage[["TRADEDATE", "SECID"]].duplicated().any():
        raise RuntimeError("Duplicate directed feature keys")
    coverage["directed_sample_valid"] = (
        coverage["directed_sample_valid"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    by_date = coverage.groupby("TRADEDATE", as_index=False).agg(
        valid_rows=("directed_sample_valid", "sum")
    )
    by_date["directed_valid"] = by_date["valid_rows"] >= 30
    joined = split.merge(by_date, on="TRADEDATE", how="left", validate="one_to_one")
    joined["directed_valid"] = joined["directed_valid"].fillna(False).astype(bool)

    calendar_counts = split.groupby("split")["TRADEDATE"].nunique().to_dict()
    valid_counts = (
        joined.loc[joined["directed_valid"]]
        .groupby("split")["TRADEDATE"]
        .nunique()
        .to_dict()
    )
    validation_dates = int(valid_counts.get("val", 0))
    return {
        "path": str(path),
        "coverage_path": str(feature_path),
        "calendar_validation_dates": int(calendar_counts.get("val", 0)),
        "validation_dates": validation_dates,
        "minimum_valid_rows_per_date": 30,
        "passed": validation_dates == 60,
    }


def check_freeze(root: Path) -> dict:
    out = confirmatory_dir(root, "v10.1")
    try:
        _, _, _, digest = frozen_parameters(root, "v10.1")
        manifest_path = out / "frozen_manifest_v10_1_v02.json"
        manifest_digest = validate_reproducibility_manifest(
            root,
            Path(__file__).resolve().parents[1],
            manifest_path,
            spec_version="v10.1",
        )
    except Exception as exc:
        return {"path": str(out), "passed": False, "reason": str(exc)}
    return {
        "path": str(out),
        "frozen_hyperparameters_sha256": digest,
        "reproducibility_manifest_sha256": manifest_digest,
        "passed": True,
    }


def check_sealed_schema(root: Path) -> dict:
    import pyarrow.parquet as pq

    path = confirmatory_dir(root, "v10.1") / "predictions_sealed.parquet"
    if not path.exists():
        return {"path": str(path), "passed": True, "status": "not_started"}
    schema_names = set(pq.read_schema(path).names)
    forbidden = sorted(schema_names & FORBIDDEN_OUTCOME_COLUMNS)
    key_columns = ["TRADEDATE", "SECID", "model"]
    keys = pd.read_parquet(path, columns=key_columns)
    duplicate_keys = int(keys.duplicated(key_columns).sum())
    return {
        "path": str(path),
        "rows": len(keys),
        "dates": int(pd.to_datetime(keys["TRADEDATE"]).nunique()),
        "duplicate_keys": duplicate_keys,
        "forbidden_columns": forbidden,
        "passed": not forbidden and duplicate_keys == 0,
    }


def main() -> None:
    args = parse_args()
    checks = {
        "split": check_split(args.root),
        "feature_alias_evidence": check_aliases(args.root),
    }
    if args.expect in {"freeze", "holdout"}:
        checks["selection"] = check_selection(args.root)
        checks["freeze"] = check_freeze(args.root)
    if args.expect == "holdout":
        checks["sealed_schema"] = check_sealed_schema(args.root)
    payload = {
        "spec_version": "v10.1",
        "expect": args.expect,
        "target_or_loss_columns_read": False,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
