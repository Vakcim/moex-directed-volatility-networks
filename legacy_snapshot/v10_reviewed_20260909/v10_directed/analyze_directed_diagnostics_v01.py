"""Exploratory v10 diagnostics: M8 comparisons and RNFT ridge failure audit.

These analyses are deliberately separate from the frozen H1/H2 family.  They
must be described as exploratory and do not replace primary QLIKE results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_directed_results_v01 import add_losses, inference_table
from train_directed_har_benchmark_v01 import (
    FEATURE_SETS,
    date_blocks,
    fit_ridge,
    load_data,
    score_dates,
    selected_alphas,
    train_rows,
)
from v10_common import EPS, holm, normalize_keys, read_table


M8_COMPARISONS = [
    ("D1_m8_vs_har_market", "M1_har_market", "M8_direct_catboost"),
    ("D2_m8_vs_har_rv", "M0_har_rv", "M8_direct_catboost"),
    ("D3_m8_vs_directed_dynamic", "M7_catboost_directed_dynamic", "M8_direct_catboost"),
]
RIDGE_MODELS = ["M4_linear_directed_current", "M5_linear_directed_dynamic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_v02"))
    return parser.parse_args()


def load_development(root: Path) -> tuple[pd.DataFrame, Path]:
    output = root / "benchmark_v10_directed_development"
    predictions = normalize_keys(read_table(output / "predictions_development.parquet"))
    return add_losses(predictions), output


def m8_diagnostics(predictions: pd.DataFrame, output: Path) -> None:
    available = set(predictions["model"].astype(str))
    missing = sorted({model for _, left, right in M8_COMPARISONS for model in (left, right)} - available)
    if missing:
        raise RuntimeError(f"M8 diagnostic models missing: {missing}")
    result, daily, audit = inference_table(predictions, M8_COMPARISONS, "m8_exploratory")
    for column in (
        "hac_p_two_sided",
        "hac_p_one_sided",
        "bootstrap_p_two_sided",
        "bootstrap_p_one_sided",
    ):
        result[f"holm_{column}"] = holm(result[column])
    result["analysis_status"] = "exploratory_post_development"
    result.to_csv(output / "paired_inference_m8_exploratory.csv", index=False)
    daily.to_csv(output / "daily_loss_differentials_m8_exploratory.csv", index=False)
    audit.assign(analysis_status="exploratory_post_development").to_csv(
        output / "fairness_audit_m8_exploratory.csv", index=False
    )
    print("\n===== EXPLORATORY M8 PAIRED INFERENCE =====")
    print(result.to_string(index=False))


def reconstruct_ridge_failures(
    root: Path, predictions: pd.DataFrame, output: Path
) -> pd.DataFrame:
    floor = predictions[
        predictions["model"].isin(RIDGE_MODELS)
        & (predictions["pred_rv"] <= EPS * (1 + 1e-9))
    ].copy()
    floor = floor.sort_values(["model", "TRADEDATE", "SECID"])
    floor.to_csv(output / "ridge_floor_failures.csv", index=False)
    if floor.empty:
        pd.DataFrame().to_csv(output / "ridge_floor_feature_contributions.csv", index=False)
        print("\nNo exact ridge prediction-floor failures found.")
        return floor

    args = argparse.Namespace(root=root, variant="raw", features=None, split_dates=None)
    data, _ = load_data(args)
    alphas = selected_alphas(root, "raw")
    development_dates = score_dates(data, "development")
    blocks = {index: values for index, values in enumerate(date_blocks(development_dates, 20), 1)}
    details: list[dict] = []

    for (model, block_id), incidents in floor.groupby(["model", "block_id"], sort=True):
        block_id = int(block_id)
        if block_id not in blocks:
            raise RuntimeError(f"Unknown development block {block_id}")
        block = blocks[block_id]
        train = train_rows(data, block[0], "development")
        features = FEATURE_SETS[str(model)]
        fitted = fit_ridge(train, features, alphas[str(model)])
        keys = incidents[["TRADEDATE", "SECID", "pred_rv", "y_rv", "qlike"]]
        score = data.merge(keys, on=["TRADEDATE", "SECID"], how="inner", validate="one_to_one", suffixes=("", "_saved"))
        z = (score[features].to_numpy(float) - fitted.mean) / fitted.scale
        contribution = z * fitted.beta
        intercept = np.array([fitted.intercepts.get(str(s), fitted.fallback) for s in score["SECID"]])
        raw_log = intercept + contribution.sum(axis=1)
        reconstructed = np.maximum(np.exp(np.clip(raw_log, -30, 5)) * fitted.rv_scale, EPS)
        if not np.allclose(reconstructed, score["pred_rv"], rtol=1e-10, atol=1e-15):
            raise RuntimeError(f"Stored/reconstructed ridge prediction mismatch: {model}/{block_id}")
        for row_index, row in score.reset_index(drop=True).iterrows():
            order = np.argsort(np.abs(contribution[row_index]))[::-1]
            for rank, feature_index in enumerate(order, 1):
                details.append({
                    "model": model,
                    "block_id": block_id,
                    "TRADEDATE": row["TRADEDATE"],
                    "SECID": row["SECID"],
                    "y_rv": row["y_rv_saved"],
                    "pred_rv": row["pred_rv"],
                    "qlike": row["qlike"],
                    "unclipped_log_prediction": raw_log[row_index],
                    "log_clip_floor": -30.0,
                    "train_qlike_scale": fitted.rv_scale,
                    "fixed_effect_intercept": intercept[row_index],
                    "rank_abs_contribution": rank,
                    "feature": features[feature_index],
                    "feature_value": row[features[feature_index]],
                    "train_mean": fitted.mean[feature_index],
                    "train_scale": fitted.scale[feature_index],
                    "standardized_value": z[row_index, feature_index],
                    "coefficient": fitted.beta[feature_index],
                    "log_contribution": contribution[row_index, feature_index],
                })
    detail = pd.DataFrame(details)
    detail.to_csv(output / "ridge_floor_feature_contributions.csv", index=False)
    print("\n===== EXACT RIDGE FLOOR FAILURES =====")
    print(floor[["model", "TRADEDATE", "SECID", "y_rv", "pred_rv", "qlike"]].to_string(index=False))
    print("\n===== TOP LOG CONTRIBUTIONS PER FAILURE =====")
    print(
        detail[detail["rank_abs_contribution"] <= 5][
            ["model", "TRADEDATE", "SECID", "rank_abs_contribution", "feature", "standardized_value", "coefficient", "log_contribution"]
        ].to_string(index=False)
    )
    return floor


def ex_post_floor_sensitivity(
    predictions: pd.DataFrame, floor: pd.DataFrame, output: Path
) -> None:
    if floor.empty:
        return
    excluded_dates = set(pd.to_datetime(floor["TRADEDATE"]))
    filtered = predictions[~predictions["TRADEDATE"].isin(excluded_dates)].copy()
    comparisons = [("S1_h1c_without_floor_dates", "M5_linear_directed_dynamic", "M7_catboost_directed_dynamic")]
    result, daily, audit = inference_table(filtered, comparisons, "ex_post_floor_sensitivity")
    result["excluded_dates"] = len(excluded_dates)
    result["analysis_status"] = "ex_post_diagnostic_not_confirmatory"
    result.to_csv(output / "paired_inference_h1c_ex_post_floor_sensitivity.csv", index=False)
    daily.to_csv(output / "daily_loss_h1c_ex_post_floor_sensitivity.csv", index=False)
    audit.assign(
        excluded_dates=len(excluded_dates),
        analysis_status="ex_post_diagnostic_not_confirmatory",
    ).to_csv(output / "fairness_audit_h1c_ex_post_floor_sensitivity.csv", index=False)


def main() -> None:
    args = parse_args()
    predictions, output = load_development(args.root)
    m8_diagnostics(predictions, output)
    floor = reconstruct_ridge_failures(args.root, predictions, output)
    ex_post_floor_sensitivity(predictions, floor, output)
    print("\nAll outputs are exploratory diagnostics; frozen H1/H2 files were not modified.")


if __name__ == "__main__":
    main()
