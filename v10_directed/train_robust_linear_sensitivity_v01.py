"""Post-development robust linear sensitivity for the frozen v10 experiment.

The script never overwrites primary v10 predictions or inference.  It replaces
unbounded raw permutation-QLIKE strength by log-strength and applies a
train-only standardized extrapolation guard to the linear HAR-X benchmark.
All outputs are exploratory until evaluated on the future sealed holdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_directed_results_v01 import add_losses, inference_table, metrics
from train_directed_har_benchmark_v01 import (
    DYNAMIC,
    MARKET_FEATURES,
    OWN_FEATURES,
    RIDGE_GRID,
    date_blocks,
    fit_ridge,
    load_data,
    prefixed,
    score_dates,
    train_rows,
)
from v10_common import EPS, calibration_scale, normalize_keys, read_table


ROBUST_CURRENT = [
    "linear_incoming_nbr_logrv_d",
    "linear_incoming_nbr_logrv_w",
    "linear_incoming_nbr_logrv_m",
    "linear_incoming_degree",
    "linear_log1p_incoming_strength",
]

ROBUST_DYNAMIC = [
    "linear_incoming_jaccard_distance",
    "linear_delta_log1p_incoming_strength",
    "linear_weighted_incoming_edge_change",
    "linear_incoming_edge_turnover",
    "linear_g_change",
]

ROBUST_FEATURE_SETS = {
    "R4_linear_current_logclip": [*OWN_FEATURES, *MARKET_FEATURES, *ROBUST_CURRENT],
    "R5_linear_dynamic_logclip": [
        *OWN_FEATURES,
        *MARKET_FEATURES,
        *ROBUST_CURRENT,
        *ROBUST_DYNAMIC,
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_v02"))
    parser.add_argument("--stage", choices=["select", "run", "analyze", "all"], default="all")
    parser.add_argument("--z-clip", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def add_robust_strength(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    current = pd.to_numeric(out["linear_incoming_strength"], errors="coerce")
    delta = pd.to_numeric(out["linear_delta_incoming_strength"], errors="coerce")
    if (current < -1e-12).any():
        raise RuntimeError("Negative raw linear strength")
    previous = current - delta
    if (previous < -1e-8).any():
        raise RuntimeError("Reconstructed previous linear strength is negative")
    current = current.clip(lower=0)
    previous = previous.clip(lower=0)
    out["linear_log1p_incoming_strength"] = np.log1p(current)
    out["linear_delta_log1p_incoming_strength"] = np.log1p(current) - np.log1p(previous)
    required = sorted(set(sum(ROBUST_FEATURE_SETS.values(), [])))
    if not np.isfinite(out[required]).all().all():
        raise RuntimeError("Non-finite robust linear feature")
    return out


def fit_robust(train: pd.DataFrame, features: list[str], alpha: float, z_clip: float):
    fitted = fit_ridge(train, features, alpha)
    z = (train[features].to_numpy(float) - fitted.mean) / fitted.scale
    z = np.clip(z, -z_clip, z_clip)
    intercept = np.array([
        fitted.intercepts.get(str(secid), fitted.fallback)
        for secid in train["SECID"]
    ])
    raw = np.exp(np.clip(intercept + z @ fitted.beta, -30, 5))
    fitted.rv_scale = calibration_scale(train["y_rv"].to_numpy(), raw)
    return fitted


def predict_robust(fitted, frame: pd.DataFrame, z_clip: float) -> np.ndarray:
    z = (frame[fitted.features].to_numpy(float) - fitted.mean) / fitted.scale
    z = np.clip(z, -z_clip, z_clip)
    intercept = np.array([
        fitted.intercepts.get(str(secid), fitted.fallback)
        for secid in frame["SECID"]
    ])
    log_prediction = intercept + z @ fitted.beta
    return np.maximum(
        np.exp(np.clip(log_prediction, -30, 5)) * fitted.rv_scale,
        EPS,
    )


def selection_path(output: Path) -> Path:
    return output / "hyperparameter_selection_robust_linear.csv"


def select(data: pd.DataFrame, output: Path, z_clip: float, force: bool) -> None:
    path = selection_path(output)
    if path.exists() and not force:
        print(f"selection exists: {path}")
        return
    dates = score_dates(data, "val")
    if len(dates) != 60:
        raise RuntimeError(f"Expected 60 validation dates, found {len(dates)}")
    rows = []
    for model, features in ROBUST_FEATURE_SETS.items():
        for alpha in RIDGE_GRID:
            losses = []
            for block in date_blocks(dates, 20):
                train = train_rows(data, block[0], "val")
                score = data[data["TRADEDATE"].isin(block)]
                fitted = fit_robust(train, features, alpha, z_clip)
                pred = predict_robust(fitted, score, z_clip)
                from v10_common import qlike
                losses.extend(qlike(score["y_rv"].to_numpy(), pred))
            rows.append({
                "model": model,
                "alpha": alpha,
                "validation_qlike": float(np.mean(losses)),
                "rows": len(losses),
                "dates": len(dates),
                "z_clip": z_clip,
            })
            print(f"robust alpha {model} {alpha:g}: {np.mean(losses):.8f}")
    table = pd.DataFrame(rows).sort_values(["model", "validation_qlike", "alpha"])
    table["selected"] = table.groupby("model").cumcount() == 0
    table["analysis_status"] = "exploratory_post_development"
    table.to_csv(path, index=False)


def selected_alphas(output: Path) -> dict[str, float]:
    table = pd.read_csv(selection_path(output))
    selected = table[table["selected"].astype(str).str.lower().isin(["true", "1"])]
    result = dict(zip(selected["model"].astype(str), selected["alpha"].astype(float)))
    if set(result) != set(ROBUST_FEATURE_SETS):
        raise RuntimeError("Robust alpha selection incomplete")
    return result


def run(data: pd.DataFrame, output: Path, z_clip: float, force: bool) -> None:
    path = output / "predictions_development_robust_linear.parquet"
    if path.exists() and not force:
        print(f"predictions exist: {path}")
        return
    alphas = selected_alphas(output)
    dates = score_dates(data, "development")
    pieces = []
    for model, features in ROBUST_FEATURE_SETS.items():
        for block_id, block in enumerate(date_blocks(dates, 20), 1):
            train = train_rows(data, block[0], "development")
            score = data[data["TRADEDATE"].isin(block)].copy()
            fitted = fit_robust(train, features, alphas[model], z_clip)
            score["pred_rv"] = predict_robust(fitted, score, z_clip)
            score["model"] = model
            score["block_id"] = block_id
            score["alpha"] = alphas[model]
            score["z_clip"] = z_clip
            pieces.append(score[[
                "TRADEDATE", "next_date", "SECID", "y_rv", "model",
                "block_id", "alpha", "z_clip", "pred_rv",
            ]])
            print(f"robust development {model}: block {block_id} rows={len(score)}")
    predictions = pd.concat(pieces, ignore_index=True)
    if predictions.duplicated(["model", "TRADEDATE", "SECID"]).any():
        raise RuntimeError("Duplicate robust prediction key")
    predictions.to_parquet(path, index=False)


def analyze(root: Path, output: Path) -> None:
    robust = normalize_keys(read_table(output / "predictions_development_robust_linear.parquet"))
    primary = normalize_keys(read_table(
        root / "benchmark_v10_directed_development" / "predictions_development.parquet"
    ))
    keep = primary[primary["model"].isin([
        "M5_linear_directed_dynamic",
        "M6_catboost_directed_current",
        "M7_catboost_directed_dynamic",
    ])]
    predictions = add_losses(pd.concat([keep, robust], ignore_index=True, sort=False))
    comparisons = [
        ("R1_robust_dynamic_vs_current", "R4_linear_current_logclip", "R5_linear_dynamic_logclip"),
        ("R2_robust_vs_original_linear", "M5_linear_directed_dynamic", "R5_linear_dynamic_logclip"),
        ("R3_catboost_vs_robust_linear", "R5_linear_dynamic_logclip", "M7_catboost_directed_dynamic"),
    ]
    result, daily, audit = inference_table(predictions, comparisons, "robust_linear_exploratory")
    result["analysis_status"] = "exploratory_post_development"
    result.to_csv(output / "paired_inference_robust_linear.csv", index=False)
    daily.to_csv(output / "daily_loss_differentials_robust_linear.csv", index=False)
    audit.assign(analysis_status="exploratory_post_development").to_csv(
        output / "fairness_audit_robust_linear.csv", index=False
    )
    metrics(predictions).to_csv(output / "model_metrics_robust_linear.csv", index=False)
    floor = predictions.groupby("model", as_index=False).agg(
        rows=("pred_rv", "size"),
        exact_floor_predictions=("pred_rv", lambda x: int((x <= EPS * (1 + 1e-9)).sum())),
        minimum_prediction=("pred_rv", "min"),
        maximum_prediction=("pred_rv", "max"),
    )
    floor.to_csv(output / "prediction_range_audit_robust_linear.csv", index=False)
    print("\n===== ROBUST LINEAR INFERENCE =====")
    print(result.to_string(index=False))
    print("\n===== PREDICTION RANGE =====")
    print(floor.to_string(index=False))


def main() -> None:
    args = parse_args()
    train_args = argparse.Namespace(root=args.root, variant="raw", features=None, split_dates=None)
    data, _ = load_data(train_args)
    data = add_robust_strength(data)
    output = args.root / "benchmark_v10_directed_robustness"
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in ("select", "all"):
        select(data, output, args.z_clip, args.force)
    if args.stage in ("run", "all"):
        run(data, output, args.z_clip, args.force)
    if args.stage in ("analyze", "all"):
        analyze(args.root, output)


if __name__ == "__main__":
    main()
