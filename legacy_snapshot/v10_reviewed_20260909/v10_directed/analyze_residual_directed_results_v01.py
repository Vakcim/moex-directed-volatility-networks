"""Development analysis for the V10 residual directed-graph branch.

This script deliberately keeps the residual target (idiosyncratic RV) separate
from the raw-RV target.  Absolute QLIKE levels are never compared across the
two branches.  The optional cross-branch table compares only within-branch
model-improvement differentials on the exact common TRADEDATE x SECID panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from v10_common import (
    circular_block_bootstrap,
    hac_mean_test,
    hash_frame,
    holm,
    normalize_keys,
    qlike,
    read_table,
)


ACTIVE_MODELS = (
    "M0_har_rv",
    "M1_har_market",
    "M6_catboost_directed_current",
    "M7_catboost_directed_dynamic",
)

# These are the two residual-branch incremental hypotheses fixed before the
# residual development losses were inspected.
CORE = (
    (
        "RH1_current_graph_increment",
        "M1_har_market",
        "M6_catboost_directed_current",
    ),
    (
        "RH2_dynamic_graph_increment",
        "M6_catboost_directed_current",
        "M7_catboost_directed_dynamic",
    ),
)

# Descriptive diagnostics, outside the two-hypothesis Holm family.
DIAGNOSTICS = (
    ("RD0_market_increment", "M0_har_rv", "M1_har_market"),
    (
        "RD1_current_graph_vs_har",
        "M0_har_rv",
        "M6_catboost_directed_current",
    ),
)

CROSS_BRANCH = (
    (
        "XB1_current_graph_increment",
        "M1_har_market",
        "M6_catboost_directed_current",
    ),
    (
        "XB2_dynamic_graph_increment",
        "M6_catboost_directed_current",
        "M7_catboost_directed_dynamic",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_losses(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    required = {"TRADEDATE", "SECID", "model", "y_rv", "pred_rv"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Prediction columns missing: {missing}")
    frame["y_rv"] = pd.to_numeric(frame["y_rv"], errors="coerce")
    frame["pred_rv"] = pd.to_numeric(frame["pred_rv"], errors="coerce")
    bad = (
        ~np.isfinite(frame["y_rv"])
        | ~np.isfinite(frame["pred_rv"])
        | (frame["y_rv"] <= 0)
        | (frame["pred_rv"] <= 0)
    )
    if bad.any():
        raise RuntimeError(f"Nonpositive/nonfinite target or prediction rows: {int(bad.sum())}")
    frame["y_log_rv"] = np.log(frame["y_rv"])
    frame["pred_log_rv"] = np.log(frame["pred_rv"])
    frame["qlike"] = qlike(frame["y_rv"], frame["pred_rv"])
    frame["squared_log_error"] = (frame["y_log_rv"] - frame["pred_log_rv"]) ** 2
    frame["absolute_log_error"] = (
        frame["y_log_rv"] - frame["pred_log_rv"]
    ).abs()
    return frame


def validate_residual_predictions(predictions: pd.DataFrame) -> None:
    models = set(predictions["model"].astype(str))
    missing = sorted(set(ACTIVE_MODELS) - models)
    unexpected = sorted(models - set(ACTIVE_MODELS))
    if missing:
        raise RuntimeError(f"Residual models missing: {missing}")
    if unexpected:
        raise RuntimeError(
            "Unexpected residual models found; stale/ineligible predictions may be present: "
            f"{unexpected}"
        )
    duplicate_count = int(
        predictions.duplicated(["model", "TRADEDATE", "SECID"]).sum()
    )
    if duplicate_count:
        raise RuntimeError(f"Duplicate model-date-SECID prediction keys: {duplicate_count}")


def model_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        rows.append(
            {
                "model": model,
                "target": "idiosyncratic_rv",
                "rows": len(group),
                "dates": group["TRADEDATE"].nunique(),
                "secids": group["SECID"].nunique(),
                "qlike": group["qlike"].mean(),
                "rmse_log": np.sqrt(group["squared_log_error"].mean()),
                "mae_log": group["absolute_log_error"].mean(),
                "calibration_ratio": group["pred_rv"].mean()
                / group["y_rv"].mean(),
            }
        )
    return pd.DataFrame(rows)


def paired_rows(
    predictions: pd.DataFrame, left: str, right: str
) -> tuple[pd.DataFrame, dict]:
    keys = ["TRADEDATE", "SECID"]
    columns = keys + ["y_rv", "qlike"]
    left_frame = predictions.loc[predictions["model"] == left, columns].rename(
        columns={"y_rv": "y_left", "qlike": "loss_left"}
    )
    right_frame = predictions.loc[predictions["model"] == right, columns].rename(
        columns={"y_rv": "y_right", "qlike": "loss_right"}
    )
    pair = left_frame.merge(
        right_frame, on=keys, how="inner", validate="one_to_one"
    )
    if pair.empty:
        raise RuntimeError(f"Empty pair: {left} vs {right}")
    if not np.array_equal(pair["y_left"].to_numpy(), pair["y_right"].to_numpy()):
        raise RuntimeError(f"Target mismatch: {left} vs {right}")
    pair["difference"] = pair["loss_left"] - pair["loss_right"]
    audit = {
        "left_model": left,
        "right_model": right,
        "rows": len(pair),
        "dates": pair["TRADEDATE"].nunique(),
        "secids": pair["SECID"].nunique(),
        "duplicate_keys": int(pair.duplicated(keys).sum()),
        "key_target_sha256": hash_frame(
            pair.rename(columns={"y_left": "y_rv"}),
            ["TRADEDATE", "SECID", "y_rv"],
        ),
    }
    return pair, audit


def infer_values(values: np.ndarray, seed: int) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    row = {
        "n_dates": len(finite),
        "positive_date_fraction": float(np.mean(finite > 0)),
        "median_difference": float(np.median(finite)),
    }
    row.update(hac_mean_test(finite, 10))
    row.update(circular_block_bootstrap(finite, 20, 10_000, seed))
    return row


def inference_family(
    predictions: pd.DataFrame,
    comparisons: Sequence[tuple[str, str, str]],
    family: str,
    analysis_status: str,
    apply_holm: bool,
    seed_base: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    daily_parts: list[pd.DataFrame] = []
    audits: list[dict] = []
    for index, (hypothesis, left, right) in enumerate(comparisons):
        pair, audit = paired_rows(predictions, left, right)
        daily = pair.groupby("TRADEDATE", as_index=False).agg(
            difference=("difference", "mean"),
            assets=("SECID", "size"),
        )
        row = {
            "family": family,
            "hypothesis": hypothesis,
            "left_model": left,
            "right_model": right,
            "direction": "positive => right model lower QLIKE",
            "analysis_status": analysis_status,
        }
        row.update(infer_values(daily["difference"].to_numpy(), seed_base + index))
        rows.append(row)
        daily_parts.append(
            daily[["TRADEDATE", "difference", "assets"]].rename(
                columns={
                    "difference": f"d_{hypothesis}",
                    "assets": f"assets_{hypothesis}",
                }
            )
        )
        audit.update(
            {
                "family": family,
                "hypothesis": hypothesis,
                "analysis_status": analysis_status,
            }
        )
        audits.append(audit)

    result = pd.DataFrame(rows)
    if apply_holm:
        for column in (
            "hac_p_two_sided",
            "hac_p_one_sided",
            "bootstrap_p_two_sided",
            "bootstrap_p_one_sided",
        ):
            result[f"holm_{column}"] = holm(result[column])
    daily_all = daily_parts[0]
    for part in daily_parts[1:]:
        daily_all = daily_all.merge(part, on="TRADEDATE", how="outer")
    return result, daily_all.sort_values("TRADEDATE"), pd.DataFrame(audits)


def block_means(daily: pd.DataFrame) -> pd.DataFrame:
    ordered = daily.sort_values("TRADEDATE").copy()
    ordered["nonoverlap_block20"] = np.arange(len(ordered)) // 20 + 1
    numeric = [column for column in ordered if column.startswith("d_")]
    return ordered.groupby("nonoverlap_block20", as_index=False).agg(
        start=("TRADEDATE", "min"),
        end=("TRADEDATE", "max"),
        dates=("TRADEDATE", "size"),
        **{column: (column, "mean") for column in numeric},
    )


def branch_gain_rows(
    predictions: pd.DataFrame,
    left: str,
    right: str,
    branch: str,
) -> pd.DataFrame:
    pair, _ = paired_rows(predictions, left, right)
    return pair[
        ["TRADEDATE", "SECID", "y_left", "difference"]
    ].rename(
        columns={
            "y_left": f"y_{branch}",
            "difference": f"gain_{branch}",
        }
    )


def cross_branch_increment_analysis(
    raw: pd.DataFrame,
    residual: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    daily_parts: list[pd.DataFrame] = []
    audits: list[dict] = []
    keys = ["TRADEDATE", "SECID"]
    for index, (hypothesis, left, right) in enumerate(CROSS_BRANCH):
        raw_gain = branch_gain_rows(raw, left, right, "raw")
        residual_gain = branch_gain_rows(residual, left, right, "residual")
        pair = raw_gain.merge(
            residual_gain,
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        if pair.empty:
            raise RuntimeError(f"No common raw/residual keys for {hypothesis}")
        pair["difference"] = pair["gain_residual"] - pair["gain_raw"]
        daily = pair.groupby("TRADEDATE", as_index=False).agg(
            difference=("difference", "mean"),
            raw_gain=("gain_raw", "mean"),
            residual_gain=("gain_residual", "mean"),
            assets=("SECID", "size"),
        )
        row = {
            "family": "raw_residual_increment_exploratory",
            "hypothesis": hypothesis,
            "left_model": left,
            "right_model": right,
            "estimand": "residual within-branch QLIKE gain minus raw within-branch QLIKE gain",
            "direction": "positive => residualization increases right-model advantage",
            "analysis_status": "exploratory_development_different_targets",
        }
        row.update(infer_values(daily["difference"].to_numpy(), 260924 + index))
        rows.append(row)
        daily_parts.append(
            daily.rename(
                columns={
                    "difference": f"d_{hypothesis}",
                    "raw_gain": f"raw_gain_{hypothesis}",
                    "residual_gain": f"residual_gain_{hypothesis}",
                    "assets": f"assets_{hypothesis}",
                }
            )
        )
        audits.append(
            {
                "family": "raw_residual_increment_exploratory",
                "hypothesis": hypothesis,
                "left_model": left,
                "right_model": right,
                "rows": len(pair),
                "dates": pair["TRADEDATE"].nunique(),
                "secids": pair["SECID"].nunique(),
                "duplicate_keys": int(pair.duplicated(keys).sum()),
                "common_key_sha256": hash_frame(pair, keys),
                "raw_key_target_sha256": hash_frame(
                    pair.rename(columns={"y_raw": "y_rv"}),
                    ["TRADEDATE", "SECID", "y_rv"],
                ),
                "residual_key_target_sha256": hash_frame(
                    pair.rename(columns={"y_residual": "y_rv"}),
                    ["TRADEDATE", "SECID", "y_rv"],
                ),
                "targets_equal_required": False,
                "absolute_cross_branch_qlike_compared": False,
                "analysis_status": "exploratory_development_different_targets",
            }
        )
    daily_all = daily_parts[0]
    for part in daily_parts[1:]:
        daily_all = daily_all.merge(part, on="TRADEDATE", how="outer")
    return pd.DataFrame(rows), daily_all.sort_values("TRADEDATE"), pd.DataFrame(audits)


def read_design(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Residual benchmark design missing: {path}")
    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("linear_residual_forecast_losses_computed") is not False:
        raise RuntimeError("Residual design does not confirm exclusion of linear losses")
    if set(design.get("active_models", [])) != set(ACTIVE_MODELS):
        raise RuntimeError("Residual design active_models do not match Amendment 06")
    return design


def markdown_summary(
    metrics_frame: pd.DataFrame,
    core: pd.DataFrame,
    diagnostics: pd.DataFrame,
    cross: pd.DataFrame | None,
) -> str:
    sections = [
        "# V10 residual directed-network development results",
        "",
        "- Target: idiosyncratic realized variance (IRV).",
        "- Positive paired loss difference means the right-hand model has lower QLIKE.",
        "- RH1 and RH2 form one two-hypothesis Holm-adjusted development family.",
        "- These are development results, not the sealed confirmatory holdout.",
        "- Absolute raw-RV and residual-IRV QLIKE levels are not compared.",
        "",
        "## Residual model metrics",
        "",
        metrics_frame.to_markdown(index=False),
        "",
        "## Residual incremental hypotheses",
        "",
        core.to_markdown(index=False),
        "",
        "## Residual diagnostics",
        "",
        diagnostics.to_markdown(index=False),
    ]
    if cross is not None:
        sections.extend(
            [
                "",
                "## Raw versus residual improvement increments (exploratory)",
                "",
                "Only within-branch loss gains are contrasted on common keys; absolute QLIKE levels are not contrasted.",
                "",
                cross.to_markdown(index=False),
            ]
        )
    return "\n".join(sections) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_v02"))
    parser.add_argument(
        "--skip-cross-branch",
        action="store_true",
        help="Skip exploratory raw-versus-residual improvement comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    residual_out = args.root / "benchmark_v10_directed_development_residual"
    residual_path = residual_out / "predictions_development.parquet"
    residual_design_path = residual_out / "residual_benchmark_design.json"
    read_design(residual_design_path)
    residual = add_losses(normalize_keys(read_table(residual_path)))
    validate_residual_predictions(residual)

    metrics_frame = model_metrics(residual)
    core, core_daily, core_audit = inference_family(
        residual,
        CORE,
        family="residual_increment_core2",
        analysis_status="development_hypothesis_test_not_confirmatory",
        apply_holm=True,
        seed_base=260824,
    )
    diagnostics, diagnostic_daily, diagnostic_audit = inference_family(
        residual,
        DIAGNOSTICS,
        family="residual_diagnostics",
        analysis_status="descriptive_development",
        apply_holm=False,
        seed_base=260834,
    )
    daily = core_daily.merge(diagnostic_daily, on="TRADEDATE", how="outer")

    cross = None
    cross_daily = None
    cross_audit = None
    raw_path = (
        args.root
        / "benchmark_v10_directed_development"
        / "predictions_development.parquet"
    )
    if not args.skip_cross_branch:
        if not raw_path.exists():
            raise RuntimeError(
                f"Raw development predictions missing: {raw_path}. "
                "Use --skip-cross-branch to run residual-only analysis."
            )
        raw = add_losses(normalize_keys(read_table(raw_path)))
        required_raw = {model for _, left, right in CROSS_BRANCH for model in (left, right)}
        missing_raw = sorted(required_raw - set(raw["model"]))
        if missing_raw:
            raise RuntimeError(f"Raw models missing for cross-branch analysis: {missing_raw}")
        cross, cross_daily, cross_audit = cross_branch_increment_analysis(raw, residual)

    metrics_frame.to_csv(residual_out / "model_metrics_residual.csv", index=False)
    core.to_csv(residual_out / "paired_inference_residual_core.csv", index=False)
    diagnostics.to_csv(
        residual_out / "paired_inference_residual_diagnostics.csv", index=False
    )
    daily.to_csv(
        residual_out / "daily_loss_differentials_residual.csv", index=False
    )
    pd.concat([core_audit, diagnostic_audit], ignore_index=True).to_csv(
        residual_out / "fairness_audit_residual.csv", index=False
    )
    block_means(daily).to_csv(
        residual_out / "block_means_residual.csv", index=False
    )
    if cross is not None and cross_daily is not None and cross_audit is not None:
        cross.to_csv(
            residual_out
            / "paired_inference_raw_residual_increment_exploratory.csv",
            index=False,
        )
        cross_daily.to_csv(
            residual_out / "daily_raw_residual_increment_exploratory.csv",
            index=False,
        )
        cross_audit.to_csv(
            residual_out / "fairness_audit_raw_residual_increment.csv",
            index=False,
        )

    design = {
        "analysis": "V10 residual directed-network development",
        "analysis_status": "development_hypothesis_test_not_confirmatory",
        "protocol_amendment": "V10 Amendment 06",
        "target": "idiosyncratic_rv",
        "active_models": list(ACTIVE_MODELS),
        "core_hypotheses": [list(item) for item in CORE],
        "core_multiplicity": "Holm within RH1-RH2",
        "hac_lags": 10,
        "bootstrap": {"method": "circular_block", "block": 20, "reps": 10000},
        "absolute_raw_residual_qlike_compared": False,
        "cross_branch_status": (
            "exploratory_development_different_targets"
            if cross is not None
            else "skipped"
        ),
        "residual_predictions_sha256": sha256_file(residual_path),
        "residual_benchmark_design_sha256": sha256_file(residual_design_path),
        "raw_predictions_sha256": (
            sha256_file(raw_path) if cross is not None else None
        ),
    }
    (residual_out / "residual_analysis_design.json").write_text(
        json.dumps(design, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (residual_out / "RESIDUAL_DEVELOPMENT_RESULTS_V10.md").write_text(
        markdown_summary(metrics_frame, core, diagnostics, cross),
        encoding="utf-8",
    )

    print("\n===== RESIDUAL MODEL METRICS =====")
    print(metrics_frame.to_string(index=False))
    print("\n===== RESIDUAL CORE (HOLM FAMILY) =====")
    print(core.to_string(index=False))
    print("\n===== RESIDUAL DIAGNOSTICS =====")
    print(diagnostics.to_string(index=False))
    if cross is not None:
        print("\n===== RAW-RESIDUAL INCREMENT (EXPLORATORY) =====")
        print(cross.to_string(index=False))


if __name__ == "__main__":
    main()
