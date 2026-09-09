"""Frozen v10 paired inference, regime test, placebos and result tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

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


CORE = [
    ("H1a", "M6_catboost_directed_current", "M7_catboost_directed_dynamic"),
    ("H1b", "M3_pearson_dynamic", "M7_catboost_directed_dynamic"),
    ("H1c", "M5_linear_directed_dynamic", "M7_catboost_directed_dynamic"),
]
PLACEBOS = [
    ("temporal_lag20", "P1_catboost_lag20_dynamic", "M7_catboost_directed_dynamic"),
    ("source_identity", "P2_catboost_identity_dynamic", "M7_catboost_directed_dynamic"),
    ("frozen_graph", "P3_catboost_frozen_dynamic", "M7_catboost_directed_dynamic"),
]


def add_losses(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["y_rv"] = pd.to_numeric(frame["y_rv"], errors="coerce")
    frame["pred_rv"] = pd.to_numeric(frame["pred_rv"], errors="coerce")
    frame["y_log_rv"] = np.log(frame["y_rv"].where(frame["y_rv"] > 0))
    frame["pred_log_rv"] = np.log(frame["pred_rv"].where(frame["pred_rv"] > 0))
    frame["qlike"] = qlike(frame["y_rv"], frame["pred_rv"])
    frame["squared_log_error"] = (frame["y_log_rv"] - frame["pred_log_rv"]) ** 2
    frame["absolute_log_error"] = (frame["y_log_rv"] - frame["pred_log_rv"]).abs()
    return frame


def metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", sort=True):
        rows.append({
            "model": model, "rows": len(group), "dates": group["TRADEDATE"].nunique(),
            "secids": group["SECID"].nunique(), "qlike": group["qlike"].mean(),
            "rmse_log": np.sqrt(group["squared_log_error"].mean()),
            "mae_log": group["absolute_log_error"].mean(),
            "calibration_ratio": group["pred_rv"].mean() / group["y_rv"].mean(),
        })
    return pd.DataFrame(rows)


def paired_daily(predictions: pd.DataFrame, left: str, right: str) -> tuple[pd.DataFrame, dict]:
    keys = ["TRADEDATE", "SECID"]
    columns = keys + ["y_rv", "qlike"]
    l = predictions[predictions["model"] == left][columns].rename(
        columns={"y_rv": "y_left", "qlike": "loss_left"}
    )
    r = predictions[predictions["model"] == right][columns].rename(
        columns={"y_rv": "y_right", "qlike": "loss_right"}
    )
    pair = l.merge(r, on=keys, how="inner", validate="one_to_one")
    if pair.empty or not np.allclose(pair["y_left"], pair["y_right"], rtol=0, atol=0):
        raise RuntimeError(f"Target mismatch/empty pair: {left} vs {right}")
    pair["difference"] = pair["loss_left"] - pair["loss_right"]
    daily = pair.groupby("TRADEDATE", as_index=False).agg(
        difference=("difference", "mean"), assets=("SECID", "size")
    )
    audit = {
        "left_model": left, "right_model": right, "rows": len(pair),
        "dates": daily["TRADEDATE"].nunique(), "secids": pair["SECID"].nunique(),
        "duplicate_keys": int(pair.duplicated(keys).sum()),
        "key_target_sha256": hash_frame(
            pair.rename(columns={"y_left": "y_rv"}), ["TRADEDATE", "SECID", "y_rv"]
        ),
    }
    return daily, audit


def inference_table(
    predictions: pd.DataFrame,
    comparisons,
    family: str,
    spec_version: str = "v10",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, daily_parts, audits = [], [], []
    for index, (hypothesis, left, right) in enumerate(comparisons):
        daily, audit = paired_daily(predictions, left, right)
        values = daily["difference"].to_numpy()
        row = {
            "family": family, "hypothesis": hypothesis, "left_model": left,
            "right_model": right, "direction": "positive => right model lower QLIKE",
            "n_dates": len(values), "positive_date_fraction": np.mean(values > 0),
            "median_difference": np.median(values),
        }
        row.update(hac_mean_test(values, 10))
        bootstrap_seed = 260821 + index if spec_version == "v10" else 260821
        row.update(circular_block_bootstrap(values, 20, 10_000, bootstrap_seed))
        row["bootstrap_seed"] = bootstrap_seed
        rows.append(row)
        part = daily.rename(columns={"difference": f"d_{hypothesis}"})
        daily_parts.append(part[["TRADEDATE", f"d_{hypothesis}"]])
        audit.update({"family": family, "hypothesis": hypothesis})
        audits.append(audit)
    result = pd.DataFrame(rows)
    if family == "directed_core3":
        for column in ("hac_p_two_sided", "hac_p_one_sided", "bootstrap_p_two_sided", "bootstrap_p_one_sided"):
            result[f"holm_{column}"] = holm(result[column])
    daily_all = daily_parts[0]
    for part in daily_parts[1:]:
        daily_all = daily_all.merge(part, on="TRADEDATE", how="outer")
    return result, daily_all.sort_values("TRADEDATE"), pd.DataFrame(audits)


def hac_regression(y: np.ndarray, z: np.ndarray, lags: int = 10) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(z)
    y, z = np.asarray(y)[valid], np.asarray(z)[valid]
    if len(y) <= lags + 3:
        raise RuntimeError("Too few observations for regime HAC regression")
    x = np.column_stack([np.ones(len(y)), z])
    inv = np.linalg.pinv(x.T @ x)
    beta = inv @ x.T @ y
    residual = y - x @ beta
    score = x * residual[:, None]
    meat = score.T @ score
    cap = min(lags, len(y) - 1)
    for lag in range(1, cap + 1):
        cross = score[lag:].T @ score[:-lag]
        meat += (1 - lag / (cap + 1)) * (cross + cross.T)
    covariance = inv @ meat @ inv
    se = math.sqrt(max(float(covariance[1, 1]), 0.0))
    z_stat = float(beta[1] / se) if se > 0 else np.nan
    return {
        "n_dates": len(y), "intercept": float(beta[0]), "stress_coefficient": float(beta[1]),
        "hac_lags": cap, "hac_se": se, "hac_z": z_stat,
        "hac_p_two_sided": math.erfc(abs(z_stat) / math.sqrt(2)) if np.isfinite(z_stat) else np.nan,
        "hac_p_one_sided_positive": 0.5 * math.erfc(z_stat / math.sqrt(2)) if np.isfinite(z_stat) else np.nan,
    }


def regime_test(predictions: pd.DataFrame, h1a_daily: pd.DataFrame) -> pd.DataFrame:
    stress = predictions[predictions["model"] == "M7_catboost_directed_dynamic"].groupby(
        "TRADEDATE", as_index=False
    )["stress_z"].first()
    merged = h1a_daily.merge(stress, on="TRADEDATE", how="inner")
    row = {"hypothesis": "H2", "equation": "d_H1a[t] = a + b*stress_z[t] + error[t]"}
    row.update(hac_regression(merged["d_H1a"].to_numpy(), merged["stress_z"].to_numpy()))
    return pd.DataFrame([row])


def block_means(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.sort_values("TRADEDATE").copy()
    out["nonoverlap_block20"] = np.arange(len(out)) // 20 + 1
    numeric = [c for c in out if c.startswith("d_")]
    return out.groupby("nonoverlap_block20", as_index=False).agg(
        start=("TRADEDATE", "min"), end=("TRADEDATE", "max"), dates=("TRADEDATE", "size"),
        **{column: (column, "mean") for column in numeric},
    )


def load_predictions(a) -> tuple[pd.DataFrame, Path, bool]:
    prefix = "benchmark_v10" if a.spec_version == "v10" else "benchmark_v10_1"
    if a.mode == "development":
        out = a.root / f"{prefix}_directed_development"
        predictions = normalize_keys(read_table(out / "predictions_development.parquet"))
        # Development prediction checkpoints contain a convenience copy of
        # stress_z.  Always replace it with the current deterministic feature
        # artifact so an implementation-only stress correction does not
        # require refitting otherwise unchanged forecasting models.
        features = normalize_keys(
            read_table(a.root / "model_data" / "directed_graph_features.parquet")
        )
        stress = features[["TRADEDATE", "stress_z"]].copy()
        inconsistent = stress.groupby("TRADEDATE")["stress_z"].nunique(dropna=False)
        if (inconsistent > 1).any():
            raise RuntimeError("stress_z is not unique within TRADEDATE")
        stress = stress.drop_duplicates("TRADEDATE")
        predictions = predictions.drop(columns="stress_z", errors="ignore").merge(
            stress, on="TRADEDATE", how="left", validate="many_to_one"
        )
        return add_losses(predictions), out, False
    out = a.root / f"{prefix}_directed_confirmatory"
    sealed = normalize_keys(read_table(out / "predictions_sealed.parquet"))
    n_dates = sealed["TRADEDATE"].nunique()
    if n_dates < 120 and sealed["TRADEDATE"].max() < pd.Timestamp("2027-03-31"):
        raise RuntimeError(f"Confirmatory holdout remains sealed: {n_dates}/120 valid dates")
    samples = normalize_keys(read_table(a.root / "model_data" / "forecast_samples.parquet"))
    targets = samples[["TRADEDATE", "SECID", "y_rv"]]
    predictions = sealed.merge(targets, on=["TRADEDATE", "SECID"], how="inner", validate="many_to_one")
    features = normalize_keys(read_table(a.root / "model_data" / "directed_graph_features.parquet"))
    predictions = predictions.merge(
        features[["TRADEDATE", "SECID", "stress_z"]],
        on=["TRADEDATE", "SECID"], how="left", validate="many_to_one",
    )
    return add_losses(predictions), out, n_dates < 120


def analyze_residual(root: Path, output: Path, spec_version: str = "v10") -> None:
    prefix = "benchmark_v10" if spec_version == "v10" else "benchmark_v10_1"
    path = root / f"{prefix}_directed_development_residual" / "predictions_development.parquet"
    if not path.exists():
        return
    pred = add_losses(normalize_keys(pd.read_parquet(path)))
    table, _, audit = inference_table(pred, [(
        "H3", "M6_catboost_directed_current", "M7_catboost_directed_dynamic"
    )], "residual_robustness", spec_version)
    table.to_csv(output / "residual_robustness.csv", index=False)
    audit.to_csv(output / "fairness_audit_residual.csv", index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data_v02"))
    p.add_argument("--mode", choices=["development", "confirmatory"], default="development")
    p.add_argument("--spec-version", choices=["v10", "v10.1"], default="v10")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    predictions, out, underpowered = load_predictions(a)
    required = sorted({m for _, left, right in CORE for m in (left, right)})
    missing = sorted(set(required) - set(predictions["model"]))
    if missing:
        raise RuntimeError(f"Core models missing: {missing}")
    model_metrics = metrics(predictions)
    stress = (
        predictions[predictions["model"] == "M7_catboost_directed_dynamic"]
        .groupby("TRADEDATE", as_index=False)["stress_z"]
        .first()
    )
    stress_audit = pd.DataFrame([{
        "prediction_dates": int(stress["TRADEDATE"].nunique()),
        "finite_stress_dates": int(np.isfinite(stress["stress_z"]).sum()),
        "missing_stress_dates": int((~np.isfinite(stress["stress_z"])).sum()),
        "first_finite_stress_date": stress.loc[
            np.isfinite(stress["stress_z"]), "TRADEDATE"
        ].min(),
        "last_finite_stress_date": stress.loc[
            np.isfinite(stress["stress_z"]), "TRADEDATE"
        ].max(),
    }])
    core, core_daily, core_audit = inference_table(
        predictions, CORE, "directed_core3", a.spec_version
    )
    h1a = core_daily[["TRADEDATE", "d_H1a"]]
    regime = regime_test(predictions, h1a)
    placebo_available = [c for c in PLACEBOS if c[1] in set(predictions["model"])]
    if placebo_available:
        placebo, placebo_daily, placebo_audit = inference_table(
            predictions, placebo_available, "placebo", a.spec_version
        )
    else:
        placebo, placebo_daily, placebo_audit = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    daily = core_daily
    if not placebo_daily.empty:
        daily = daily.merge(placebo_daily, on="TRADEDATE", how="outer")
    model_metrics.to_csv(out / "model_metrics.csv", index=False)
    stress_audit.to_csv(out / "stress_coverage.csv", index=False)
    core.to_csv(out / "paired_inference_core.csv", index=False)
    regime.to_csv(out / "paired_inference_regime.csv", index=False)
    placebo.to_csv(out / "paired_inference_placebo.csv", index=False)
    daily.to_csv(out / "daily_loss_differentials.csv", index=False)
    pd.concat([core_audit, placebo_audit], ignore_index=True).to_csv(out / "fairness_audit.csv", index=False)
    block_means(daily).to_csv(out / "block_means.csv", index=False)
    analyze_residual(a.root, out, a.spec_version)
    summary = [
        "# Directed volatility networks v10 — analysis summary", "",
        f"- Mode: `{a.mode}`.", f"- Specification: `{a.spec_version}`.",
        f"- Underpowered confirmatory: `{underpowered}`.",
        f"- Dates: {predictions['TRADEDATE'].nunique()}.",
        "- Positive loss difference means the right-hand model has lower QLIKE.", "",
        "## Core family", "", core.to_markdown(index=False), "", "## Regime H2", "",
        regime.to_markdown(index=False),
    ]
    version_label = "V10" if a.spec_version == "v10" else "V10_1"
    result_name = (
        f"FINAL_DIRECTED_RESULTS_{version_label}.md"
        if a.mode == "confirmatory"
        else f"DEVELOPMENT_DIRECTED_RESULTS_{version_label}.md"
    )
    (out / result_name).write_text(
        "\n".join(summary), encoding="utf-8"
    )
    print(core.to_string(index=False))
    print(regime.to_string(index=False))


if __name__ == "__main__":
    main()
