"""Resumable v10 walk-forward training for M0--M8 and placebo models.

Scientific outputs are written only under benchmark_v10_directed_*; v09 is
never read for outcomes or overwritten.  Each model/block prediction is a
checkpoint, so a killed run resumes without recomputing completed blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from v10_common import (
    DEVELOPMENT_END,
    MARKET_FEATURES,
    OWN_FEATURES,
    build_har_panel,
    calibration_scale,
    catboost_grid,
    date_blocks,
    hash_frame,
    json_dump,
    normalize_keys,
    qlike,
    read_table,
    recover_date_table,
    require_columns,
    source_columns,
)


RIDGE_GRID = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
CURRENT = ["incoming_nbr_logrv_d", "incoming_nbr_logrv_w", "incoming_nbr_logrv_m",
           "incoming_degree", "incoming_strength"]
DYNAMIC = ["incoming_jaccard_distance", "delta_incoming_strength",
           "weighted_incoming_edge_change", "incoming_edge_turnover", "g_change"]


def prefixed(prefix: str, suffixes: list[str]) -> list[str]:
    return [f"{prefix}_{x}" for x in suffixes]


FEATURE_SETS = {
    "M0_har_rv": OWN_FEATURES,
    "M1_har_market": OWN_FEATURES + MARKET_FEATURES,
    "M2_pearson_current": OWN_FEATURES + MARKET_FEATURES + prefixed("pearson", CURRENT),
    "M3_pearson_dynamic": OWN_FEATURES + MARKET_FEATURES + prefixed("pearson", CURRENT + DYNAMIC),
    "M4_linear_directed_current": OWN_FEATURES + MARKET_FEATURES + prefixed("linear", CURRENT),
    "M5_linear_directed_dynamic": OWN_FEATURES + MARKET_FEATURES + prefixed("linear", CURRENT + DYNAMIC),
    "M6_catboost_directed_current": OWN_FEATURES + MARKET_FEATURES + prefixed("catboost", CURRENT),
    "M7_catboost_directed_dynamic": OWN_FEATURES + MARKET_FEATURES + prefixed("catboost", CURRENT + DYNAMIC),
}

PLACEBO_FEATURE_SETS = {
    "P1_catboost_lag20_dynamic": OWN_FEATURES + MARKET_FEATURES + prefixed("catboost", CURRENT)
        + prefixed("catboost_lag20", DYNAMIC),
    "P2_catboost_identity_dynamic": OWN_FEATURES + MARKET_FEATURES
        + prefixed("catboost_identity", CURRENT + DYNAMIC),
    "P3_catboost_frozen_dynamic": OWN_FEATURES + MARKET_FEATURES
        + prefixed("catboost_frozen", CURRENT + DYNAMIC),
}


def residual_learner_feasibility(
    root: Path,
) -> tuple[set[str], set[str], str]:
    """Load the Amendment 06 pre-forecast residual learner decision."""
    path = root / "graphs_v10" / "directed_learner_feasibility_residual.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing residual learner feasibility audit: {path}. "
            "Run residual graph assembly with Amendment 06 first."
        )
    table = pd.read_csv(path)
    require_columns(
        table,
        ["variant", "learner", "feasibility_passed", "downstream_status"],
        "residual graph learner feasibility",
    )
    if table["learner"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate residual learner feasibility rows")
    if set(table["variant"].astype(str)) != {"residual"}:
        raise RuntimeError("Residual learner feasibility audit has wrong variant")
    passed = table["feasibility_passed"].astype(str).str.lower().isin(["true", "1"])
    eligible = set(table.loc[passed, "learner"].astype(str))
    ineligible = set(table.loc[~passed, "learner"].astype(str))
    if not eligible or "catboost" not in eligible:
        raise RuntimeError("No eligible residual CatBoost learner under Amendment 06")
    if not table.loc[passed, "downstream_status"].astype(str).eq("eligible").all():
        raise RuntimeError("Residual learner feasibility status is internally inconsistent")
    return eligible, ineligible, hashlib.sha256(path.read_bytes()).hexdigest()


def active_feature_sets(root: Path, variant: str) -> dict[str, list[str]]:
    if variant == "raw":
        return FEATURE_SETS
    eligible, _, _ = residual_learner_feasibility(root)
    keep = ["M0_har_rv", "M1_har_market"]
    if "linear" in eligible:
        keep.extend(["M4_linear_directed_current", "M5_linear_directed_dynamic"])
    if "catboost" in eligible:
        keep.extend(["M6_catboost_directed_current", "M7_catboost_directed_dynamic"])
    return {name: FEATURE_SETS[name] for name in keep}


def development_dir(root: Path, variant: str) -> Path:
    name = "benchmark_v10_directed_development" if variant == "raw" else "benchmark_v10_directed_development_residual"
    return root / name


def validate_residual_feature_artifact(root: Path, data: pd.DataFrame) -> None:
    """Reject stale or ineligible residual graph features before model fitting."""
    eligible, ineligible, feasibility_sha256 = residual_learner_feasibility(root)
    audit_path = root / "model_data" / "directed_feature_audit_residual.csv"
    if not audit_path.exists():
        raise FileNotFoundError(
            f"Missing residual directed feature audit: {audit_path}. "
            "Rebuild residual features with Amendment 06."
        )
    audit = pd.read_csv(audit_path, dtype=str)
    require_columns(audit, ["metric", "value"], "residual directed feature audit")
    if audit["metric"].duplicated().any():
        raise RuntimeError("Duplicate metrics in residual directed feature audit")
    values = dict(zip(audit["metric"], audit["value"]))
    if values.get("learner_feasibility_sha256") != feasibility_sha256:
        raise RuntimeError("Residual feature audit uses a stale learner feasibility file")
    included = set(filter(None, values.get("graph_learners_included", "").split("|")))
    excluded = set(filter(None, values.get("graph_learners_excluded", "").split("|")))
    if included != eligible or excluded != ineligible:
        raise RuntimeError("Residual feature learner set disagrees with Amendment 06")
    for learner in ineligible:
        leaked = [column for column in data if column.startswith(f"{learner}_")]
        if leaked:
            raise RuntimeError(
                f"Ineligible residual learner {learner} leaked into feature artifact: "
                f"{leaked[:5]}"
            )


def find_split(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    for rel in [
        "model_audit/directed_split_dates_v10.csv",
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
    raise FileNotFoundError("No frozen split_dates.csv; pass --split-dates")


@dataclass
class FixedEffectRidge:
    features: list[str]
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    beta: np.ndarray
    intercepts: dict[str, float]
    fallback: float
    rv_scale: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        z = (frame[self.features].to_numpy(float) - self.mean) / self.scale
        intercept = np.array([self.intercepts.get(str(s), self.fallback) for s in frame["SECID"]])
        log = intercept + z @ self.beta
        return np.maximum(np.exp(np.clip(log, -30, 5)) * self.rv_scale, 1e-12)


@dataclass
class DirectHistory:
    targets: pd.DataFrame
    sources: pd.DataFrame
    source_ids: list[str]


def fit_ridge(frame: pd.DataFrame, features: list[str], alpha: float) -> FixedEffectRidge:
    x = frame[features].to_numpy(float)
    y = frame["y_log_rv"].to_numpy(float)
    secid = frame["SECID"].astype(str)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, 1.0)
    z = (x - mean) / scale
    zdf = pd.DataFrame(z, columns=features, index=frame.index)
    zdf["SECID"] = secid.to_numpy()
    zw = z - zdf.groupby("SECID")[features].transform("mean").to_numpy()
    ys = pd.Series(y, index=frame.index)
    yw = y - ys.groupby(secid).transform("mean").to_numpy()
    gram = zw.T @ zw + float(alpha) * np.eye(len(features))
    try:
        beta = np.linalg.solve(gram, zw.T @ yw)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(gram) @ (zw.T @ yw)
    means = pd.DataFrame(z, columns=features)
    means["SECID"] = secid.to_numpy()
    means["y"] = y
    grouped = means.groupby("SECID")
    mean_x, mean_y = grouped[features].mean(), grouped["y"].mean()
    intercept_values = mean_y - mean_x.to_numpy() @ beta
    intercepts = dict(zip(mean_x.index.astype(str), intercept_values))
    fallback = float(y.mean() - z.mean(axis=0) @ beta)
    train_log = np.array([intercepts.get(s, fallback) for s in secid]) + z @ beta
    rv_scale = calibration_scale(frame["y_rv"].to_numpy(), np.exp(np.clip(train_log, -30, 5)))
    return FixedEffectRidge(features, alpha, mean, scale, beta, intercepts, fallback, rv_scale)


def load_data(a) -> tuple[pd.DataFrame, pd.DataFrame]:
    default_name = "directed_graph_features.parquet" if a.variant == "raw" else "residual_directed_graph_features.parquet"
    path = a.features or (a.root / "model_data" / default_name)
    data = normalize_keys(read_table(path))
    if a.variant == "residual":
        validate_residual_feature_artifact(a.root, data)
    data["y_rv"] = pd.to_numeric(data["y_rv"], errors="coerce")
    data["y_log_rv"] = np.log(data["y_rv"].where(data["y_rv"] > 0))
    split = normalize_keys(pd.read_csv(find_split(a.root, a.split_dates)), secid=False)
    split["split"] = split["split"].astype(str).str.lower()
    data = data.merge(split[["TRADEDATE", "split"]], on="TRADEDATE", how="left", validate="many_to_one")
    valid = data["directed_sample_valid"].fillna(False) & (data["y_rv"] > 0)
    data = data[valid].sort_values(["TRADEDATE", "SECID"]).reset_index(drop=True)
    required = sorted(set(sum(active_feature_sets(a.root, a.variant).values(), [])))
    if not np.isfinite(data[required]).all().all():
        raise RuntimeError("Non-finite core feature on directed common sample")
    return data, split


def load_direct_history(root: Path) -> DirectHistory:
    """Build M8's causal all-82 history without graph-validity truncation."""
    node = normalize_keys(read_table(root / "model_data/node_features_all82.parquet"))
    rv = normalize_keys(read_table(root / "model_data/rv_all82.parquet"))
    market = normalize_keys(
        read_table(root / "market_factor_v10/market_rv.parquet"), secid=False
    )
    require_columns(rv, ["rv_10m_core", "rv_valid"], "all-82 RV")
    har = build_har_panel(node, market)
    dates = recover_date_table(node)
    next_dates = dates[["TRADEDATE", "segment_id"]].copy()
    next_dates["next_date"] = next_dates.groupby("segment_id")["TRADEDATE"].shift(-1)

    targets = har[[
        "TRADEDATE", "SECID", "segment_id", *OWN_FEATURES, *MARKET_FEATURES,
        "har_w_count", "har_m_count", "mkt_w_count", "mkt_m_count",
    ]].merge(
        next_dates,
        on=["TRADEDATE", "segment_id"],
        how="left",
        validate="many_to_one",
    )
    labels = rv[["TRADEDATE", "SECID", "rv_10m_core", "rv_valid"]].rename(
        columns={
            "TRADEDATE": "next_date",
            "rv_10m_core": "y_rv",
            "rv_valid": "y_valid",
        }
    )
    if labels.duplicated(["next_date", "SECID"]).any():
        raise RuntimeError("Duplicate all-82 M8 target labels")
    targets = targets.merge(
        labels, on=["next_date", "SECID"], how="left", validate="many_to_one"
    )
    targets["y_rv"] = pd.to_numeric(targets["y_rv"], errors="coerce")
    finite_core = np.isfinite(targets[[*OWN_FEATURES, *MARKET_FEATURES]]).all(axis=1)
    valid = (
        targets["next_date"].notna()
        & targets["y_valid"].astype("boolean").fillna(False).astype(bool)
        & np.isfinite(targets["y_rv"])
        & (targets["y_rv"] > 0)
        & (targets["har_w_count"] >= 5)
        & (targets["har_m_count"] >= 22)
        & (targets["mkt_w_count"] >= 5)
        & (targets["mkt_m_count"] >= 22)
        & finite_core
    )
    targets = targets.loc[valid].copy()
    targets["y_log_rv"] = np.log(targets["y_rv"])
    if targets.duplicated(["TRADEDATE", "SECID"]).any():
        raise RuntimeError("Duplicate all-82 M8 history keys")

    source_ids = sorted(har["SECID"].astype(str).unique())
    wide_parts = []
    for horizon, column in (
        ("d", "har_logrv_d"),
        ("w", "har_logrv_w"),
        ("m", "har_logrv_m"),
    ):
        wide = har.pivot(index="TRADEDATE", columns="SECID", values=column).reindex(
            columns=source_ids
        )
        wide.columns = [f"src__{secid}__{horizon}" for secid in source_ids]
        wide_parts.append(wide)
    sources = pd.concat(wide_parts, axis=1).reset_index()
    return DirectHistory(
        targets=targets.sort_values(["TRADEDATE", "SECID"]).reset_index(drop=True),
        sources=sources,
        source_ids=source_ids,
    )


def model_names(a) -> list[str]:
    available = [*active_feature_sets(a.root, a.variant)]
    if a.variant == "raw":
        available += [*PLACEBO_FEATURE_SETS, "M8_direct_catboost"]
    if not a.models:
        return available
    names = [x.strip() for x in a.models.split(",") if x.strip()]
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise RuntimeError(f"Unknown models: {unknown}")
    return names


def score_dates(data: pd.DataFrame, period: str) -> list[pd.Timestamp]:
    if period == "val":
        values = data.loc[data["split"] == "val", "TRADEDATE"]
    elif period == "development":
        values = data.loc[data["split"] == "test", "TRADEDATE"]
    else:
        values = data.loc[data["TRADEDATE"] > DEVELOPMENT_END, "TRADEDATE"]
    return sorted(pd.to_datetime(values.dropna().unique()))


def train_rows(data: pd.DataFrame, first_score: pd.Timestamp, period: str) -> pd.DataFrame:
    train = data[data["TRADEDATE"] < first_score].copy()
    if period == "val":
        train = train[train["split"].isin(["train", "val"])]
    return train


def prediction_part(
    score: pd.DataFrame, pred_rv: np.ndarray, model: str, period: str,
    block_id: int, sealed: bool, parameter: str,
) -> pd.DataFrame:
    columns = ["TRADEDATE", "next_date", "SECID"]
    if not sealed:
        columns += ["y_rv", "y_log_rv", "stress_z"]
    out = score[columns].copy()
    out["model"] = model
    out["period"] = period
    out["block_id"] = block_id
    out["parameter"] = parameter
    out["pred_rv"] = pred_rv
    out["pred_log_rv"] = np.log(pred_rv)
    if not sealed:
        out["qlike"] = qlike(out["y_rv"].to_numpy(), pred_rv)
        out["squared_log_error"] = (out["y_log_rv"] - out["pred_log_rv"]) ** 2
        out["absolute_log_error"] = (out["y_log_rv"] - out["pred_log_rv"]).abs()
    return out


def alpha_selection(a, data: pd.DataFrame) -> None:
    out = development_dir(a.root, a.variant)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "alpha_selection_parts.csv"
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    feature_sets = active_feature_sets(a.root, a.variant)
    if not current.empty:
        current = current[current["model"].astype(str).isin(feature_sets)].copy()
    done = set(zip(current.get("model", []), current.get("alpha", [])))
    rows = current.to_dict("records")
    names = [n for n in model_names(a) if n in feature_sets]
    dates = score_dates(data, "val")
    if len(dates) < 60:
        raise RuntimeError(
            f"Directed alpha selection requires 60 validation dates; found {len(dates)}. "
            "Run build_directed_split_v01.py before training."
        )
    initial_train = train_rows(data, dates[0], "val")
    initial_train_dates = initial_train["TRADEDATE"].nunique()
    initial_train_secids = initial_train["SECID"].nunique()
    if initial_train_dates < 15 or initial_train_secids < 30 or len(initial_train) < 500:
        raise RuntimeError(
            "Insufficient directed common-sample warm-up before validation: "
            f"dates={initial_train_dates}, secids={initial_train_secids}, "
            f"rows={len(initial_train)}"
        )
    for model in names:
        features = feature_sets[model]
        for alpha in RIDGE_GRID:
            if (model, alpha) in done:
                continue
            losses = []
            for block in date_blocks(dates, 20):
                train = train_rows(data, block[0], "val")
                score = data[data["TRADEDATE"].isin(block)]
                fitted = fit_ridge(train, features, alpha)
                pred = fitted.predict(score)
                losses.extend(qlike(score["y_rv"].to_numpy(), pred))
            rows.append({"model": model, "alpha": alpha, "validation_qlike": np.mean(losses),
                         "rows": len(losses), "dates": len(dates)})
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"alpha {model} {alpha:g}: {np.mean(losses):.8f}")
    current = pd.DataFrame(rows)
    expected = set(feature_sets)
    if expected.issubset(current["model"].unique()):
        table = current[current["model"].isin(expected)].sort_values(
            ["model", "validation_qlike", "alpha"]
        ).copy()
        table["selected"] = table.groupby("model").cumcount() == 0
        table.to_csv(out / "hyperparameter_selection.csv", index=False)
        if a.variant == "residual":
            eligible, ineligible, feasibility_sha256 = residual_learner_feasibility(a.root)
            json_dump(out / "residual_benchmark_design.json", {
                "protocol_amendment": "V10 Amendment 06",
                "graph_learners_included": sorted(eligible),
                "graph_learners_excluded": sorted(ineligible),
                "learner_feasibility_sha256": feasibility_sha256,
                "active_models": sorted(expected),
                "linear_residual_forecast_losses_computed": False,
            })


def selected_alphas(root: Path, variant: str) -> dict[str, float]:
    table = pd.read_csv(development_dir(root, variant) / "hyperparameter_selection.csv")
    selected = table[table["selected"].astype(str).str.lower().isin(["true", "1"])]
    active = active_feature_sets(root, variant)
    selected = selected[selected["model"].astype(str).isin(active)]
    result = dict(zip(selected["model"].astype(str), selected["alpha"].astype(float)))
    missing = sorted(set(active) - set(result))
    if missing:
        raise RuntimeError(f"Missing selected alphas: {missing}")
    return result


def selected_catboost(root: Path, variant: str) -> tuple[str, dict]:
    suffix = "" if variant == "raw" else "_residual"
    table = pd.read_csv(root / "graphs_v10" / f"hyperparameter_selection{suffix}.csv")
    selected = table[(table["learner"] == "catboost")
                     & table["selected"].astype(str).str.lower().isin(["true", "1"])]
    if len(selected) != 1:
        raise RuntimeError("Expected one selected CatBoost graph configuration")
    config_id = str(selected.iloc[0]["config_id"])
    config = next(c for c in catboost_grid() if c.config_id == config_id)
    return config_id, config.params


def fit_direct_catboost(train: pd.DataFrame, features: list[str], params: dict):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("Install catboost: pip install catboost") from exc
    train_x = train[features].replace([np.inf, -np.inf], np.nan)
    model = CatBoostRegressor(**params, loss_function="RMSE", random_seed=260818,
                              thread_count=1, allow_writing_files=False, verbose=False)
    model.fit(train_x, train["y_log_rv"])
    raw = np.exp(np.clip(model.predict(train_x), -30, 5))
    scale = calibration_scale(train["y_rv"].to_numpy(), raw)
    return model, scale


def save_m8_history_audit(out: Path, period: str, row: dict) -> None:
    path = out / f"m8_history_coverage_{period}.csv"
    current = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not current.empty:
        current = current[
            ~((current["period"] == period) & (current["block_id"] == row["block_id"]))
        ]
    pd.concat([current, pd.DataFrame([row])], ignore_index=True).sort_values(
        ["period", "block_id"]
    ).to_csv(path, index=False)


def run(a, data: pd.DataFrame) -> None:
    period = a.period
    sealed = period == "confirmatory"
    if sealed and a.variant != "raw":
        raise RuntimeError("Confirmatory residual branch is run only after the raw holdout closes")
    out = a.root / "benchmark_v10_directed_confirmatory" if sealed else development_dir(a.root, a.variant)
    part_root = out / "prediction_parts" / f"period={period}"
    part_root.mkdir(parents=True, exist_ok=True)
    alphas = selected_alphas(a.root, a.variant)
    cb_id, cb_params = selected_catboost(a.root, a.variant)
    names = model_names(a)
    direct_history = load_direct_history(a.root) if "M8_direct_catboost" in names else None
    dates = score_dates(data, period)
    if not dates:
        raise RuntimeError(f"No scoring dates for {period}")
    if period == "val" and len(dates) < 60:
        raise RuntimeError(
            f"Validation run requires 60 directed-valid dates; found {len(dates)}"
        )
    if period == "development" and len(dates) < 120:
        raise RuntimeError(
            f"Development run requires at least 120 directed-valid dates; found {len(dates)}"
        )
    src = [c for c in data.columns if c.startswith("src__")]
    if direct_history is not None:
        expected_src = set(source_columns(direct_history.source_ids))
        if set(src) != expected_src:
            raise RuntimeError("M8/common source-column mismatch")
    for name in names:
        feature_sets = active_feature_sets(a.root, a.variant)
        if name in feature_sets:
            features, alpha = feature_sets[name], alphas[name]
        elif name in PLACEBO_FEATURE_SETS:
            features, alpha = PLACEBO_FEATURE_SETS[name], alphas["M7_catboost_directed_dynamic"]
        else:
            features, alpha = [*OWN_FEATURES, *MARKET_FEATURES, *src], None
        for block_id, block in enumerate(date_blocks(dates, 20), 1):
            if a.block_id is not None and block_id != a.block_id:
                continue
            path = part_root / f"model={name}" / f"block={block_id:04d}.parquet"
            if path.exists() and not a.force:
                continue
            score = data[data["TRADEDATE"].isin(block)].copy()
            if name == "M8_direct_catboost":
                if score.empty or direct_history is None:
                    raise RuntimeError(f"Empty M8 score block: {block_id}")
                segments = score["segment_id"].dropna().astype(int).unique()
                if len(segments) != 1:
                    raise RuntimeError(f"M8 block spans temporal segments: {block_id}")
                segment_id = int(segments[0])
                pieces = []
                history_counts = []
                fitted_targets = []
                skipped_targets = []
                for target, target_score in score.groupby("SECID", sort=True):
                    target_train = direct_history.targets[
                        (direct_history.targets["SECID"] == target)
                        & (direct_history.targets["segment_id"] == segment_id)
                        & (direct_history.targets["TRADEDATE"] < block[0])
                    ].copy()
                    history_counts.append(len(target_train))
                    if len(target_train) < 400:
                        skipped_targets.append(str(target))
                        continue
                    target_train = target_train.merge(
                        direct_history.sources,
                        on="TRADEDATE",
                        how="left",
                        validate="many_to_one",
                    )
                    if target_train["TRADEDATE"].max() >= target_score["TRADEDATE"].min():
                        raise RuntimeError(f"M8 chronology failure: {target}/{block_id}")
                    target_source = [f"src__{target}__{h}" for h in ("d", "w", "m")]
                    target_features = [column for column in features if column not in target_source]
                    fitted, scale = fit_direct_catboost(target_train, target_features, cb_params)
                    score_x = target_score[target_features].replace([np.inf, -np.inf], np.nan)
                    pred = np.maximum(
                        np.exp(np.clip(fitted.predict(score_x), -30, 5)) * scale,
                        1e-12,
                    )
                    pieces.append(prediction_part(
                        target_score, pred, name, period, block_id, sealed,
                        f"{cb_id};history=all82_current_segment",
                    ))
                    fitted_targets.append(str(target))
                if not pieces:
                    raise RuntimeError(f"No direct CatBoost predictions in block {block_id}")
                result = pd.concat(pieces, ignore_index=True)
                save_m8_history_audit(out, period, {
                    "period": period,
                    "block_id": block_id,
                    "forecast_start": str(pd.Timestamp(block[0]).date()),
                    "forecast_end": str(pd.Timestamp(block[-1]).date()),
                    "score_rows": len(score),
                    "prediction_rows": len(result),
                    "targets_attempted": score["SECID"].nunique(),
                    "targets_fitted": len(fitted_targets),
                    "targets_skipped_lt400": len(skipped_targets),
                    "minimum_history_rows": min(history_counts),
                    "median_history_rows": float(np.median(history_counts)),
                    "maximum_history_rows": max(history_counts),
                    "history_scope": "all82_current_segment",
                })
                path.parent.mkdir(parents=True, exist_ok=True)
                result.to_parquet(path, index=False)
                print(
                    f"{period} {name}: block {block_id}/{math.ceil(len(dates)/20)} "
                    f"rows={len(result)} targets={len(fitted_targets)} "
                    f"skipped_lt400={len(skipped_targets)}"
                )
                continue

            train = train_rows(data, block[0], period)
            if name in PLACEBO_FEATURE_SETS:
                train = train[np.isfinite(train[features]).all(axis=1)].copy()
                score = score[np.isfinite(score[features]).all(axis=1)].copy()
            if train.empty or score.empty or train["TRADEDATE"].max() >= score["TRADEDATE"].min():
                raise RuntimeError(f"Chronology/empty block failure: {name}/{block_id}")
            if not np.isfinite(train[features]).all().all() or not np.isfinite(score[features]).all().all():
                raise RuntimeError(f"Non-finite features for {name}/{block_id}")
            fitted = fit_ridge(train, features, alpha)
            result = prediction_part(score, fitted.predict(score), name, period, block_id, sealed, f"alpha={alpha:g}")
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_parquet(path, index=False)
            print(f"{period} {name}: block {block_id}/{math.ceil(len(dates)/20)} rows={len(result)}")


def assemble(a) -> None:
    period = a.period
    sealed = period == "confirmatory"
    out = a.root / "benchmark_v10_directed_confirmatory" if sealed else development_dir(a.root, a.variant)
    files = sorted((out / "prediction_parts" / f"period={period}").glob("model=*/block=*.parquet"))
    if not files:
        raise RuntimeError("No prediction parts")
    predictions = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    models = set(predictions["model"])
    feature_sets = active_feature_sets(a.root, a.variant)
    if a.variant == "residual":
        unexpected = sorted(models - set(feature_sets))
        if unexpected:
            raise RuntimeError(
                "Ineligible or stale residual prediction parts detected: "
                f"{unexpected}. Move them out of the residual prediction_parts "
                "directory before assembly."
            )
    missing = sorted(set(feature_sets) - models)
    if missing:
        raise RuntimeError(f"Core prediction models missing: {missing}")
    keys = ["TRADEDATE", "SECID"]
    hashes = []
    for model in feature_sets:
        frame = predictions[predictions["model"] == model].sort_values(keys)
        if frame.duplicated(keys).any():
            raise RuntimeError(f"Duplicate keys for {model}")
        hash_cols = keys if sealed else ["TRADEDATE", "SECID", "y_rv"]
        hashes.append(hash_frame(frame, hash_cols))
    if len(set(hashes)) != 1:
        raise RuntimeError("Core paired sample hash mismatch")
    if (predictions["pred_rv"] <= 0).any() or not np.isfinite(predictions["pred_rv"]).all():
        raise RuntimeError("Invalid predictions")
    filename = {
        "val": "predictions_val.parquet",
        "development": "predictions_development.parquet",
        "confirmatory": "predictions_sealed.parquet",
    }[period]
    predictions.to_parquet(out / filename, index=False)
    audit = predictions.groupby("model", as_index=False).agg(
        rows=("SECID", "size"), dates=("TRADEDATE", "nunique"), secids=("SECID", "nunique"),
        nonpositive_predictions=("pred_rv", lambda x: int((x <= 0).sum())),
    )
    audit["sealed_no_targets_or_losses"] = sealed
    audit.to_csv(out / ("technical_coverage_no_losses.csv" if sealed else f"sample_coverage_{period}.csv"), index=False)
    print(audit.to_string(index=False))


def freeze(a) -> None:
    if a.variant != "raw":
        raise RuntimeError("Only the primary raw branch freezes the confirmatory design")
    alphas = selected_alphas(a.root, a.variant)
    cb_id, cb_params = selected_catboost(a.root, a.variant)
    source_files = [
        a.root / "benchmark_v10_directed_development" / "hyperparameter_selection.csv",
        a.root / "graphs_v10" / "hyperparameter_selection.csv",
        a.root / "graphs_v10" / "graph_design.json",
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    payload = {
        "frozen_at_protocol_date": "2026-08-18", "development_end": "2026-08-10",
        "ridge_alphas": alphas, "catboost_config_id": cb_id,
        "catboost_params": cb_params, "source_sha256": hashes,
    }
    out = a.root / "benchmark_v10_directed_confirmatory"
    json_dump(out / "frozen_hyperparameters.json", payload)
    json_dump(out / "frozen_design.json", {
        "holdout_valid_dates_required": 120, "losses_sealed_until_close": True,
        "calendar_cap": "2027-03-31", "target": "next-day realized variance",
        "primary_loss": "QLIKE", "protocol": "v10 frozen 2026-08-18",
    })
    print(out / "frozen_hyperparameters.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data_v02"))
    p.add_argument("--variant", choices=["raw", "residual"], default="raw")
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--split-dates", type=Path, default=None)
    p.add_argument("--stage", choices=["select", "run", "assemble", "freeze"], required=True)
    p.add_argument("--period", choices=["val", "development", "confirmatory"], default="development")
    p.add_argument("--models", default=None, help="Comma-separated shard of model ids")
    p.add_argument("--block-id", type=int, default=None, help="Optional forecasting-block shard")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.stage == "freeze":
        freeze(a)
        return
    data, _ = load_data(a)
    if a.stage == "select":
        alpha_selection(a, data)
    elif a.stage == "run":
        run(a, data)
    else:
        assemble(a)


if __name__ == "__main__":
    main()
