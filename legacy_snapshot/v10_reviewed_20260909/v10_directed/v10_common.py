"""Shared, leakage-safe utilities for the frozen MOEX directed-network v10.

The module intentionally contains no outcome-driven defaults.  Scientific
constants mirror RESEARCH_PROTOCOL_V10_DIRECTED_VOLATILITY_NETWORKS.md.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12
DEVELOPMENT_END = pd.Timestamp("2026-08-10")
GRAPH_TRAIN_DATES = 504
IMPORTANCE_DATES = 126
REFIT_DATES = 20
TOP_K = 5

OWN_FEATURES = ["har_logrv_d", "har_logrv_w", "har_logrv_m"]
MARKET_FEATURES = ["mkt_logrv_d", "mkt_logrv_w", "mkt_logrv_m"]


def read_table(path: Path) -> pd.DataFrame:
    """Read Parquet, falling back to a same-stem CSV."""
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"Neither {path} nor {csv_path} exists")


def write_table(frame: pd.DataFrame, path: Path, csv: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    if csv:
        frame.to_csv(path.with_suffix(".csv"), index=False)


def normalize_keys(frame: pd.DataFrame, secid: bool = True) -> pd.DataFrame:
    frame = frame.copy()
    if "TRADEDATE" in frame:
        frame["TRADEDATE"] = pd.to_datetime(frame["TRADEDATE"]).dt.normalize()
    if "next_date" in frame:
        frame["next_date"] = pd.to_datetime(frame["next_date"]).dt.normalize()
    if secid and "SECID" in frame:
        frame["SECID"] = frame["SECID"].astype(str)
    return frame


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{name} is missing columns: {missing}")


def qlike(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(y, float), EPS)
    prediction = np.maximum(np.asarray(prediction, float), EPS)
    return np.log(prediction) + y / prediction


def calibration_scale(y_rv: np.ndarray, raw_pred_rv: np.ndarray) -> float:
    raw = np.maximum(np.asarray(raw_pred_rv, float), EPS)
    y = np.maximum(np.asarray(y_rv, float), EPS)
    value = float(np.mean(y / raw))
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError("Invalid train-only QLIKE calibration scale")
    return value


def strict_rolling_mean(
    values: pd.Series, groupers: list[pd.Series], window: int
) -> tuple[pd.Series, pd.Series]:
    grouped = values.groupby(groupers, sort=False)
    mean = grouped.rolling(window, min_periods=window).mean()
    count = grouped.rolling(window, min_periods=1).count()
    levels = list(range(len(groupers)))
    mean = mean.reset_index(level=levels, drop=True).sort_index()
    count = count.reset_index(level=levels, drop=True).sort_index()
    return mean, count


def recover_date_table(node: pd.DataFrame) -> pd.DataFrame:
    require_columns(node, ["TRADEDATE", "segment_id"], "node features")
    dates = (
        node.loc[node["segment_id"].notna(), ["TRADEDATE", "segment_id"]]
        .drop_duplicates()
        .sort_values("TRADEDATE")
    )
    counts = dates.groupby("TRADEDATE")["segment_id"].nunique()
    if (counts != 1).any():
        raise RuntimeError("A research date maps to multiple temporal segments")
    dates = dates.drop_duplicates("TRADEDATE").reset_index(drop=True)
    dates["segment_id"] = dates["segment_id"].astype(int)
    dates["segment_index"] = dates.groupby("segment_id").cumcount()
    return dates


def build_har_panel(
    node: pd.DataFrame,
    market: pd.DataFrame,
    value_column: str = "log_rv",
    valid_column: str = "rv_feature_valid",
) -> pd.DataFrame:
    """Create 1/5/22 asset and IMOEX HAR features without crossing segments."""
    node = normalize_keys(node)
    market = normalize_keys(market, secid=False)
    dates = recover_date_table(node)
    node = node.drop(columns=["segment_id"], errors="ignore").merge(
        dates[["TRADEDATE", "segment_id"]],
        on="TRADEDATE",
        how="left",
        validate="many_to_one",
    )
    if node.duplicated(["TRADEDATE", "SECID"]).any():
        raise RuntimeError("Duplicate node feature keys")
    node = node.sort_values(["SECID", "segment_id", "TRADEDATE"]).reset_index(drop=True)
    value = pd.to_numeric(node[value_column], errors="coerce")
    if valid_column in node:
        value = value.where(node[valid_column].fillna(False).astype(bool))
    node["har_logrv_d"] = value
    groups = [node["SECID"], node["segment_id"]]
    node["har_logrv_w"], node["har_w_count"] = strict_rolling_mean(value, groups, 5)
    node["har_logrv_m"], node["har_m_count"] = strict_rolling_mean(value, groups, 22)

    m = market.drop(columns=["segment_id"], errors="ignore").merge(
        dates[["TRADEDATE", "segment_id"]],
        on="TRADEDATE",
        how="inner",
        validate="one_to_one",
    ).sort_values(["segment_id", "TRADEDATE"]).reset_index(drop=True)
    require_columns(m, ["market_log_rv", "market_rv_valid"], "market RV")
    mv = pd.to_numeric(m["market_log_rv"], errors="coerce").where(
        m["market_rv_valid"].fillna(False).astype(bool)
    )
    m["mkt_logrv_d"] = mv
    m["mkt_logrv_w"], m["mkt_w_count"] = strict_rolling_mean(
        mv, [m["segment_id"]], 5
    )
    m["mkt_logrv_m"], m["mkt_m_count"] = strict_rolling_mean(
        mv, [m["segment_id"]], 22
    )
    node = node.merge(
        m[["TRADEDATE", *MARKET_FEATURES, "mkt_w_count", "mkt_m_count"]],
        on="TRADEDATE",
        how="left",
        validate="many_to_one",
    )
    return node


def source_column(secid: str, horizon: str) -> str:
    return f"src__{secid}__{horizon}"


def source_columns(secids: Iterable[str]) -> list[str]:
    return [source_column(str(s), h) for s in secids for h in ("d", "w", "m")]


def make_wide_model_frame(
    har: pd.DataFrame, samples: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """One row per target/date with all source HAR groups attached."""
    har = normalize_keys(har)
    samples = normalize_keys(samples)
    require_columns(
        samples,
        ["TRADEDATE", "SECID", "next_date", "y_rv", "sample_valid"],
        "forecast samples",
    )
    source_ids = sorted(har["SECID"].unique())
    wide_parts = []
    for h, col in (("d", "har_logrv_d"), ("w", "har_logrv_w"), ("m", "har_logrv_m")):
        w = har.pivot(index="TRADEDATE", columns="SECID", values=col).reindex(columns=source_ids)
        w.columns = [source_column(s, h) for s in source_ids]
        wide_parts.append(w)
    wide = pd.concat(wide_parts, axis=1).reset_index()
    own = har[["TRADEDATE", "SECID", "segment_id", *OWN_FEATURES, *MARKET_FEATURES,
               "har_w_count", "har_m_count", "mkt_w_count", "mkt_m_count"]]
    out = samples.merge(
        own,
        on=["TRADEDATE", "SECID"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    if "segment_id_feature" in out:
        if "segment_id" in out:
            mismatch = (
                out["segment_id"].notna()
                & out["segment_id_feature"].notna()
                & (out["segment_id"] != out["segment_id_feature"])
            )
            if mismatch.any():
                raise RuntimeError("Sample/feature segment_id mismatch")
            out["segment_id"] = out["segment_id"].combine_first(out["segment_id_feature"])
        else:
            out["segment_id"] = out["segment_id_feature"]
        out = out.drop(columns=["segment_id_feature"])
    out = out.merge(wide, on="TRADEDATE", how="left", validate="many_to_one")
    out["y_rv"] = pd.to_numeric(out["y_rv"], errors="coerce")
    out["y_log_rv"] = np.log(out["y_rv"].where(out["y_rv"] > 0))
    return out.sort_values(["TRADEDATE", "SECID"]).reset_index(drop=True), source_ids


def graph_refit_calendar(date_table: pd.DataFrame) -> pd.DataFrame:
    """Common 20-date clock; every block carries its strictly prior windows."""
    rows = []
    block_id = 0
    for segment_id, group in date_table.groupby("segment_id", sort=True):
        dates = list(pd.to_datetime(group["TRADEDATE"]).sort_values())
        for start_index in range(0, len(dates), REFIT_DATES):
            block_id += 1
            block = dates[start_index : start_index + REFIT_DATES]
            imp_end = start_index
            imp_start = imp_end - IMPORTANCE_DATES
            train_end = imp_start
            train_start = train_end - GRAPH_TRAIN_DATES
            windows_valid = train_start >= 0
            rows.append({
                "block_id": block_id,
                "segment_id": int(segment_id),
                "refit_date": block[0],
                "forecast_start": block[0],
                "forecast_end": block[-1],
                "forecast_dates": len(block),
                "train_start": dates[train_start] if windows_valid else pd.NaT,
                "train_end": dates[train_end - 1] if windows_valid else pd.NaT,
                "importance_start": dates[imp_start] if windows_valid else pd.NaT,
                "importance_end": dates[imp_end - 1] if windows_valid else pd.NaT,
                "windows_valid": bool(windows_valid),
            })
    return pd.DataFrame(rows)


def dates_between(date_table: pd.DataFrame, segment_id: int, start, end) -> list[pd.Timestamp]:
    mask = (
        (date_table["segment_id"] == int(segment_id))
        & (date_table["TRADEDATE"] >= pd.Timestamp(start))
        & (date_table["TRADEDATE"] <= pd.Timestamp(end))
    )
    return list(pd.to_datetime(date_table.loc[mask, "TRADEDATE"]).sort_values())


def deterministic_shifts(n_dates: int, seed: int = 260819, reps: int = 20) -> list[int]:
    allowed = np.arange(20, max(20, n_dates - 19), dtype=int)
    if len(allowed) == 0:
        raise RuntimeError(f"No circular shift >=20 is possible for {n_dates} dates")
    rng = np.random.default_rng(seed)
    replace = len(allowed) < reps
    return [int(x) for x in rng.choice(allowed, size=reps, replace=replace)]


def shift_group_by_dates(
    frame: pd.DataFrame, columns: list[str], shift: int
) -> pd.DataFrame:
    """Circularly shift a source group on the ordered date axis, not row axis."""
    ordered = frame.sort_values("TRADEDATE").copy()
    values = ordered[columns].to_numpy(copy=True)
    ordered.loc[:, columns] = np.roll(values, shift=int(shift), axis=0)
    return ordered.reindex(frame.index)


def hash_frame(frame: pd.DataFrame, columns: list[str]) -> str:
    copy = frame[columns].copy()
    for column in copy:
        if pd.api.types.is_datetime64_any_dtype(copy[column]):
            copy[column] = copy[column].dt.strftime("%Y-%m-%d")
    payload = copy.sort_values(columns[:2]).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def date_blocks(dates: list[pd.Timestamp], size: int = REFIT_DATES) -> list[list[pd.Timestamp]]:
    return [dates[i : i + size] for i in range(0, len(dates), size)]


def hac_mean_test(values: np.ndarray, lags: int = 10) -> dict[str, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise RuntimeError("At least two daily differences are required")
    mean = float(x.mean())
    centered = x - mean
    cap = min(int(lags), len(x) - 1)
    long_run = float(centered @ centered / len(x))
    for lag in range(1, cap + 1):
        gamma = float(centered[lag:] @ centered[:-lag] / len(x))
        long_run += 2.0 * (1.0 - lag / (cap + 1.0)) * gamma
    se = math.sqrt(max(long_run, 0.0) / len(x))
    z = mean / se if se > 0 else np.nan
    return {
        "mean_difference": mean,
        "hac_lags": cap,
        "hac_se": se,
        "hac_z": z,
        "hac_p_two_sided": math.erfc(abs(z) / math.sqrt(2.0)) if np.isfinite(z) else np.nan,
        "hac_p_one_sided": 0.5 * math.erfc(z / math.sqrt(2.0)) if np.isfinite(z) else np.nan,
    }


def circular_block_bootstrap(
    values: np.ndarray, block: int = 20, reps: int = 10_000, seed: int = 260821
) -> dict[str, float]:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise RuntimeError("At least two daily differences are required")
    b = min(int(block), max(1, len(x) // 2))
    n_blocks = math.ceil(len(x) / b)
    offsets = np.arange(b)
    rng = np.random.default_rng(seed)
    centered = x - x.mean()
    means = np.empty(reps)
    null = np.empty(reps)
    for rep in range(reps):
        starts = rng.integers(0, len(x), n_blocks)
        idx = ((starts[:, None] + offsets) % len(x)).ravel()[: len(x)]
        means[rep] = x[idx].mean()
        null[rep] = centered[idx].mean()
    observed = float(x.mean())
    return {
        "bootstrap_block": b,
        "bootstrap_reps": reps,
        "bootstrap_ci_low": float(np.quantile(means, 0.025)),
        "bootstrap_ci_high": float(np.quantile(means, 0.975)),
        "bootstrap_p_two_sided": float((1 + (np.abs(null) >= abs(observed)).sum()) / (reps + 1)),
        "bootstrap_p_one_sided": float((1 + (null >= observed).sum()) / (reps + 1)),
    }


def holm(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[idx]))
        out[idx] = running
    return pd.Series(out, index=values.index)


@dataclass(frozen=True)
class GraphConfig:
    learner: str
    config_id: str
    params: dict


def catboost_grid() -> list[GraphConfig]:
    out = []
    k = 0
    for depth in (3, 5):
        for learning_rate in (0.03, 0.05):
            for iterations in (300, 600):
                for l2_leaf_reg in (3, 10):
                    k += 1
                    out.append(GraphConfig("catboost", f"cb{k:02d}", {
                        "depth": depth,
                        "learning_rate": learning_rate,
                        "iterations": iterations,
                        "l2_leaf_reg": l2_leaf_reg,
                    }))
    return out


def linear_grid() -> list[GraphConfig]:
    out = []
    k = 0
    for alpha in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        for l1_ratio in (0.25, 0.50, 0.75):
            k += 1
            out.append(GraphConfig("linear", f"en{k:02d}", {
                "alpha": alpha, "l1_ratio": l1_ratio, "max_iter": 10_000
            }))
    return out
