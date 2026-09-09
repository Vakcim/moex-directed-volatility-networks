"""Build strict IMOEX RV and causal stock-market residual realized variance.

For each stock/day, alpha and beta are estimated from the previous 60 valid
days (minimum 40) inside the same temporal segment and are never updated with
the current day.  The script processes one SECID at a time to bound memory.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from v10_common import EPS, json_dump, normalize_keys, read_table, require_columns, write_table


EXPECTED_BINS = 52
BETA_WINDOW = 60
MIN_BETA_DAYS = 40
ASSET_MIN_COVERAGE = 0.90


def core_grid(day: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(day.normalize() + pd.Timedelta(hours=10), periods=52, freq="10min")


def clean_candles(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["begin"] = pd.to_datetime(frame["begin"], errors="coerce")
    for column in ("open", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["begin", "open", "close"])
    frame = frame[(frame["open"] > 0) & (frame["close"] > 0)]
    minute = frame["begin"].dt.hour * 60 + frame["begin"].dt.minute
    frame = frame[
        (minute >= 600)
        & (minute < 1120)
        & (frame["begin"].dt.minute % 10 == 0)
        & (frame["begin"].dt.second == 0)
    ]
    frame["TRADEDATE"] = frame["begin"].dt.normalize()
    return frame.drop_duplicates(["TRADEDATE", "begin"], keep="last")


def intraday_returns(day_frame: pd.DataFrame, day: pd.Timestamp) -> tuple[np.ndarray, int]:
    grid = core_grid(day)
    group = day_frame.sort_values("begin").drop_duplicates("begin", keep="last")
    observed = int(group["begin"].isin(grid).sum())
    if group.empty:
        return np.full(EXPECTED_BINS, np.nan), 0
    first = group.iloc[0]
    p0 = float(first["open"])
    close = group.set_index("begin")["close"].reindex(grid)
    endpoints = close.ffill().fillna(p0).to_numpy(float)
    prices = np.r_[p0, endpoints]
    if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        return np.full(EXPECTED_BINS, np.nan), observed
    return np.diff(np.log(prices)), observed


def market_panel(raw: pd.DataFrame, date_table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw = clean_candles(raw)
    groups = {pd.Timestamp(d): g for d, g in raw.groupby("TRADEDATE", sort=False)}
    vectors = {}
    rows = []
    empty = raw.iloc[:0]
    for record in date_table.itertuples(index=False):
        day = pd.Timestamp(record.TRADEDATE)
        returns, observed = intraday_returns(groups.get(day, empty), day)
        valid = observed == EXPECTED_BINS and np.isfinite(returns).all()
        rv = float(returns @ returns) if valid else np.nan
        if valid:
            vectors[day] = returns
        rows.append({
            "TRADEDATE": day,
            "segment_id": int(record.segment_id),
            "market_observed_bins": observed,
            "market_rv": rv,
            "market_log_rv": np.log(max(rv, EPS)) if valid and rv > 0 else np.nan,
            "market_rv_valid": bool(valid and rv > 0),
        })
    return pd.DataFrame(rows), vectors


def stock_files(root: Path, secid: str) -> list[Path]:
    return sorted((root / "raw" / "moex_tqbr_10m" / f"SECID={secid}").glob("year=*/month=*/part.parquet"))


def load_stock(root: Path, secid: str) -> pd.DataFrame:
    files = stock_files(root, secid)
    if not files:
        return pd.DataFrame(columns=["begin", "open", "close", "TRADEDATE"])
    parts = []
    for path in files:
        frame = pd.read_parquet(path)
        if len(frame):
            parts.append(frame[[c for c in ("begin", "open", "close") if c in frame]])
    if not parts:
        return pd.DataFrame(columns=["begin", "open", "close", "TRADEDATE"])
    return clean_candles(pd.concat(parts, ignore_index=True))


def fit_ols(history: deque[tuple[pd.Timestamp, np.ndarray, np.ndarray]]) -> tuple[float, float]:
    market = np.concatenate([x for _, x, _ in history])
    asset = np.concatenate([y for _, _, y in history])
    design = np.column_stack([np.ones(len(market)), market])
    coef, *_ = np.linalg.lstsq(design, asset, rcond=None)
    return float(coef[0]), float(coef[1])


def process_stock(
    root: Path,
    secid: str,
    date_table: pd.DataFrame,
    market_vectors: dict,
    rv_lookup: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    raw = load_stock(root, secid)
    groups = {pd.Timestamp(d): g for d, g in raw.groupby("TRADEDATE", sort=False)}
    rv_sec = rv_lookup[rv_lookup["SECID"] == secid].set_index("TRADEDATE")
    empty = raw.iloc[:0]
    betas = []
    irv = []
    for segment_id, segment in date_table.groupby("segment_id", sort=True):
        history: deque[tuple[pd.Timestamp, np.ndarray, np.ndarray]] = deque(maxlen=BETA_WINDOW)
        for record in segment.sort_values("TRADEDATE").itertuples(index=False):
            day = pd.Timestamp(record.TRADEDATE)
            market = market_vectors.get(day)
            asset, observed = intraday_returns(groups.get(day, empty), day)
            rv_valid = bool(
                day in rv_sec.index
                and rv_sec.loc[day, "rv_valid"]
                and observed / EXPECTED_BINS >= ASSET_MIN_COVERAGE
            )
            current_valid = market is not None and rv_valid and np.isfinite(asset).all()
            n_history_days = len(history)
            alpha = beta = np.nan
            residual_rv = np.nan
            causal_valid = current_valid and n_history_days >= MIN_BETA_DAYS
            if causal_valid:
                alpha, beta = fit_ols(history)
                residual = asset - alpha - beta * market
                residual_rv = float(residual @ residual)
                causal_valid = bool(np.isfinite(residual_rv) and residual_rv > 0)
            betas.append({
                "TRADEDATE": day,
                "segment_id": int(segment_id),
                "SECID": secid,
                "alpha": alpha,
                "beta": beta,
                "beta_history_days": n_history_days,
                "beta_window_start": history[0][0] if history else pd.NaT,
                "beta_window_end": history[-1][0] if history else pd.NaT,
                "asset_observed_bins": observed,
                "current_joint_valid": current_valid,
                "causal_beta_valid": causal_valid,
            })
            irv.append({
                "TRADEDATE": day,
                "segment_id": int(segment_id),
                "SECID": secid,
                "idiosyncratic_rv": residual_rv,
                "log_idiosyncratic_rv": np.log(residual_rv) if causal_valid else np.nan,
                "irv_valid": causal_valid,
            })
            # Updating happens strictly after the current day's residual was computed.
            if current_valid:
                history.append((day, market.copy(), asset.copy()))
    return betas, irv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data_v02"))
    p.add_argument("--secids", default=None, help="Optional comma-separated shard")
    p.add_argument("--assemble", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out = a.root / "market_factor_v10"
    parts = out / "residual_parts"
    parts.mkdir(parents=True, exist_ok=True)
    node = normalize_keys(read_table(a.root / "model_data" / "node_features_all82.parquet"))
    date_table = (
        node.loc[node["segment_id"].notna(), ["TRADEDATE", "segment_id"]]
        .drop_duplicates()
        .sort_values("TRADEDATE")
    )
    if date_table.groupby("TRADEDATE")["segment_id"].nunique().max() != 1:
        raise RuntimeError("Ambiguous segment_id")
    date_table = date_table.drop_duplicates("TRADEDATE")

    imoex = read_table(out / "imoex_10m.parquet")
    market, market_vectors = market_panel(imoex, date_table)
    write_table(market, out / "market_rv.parquet", csv=True)

    if a.assemble:
        beta_files = sorted(parts.glob("SECID=*/asset_market_betas.parquet"))
        irv_files = sorted(parts.glob("SECID=*/idiosyncratic_rv.parquet"))
        expected_secids = set(pd.read_csv(a.root / "universe_v03" / "intraday_download_secids.csv")["SECID"].dropna().astype(str))
        found_secids = {p.parent.name.split("=", 1)[1] for p in irv_files}
        if not beta_files or len(beta_files) != len(irv_files) or found_secids != expected_secids:
            raise RuntimeError("Residual shards are missing or mismatched")
        betas = pd.concat([pd.read_parquet(p) for p in beta_files], ignore_index=True)
        irv = pd.concat([pd.read_parquet(p) for p in irv_files], ignore_index=True)
        write_table(betas, out / "asset_market_betas.parquet")
        write_table(irv, out / "idiosyncratic_rv.parquet")
        universe = normalize_keys(pd.read_csv(a.root / "universe_v03" / "dynamic_universe.csv"))
        selected = universe[["TRADEDATE", "SECID"]].merge(
            irv[["TRADEDATE", "SECID", "irv_valid"]], on=["TRADEDATE", "SECID"], how="left"
        )
        coverage = pd.DataFrame([
            {"metric": "imoex_strict_valid_date_rate", "value": market["market_rv_valid"].mean(), "gate": 0.95},
            {"metric": "selected_causal_irv_row_rate", "value": selected["irv_valid"].fillna(False).mean(), "gate": 0.90},
        ])
        coverage["passed"] = coverage["value"] >= coverage["gate"]
        coverage.to_csv(out / "market_factor_coverage.csv", index=False)
        json_dump(out / "market_factor_design.json", {
            "market": "IMOEX 10-minute returns", "expected_bins": EXPECTED_BINS,
            "beta_window_previous_valid_days": BETA_WINDOW,
            "minimum_beta_days": MIN_BETA_DAYS, "intercept": True,
            "current_day_used_for_beta_fit": False,
            "residual_branch_gate_passed": bool(coverage["passed"].all()),
        })
        print(coverage.to_string(index=False))
        return

    secid_file = a.root / "universe_v03" / "intraday_download_secids.csv"
    all_secids = sorted(pd.read_csv(secid_file)["SECID"].dropna().astype(str).unique())
    requested = all_secids if a.secids is None else [x.strip() for x in a.secids.split(",") if x.strip()]
    unknown = sorted(set(requested) - set(all_secids))
    if unknown:
        raise RuntimeError(f"Unknown SECIDs: {unknown}")
    rv = normalize_keys(read_table(a.root / "model_data" / "rv_all82.parquet"))
    require_columns(rv, ["TRADEDATE", "SECID", "rv_valid"], "rv_all82")
    for number, secid in enumerate(requested, 1):
        target = parts / f"SECID={secid}"
        target.mkdir(parents=True, exist_ok=True)
        beta_path = target / "asset_market_betas.parquet"
        irv_path = target / "idiosyncratic_rv.parquet"
        if beta_path.exists() and irv_path.exists() and not a.force:
            print(f"[{number}/{len(requested)}] checkpoint {secid}")
            continue
        beta_rows, irv_rows = process_stock(a.root, secid, date_table, market_vectors, rv)
        pd.DataFrame(beta_rows).to_parquet(beta_path, index=False)
        pd.DataFrame(irv_rows).to_parquet(irv_path, index=False)
        print(f"[{number}/{len(requested)}] {secid}: {sum(r['irv_valid'] for r in irv_rows):,} valid IRV")
    print("Run again with --assemble after all SECID shards exist.")


if __name__ == "__main__":
    main()
