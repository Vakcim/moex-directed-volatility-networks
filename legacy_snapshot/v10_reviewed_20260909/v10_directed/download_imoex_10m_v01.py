"""Checkpointed MOEX ISS downloader for 10-minute IMOEX candles.

Output follows section 18.2 of the frozen v10 protocol.  Raw responses are
checkpointed by month and then assembled into one canonical parquet file.
"""

from __future__ import annotations

import argparse
import calendar
import time
from pathlib import Path
from typing import Any

import pandas as pd


BASE = "https://iss.moex.com/iss"
SECID = "IMOEX"
BOARD = "SNDX"
INTERVAL = 10


def session() -> Any:
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:
        raise RuntimeError("Install downloader dependencies: pip install requests") from exc
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "academic-moex-directed-volatility-v10/1.0"})
    return s


def month_ranges(start: pd.Timestamp, end: pd.Timestamp):
    current = pd.Timestamp(start.year, start.month, 1)
    last_month = pd.Timestamp(end.year, end.month, 1)
    while current <= last_month:
        last_day = calendar.monthrange(current.year, current.month)[1]
        d0 = max(start.normalize(), current)
        d1 = min(end.normalize(), pd.Timestamp(current.year, current.month, last_day))
        yield d0, d1
        current += pd.offsets.MonthBegin(1)


def page(s: Any, d0: pd.Timestamp, d1: pd.Timestamp, start: int) -> pd.DataFrame:
    url = (
        f"{BASE}/engines/stock/markets/index/boards/{BOARD}"
        f"/securities/{SECID}/candles.json"
    )
    response = s.get(
        url,
        params={
            "from": d0.date().isoformat(),
            "till": d1.date().isoformat(),
            "interval": INTERVAL,
            "start": int(start),
            "iss.meta": "off",
            "iss.only": "candles",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if "candles" not in payload:
        raise RuntimeError(f"ISS response has no candles block: {list(payload)}")
    block = payload["candles"]
    return pd.DataFrame(block["data"], columns=block["columns"])


def download_month(
    s: Any, d0: pd.Timestamp, d1: pd.Timestamp, pause: float
) -> pd.DataFrame:
    parts = []
    start = 0
    while True:
        frame = page(s, d0, d1, start)
        if frame.empty:
            break
        parts.append(frame)
        start += len(frame)
        time.sleep(pause)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["SECID"] = SECID
    for column in ("begin", "end"):
        if column in out:
            out[column] = pd.to_datetime(out[column], errors="coerce")
    for column in ("open", "close", "high", "low", "value", "volume"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["begin", "open", "close"])
    out = out[(out["open"] > 0) & (out["close"] > 0)]
    out = out.drop_duplicates(["SECID", "begin"], keep="last")
    return out.sort_values("begin").reset_index(drop=True)


def validate(path: Path, d0: pd.Timestamp, d1: pd.Timestamp) -> bool:
    try:
        frame = pd.read_parquet(path)
        begin = pd.to_datetime(frame["begin"], errors="coerce")
        return bool(
            len(frame)
            and begin.notna().all()
            and begin.dt.normalize().between(d0.normalize(), d1.normalize()).all()
        )
    except Exception:
        return False


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("data_v02"))
    p.add_argument("--from-date", type=pd.Timestamp, default=None)
    p.add_argument("--till-date", type=pd.Timestamp, default=None)
    p.add_argument("--pause", type=float, default=0.12)
    return p.parse_args()


def main() -> None:
    a = args()
    universe_path = a.root / "universe_v03" / "dynamic_universe.csv"
    universe = pd.read_csv(universe_path)
    dates = pd.to_datetime(universe["TRADEDATE"], errors="coerce")
    d0 = (a.from_date or dates.min()).normalize()
    d1 = (a.till_date or dates.max()).normalize()
    if pd.isna(d0) or pd.isna(d1) or d0 > d1:
        raise RuntimeError("Invalid download date range")

    out_dir = a.root / "market_factor_v10"
    checkpoint_dir = out_dir / "imoex_monthly"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    s = session()
    audit = []
    months = list(month_ranges(d0, d1))
    for number, (start, end) in enumerate(months, 1):
        path = checkpoint_dir / f"{start:%Y-%m}.parquet"
        if path.exists() and validate(path, start, end):
            status = "checkpoint"
            rows = len(pd.read_parquet(path, columns=["begin"]))
        else:
            frame = download_month(s, start, end, a.pause)
            if frame.empty:
                raise RuntimeError(f"No IMOEX candles for {start.date()}..{end.date()}")
            frame.to_parquet(path, index=False)
            status = "downloaded"
            rows = len(frame)
        audit.append({"month": f"{start:%Y-%m}", "status": status, "rows": rows})
        print(f"[{number}/{len(months)}] {status} {start:%Y-%m}: {rows:,}")

    parts = [pd.read_parquet(checkpoint_dir / f"{m:%Y-%m}.parquet") for m, _ in months]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(["SECID", "begin"], keep="last").sort_values("begin")
    combined.to_parquet(out_dir / "imoex_10m.parquet", index=False)
    pd.DataFrame(audit).to_csv(out_dir / "imoex_download_manifest.csv", index=False)
    print(f"saved {len(combined):,} rows -> {out_dir / 'imoex_10m.parquet'}")


if __name__ == "__main__":
    main()
