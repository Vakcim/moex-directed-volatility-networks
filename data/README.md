# Data contract

Raw data and large derived artifacts are not committed. The pipeline expects a
local `data_v02/` directory at repository root (or another directory passed via
`--root`).

## Required upstream artifacts

| Relative path | Minimum role |
|---|---|
| `model_data/node_features_all82.parquet` | Point-in-time node/HAR panel, activity and `segment_id` |
| `model_data/rv_all82.parquet` | All-82 realized variance and validity flags |
| `model_data/forecast_samples.parquet` | Top-40 forecast origins and next-day target |
| `universe_v03/dynamic_universe.csv` | Point-in-time active universe |
| `universe_v03/intraday_download_secids.csv` | Permitted security identifiers |
| `graphs/raw_pearson_edges.parquet` | Historical Pearson graph input |
| `model_audit/graph_quality_pearson.csv` | Pearson graph quality audit |
| one frozen legacy `split_dates.csv` | Chronological source split used by the v10 split builder |

The upstream builders shown in the user's source inventory
(`download_intraday_10m_v01.py`, `build_realized_variance_v02.py`,
`build_dynamic_universe_v03.py`, `build_daily_features_v01.py`, and related
audits) were not included in the reviewed archive. Add them later under a
separate `upstream/` directory only after checking their licenses, secrets and
paths. Do not fabricate or silently replace them.

## Core key rules

- `TRADEDATE` must parse as a normalized date.
- `SECID` is upper-case and unique with `TRADEDATE` where applicable.
- Each date maps to exactly one `segment_id`.
- Rolling histories, graph transitions and permutations cannot cross segments.
- `RV` targets must be positive and finite when marked valid.
- No API credentials, personal paths, raw parquet data or large prediction
  shards belong in Git.

## Derived v10 artifacts

The code builds these major directories below the data root:

- `market_factor_v10/`
- `graphs_v10/`
- `model_audit/`
- `model_data/`
- `benchmark_v10_directed_development/`
- `benchmark_v10_directed_confirmatory/`
- `benchmark_v10_1_directed_development/`
- `benchmark_v10_1_directed_confirmatory/`

Small schemas, audits and frozen hashes may be published after reviewing them
for licensed or identifying content. Parquet inputs and sealed prediction
parts remain private until the protocol permits release.

