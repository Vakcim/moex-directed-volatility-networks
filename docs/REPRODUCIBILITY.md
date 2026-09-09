# Reproducibility guide

Run every command from repository root. Use a dedicated environment and keep
the private `data_v02/` directory outside version control.

## 1. Verify code

```bash
python -m pip install -r v10_directed/requirements_v10.txt
python -m pip install -r requirements-dev.txt
make check
```

## 2. Market factor and residual RV

The downloader refreshes the final requested month by default. This prevents a
checkpoint saved halfway through a month from being mistaken for a complete
month on a later run.

```bash
python -u v10_directed/download_imoex_10m_v01.py --root data_v02
MALLOC_ARENA_MAX=2 python -u \
  v10_directed/build_market_factor_residuals_v01.py --root data_v02
python -u v10_directed/build_market_factor_residuals_v01.py \
  --root data_v02 --assemble
```

Only use `--reuse-final-month` for an intentionally closed, already verified
range.

## 3. Raw predictive graphs

```bash
python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage audit-panel

for cfg in cb{01..16}; do
  MALLOC_ARENA_MAX=2 python -u \
    v10_directed/build_directed_predictive_graphs_v01.py \
    --root data_v02 --stage select --learner catboost --config-id "$cfg"
done

for cfg in en{01..15}; do
  MALLOC_ARENA_MAX=2 python -u \
    v10_directed/build_directed_predictive_graphs_v01.py \
    --root data_v02 --stage select --learner linear --config-id "$cfg"
done

python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage finalize-selection
python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage build --learner linear
python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage build --learner catboost
python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage pearson
python -u v10_directed/build_directed_predictive_graphs_v01.py \
  --root data_v02 --stage assemble
python -u v10_directed/build_directed_graph_features_v01.py \
  --root data_v02 --variant raw
```

## 4. Frozen v10 development reproduction

```bash
python -u v10_directed/build_directed_split_v01.py --root data_v02
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10 --stage select
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10 --stage run --period val
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10 --stage assemble --period val
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10 --stage run --period development
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10 --stage assemble --period development
python -u v10_directed/analyze_directed_results_v01.py \
  --root data_v02 --spec-version v10 --mode development
```

These commands reproduce development results; they do not turn them into
confirmatory evidence. The exact supplied pre-audit code is in
`legacy_snapshot/v10_reviewed_20260909` for byte-level provenance.

## 5. Prospective v10.1 pre-freeze

Do this before running any v10.1 development or confirmatory analyzer:

```bash
python -u v10_directed/preflight_v101.py \
  --root data_v02 --expect selection
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10.1 --stage select
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10.1 --stage freeze
python -u v10_directed/freeze_reproducibility_manifest_v02.py \
  --root data_v02 --spec-version v10.1 --phase preconfirmatory
python -u v10_directed/preflight_v101.py \
  --root data_v02 --expect freeze
```

Expected immutable files:

- `benchmark_v10_1_directed_confirmatory/frozen_hyperparameters.json`;
- `benchmark_v10_1_directed_confirmatory/frozen_design.json`;
- `benchmark_v10_1_directed_confirmatory/frozen_manifest_v10_1_v02.{json,csv}`.

The freeze command is create-once. `--force` is rejected for this stage.

## 6. Append sealed v10.1 predictions

Update upstream data, graphs and feature artifacts causally, then run:

```bash
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10.1 --stage run --period confirmatory
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10.1 --stage assemble --period confirmatory
```

Exact completed blocks are skipped. If the last 20-date block has gained new
keys, it is recomputed and atomically replaced. A checkpoint with unexpected
keys stops the run.

## 7. Open the holdout only at the stopping rule

```bash
python -u v10_directed/analyze_directed_results_v01.py \
  --root data_v02 --spec-version v10.1 --mode confirmatory
```

The analyzer refuses to disclose metrics before 120 valid dates unless the
latest date has reached the prespecified 2027-03-31 cap.
