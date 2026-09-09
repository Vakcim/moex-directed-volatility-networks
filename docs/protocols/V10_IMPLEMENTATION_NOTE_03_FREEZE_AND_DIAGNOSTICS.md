# V10 Implementation Note 03: freeze manifest and post-development diagnostics

## Status

This supplement does not alter the frozen H1/H2 hypotheses, models, predictions,
losses, graph construction, or confirmatory holdout. It adds reproducibility
metadata and explicitly exploratory diagnostics discovered or requested after
inspection of development outcomes.

## A. Complete development freeze

The existing training script writes the frozen hyperparameters and confirmatory
design. `freeze_v10_manifest_v01.py` additionally hashes the active V10 code,
protocol and implementation notes, graph/split designs, directed features,
development predictions, and result tables. Existing manifests are not
overwritten unless `--force` is explicitly supplied.

## B. Exploratory M8 comparisons

`analyze_directed_diagnostics_v01.py` performs paired date-level QLIKE tests for:

1. HAR-market versus direct CatBoost (M1 vs M8);
2. HAR-RV versus direct CatBoost (M0 vs M8);
3. dynamic directed CatBoost features versus direct CatBoost (M7 vs M8).

Positive loss differences mean that M8 has lower QLIKE. HAC and circular
20-date block-bootstrap inference are reported, with Holm adjustment across
these three exploratory comparisons. They are not promoted into the frozen
directed-core family.

## C. RNFT/floor diagnostic

For exact `1e-12` prediction-floor events in M4/M5, the diagnostic reconstructs
the original walk-forward ridge fit, verifies the stored prediction, and writes
the fixed effect and every standardized feature contribution to the unclipped
log prediction. A secondary H1c result excluding affected dates is written only
as an ex-post stability diagnostic. The primary QLIKE result retains all rows.

## Run order

```bash
python -u V10_DIRECTED_CODE_PACKAGE/v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --stage freeze

python -u V10_DIRECTED_CODE_PACKAGE/v10_directed/freeze_v10_manifest_v01.py \
  --root data_v02

MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u V10_DIRECTED_CODE_PACKAGE/v10_directed/analyze_directed_diagnostics_v01.py \
  --root data_v02 2>&1 | tee analyze_directed_diagnostics.log
```

The diagnostic script must not change the hashes of `model_metrics.csv`,
`paired_inference_core.csv`, `paired_inference_regime.csv`, or
`paired_inference_placebo.csv`.
