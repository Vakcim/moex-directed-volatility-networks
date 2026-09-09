# V10 Implementation Note 04: robust linear-strength sensitivity

## Motivation

Post-development diagnostics found one isolated linear graph state
(`RNFT`, graph block 73) whose raw incoming permutation-QLIKE strength was
approximately `1.93e8`. The remaining linear-state distribution had median
`0.0204`, 99th percentile `0.197`, and only one other state above one.
CatBoost raw strength had maximum `0.639`.

The raw importance definition is nonnegative mean QLIKE degradation after
source permutation. Because QLIKE contains `y / prediction`, its magnitude is
unbounded when a permuted model predicts near zero. Raw strength is therefore
not a robust linear covariate even though normalized graph weights remain
finite and sum to one.

## Sensitivity specification

This explicitly post-development branch:

1. replaces raw strength `s_t` by `log(1 + s_t)`;
2. replaces raw change `s_t - s_{t-1}` by
   `log(1 + s_t) - log(1 + s_{t-1})`;
3. applies a train-only extrapolation guard by clipping standardized model
   inputs to `[-10, 10]` at both training calibration and prediction;
4. reselects ridge alpha using only the original 60-date validation period;
5. evaluates on the unchanged 181-date development sample;
6. writes only to `benchmark_v10_directed_robustness`.

It does not overwrite or replace primary H1c. Any development result from this
branch is exploratory. The specification may be frozen prospectively for the
future confirmatory holdout before any holdout losses are examined.

## Run

```bash
MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -u V10_DIRECTED_CODE_PACKAGE/v10_directed/train_robust_linear_sensitivity_v01.py \
  --root data_v02 --stage all 2>&1 | tee robust_linear_sensitivity.log
```
