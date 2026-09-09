# V10 Amendment 06 — residual graph-learner feasibility

Date: 2026-08-24  
Record status: reconstructed on 2026-09-09 from the dated research dialogue;
the original file was absent from the review archive.

## Status before amendment

Residual graph construction and its coverage-only audit had completed. No
residual graph features, downstream residual forecasts, forecast losses, or
residual hypothesis tests had been computed.

## Evidence

- Residual CatBoost: 826 valid graph states; median 35 feasible targets.
- Residual ElasticNet: 818 valid states; median 34.5 feasible targets.
- Eight ElasticNet states failed with `no_positive_weight`.

## Decision

- Residual CatBoost passes the frozen feasibility gates and remains eligible.
- Residual ElasticNet is excluded before downstream feature construction and
  forecasting.
- Residual Pearson is undefined for this branch and is not substituted.
- Coverage thresholds and all raw-branch results remain unchanged.
- The residual analysis is development robustness evidence, not confirmatory
  evidence.

