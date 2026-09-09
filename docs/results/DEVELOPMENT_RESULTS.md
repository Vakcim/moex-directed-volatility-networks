# Development results through 2026-08-10

Status: exploratory/development only. These values were recovered from the
dated research dialogue and were not recomputed from the review archive,
because the archive contains code and protocol documents but not the underlying
`data_v02` tables or result artifacts.

## Main forecasting result

Lower QLIKE is better.

| Branch | Model | Mean QLIKE | Interpretation |
|---|---|---:|---|
| Raw RV | M0 HAR-RV | -7.144596 | Strong baseline |
| Raw RV | M2 Pearson current | -7.1354 | Worse than HAR-RV |
| Raw RV | M3 Pearson dynamic | -7.1343 | Worse than Pearson current and HAR-RV |
| Raw RV | M6 CatBoost directed current | -7.145484 | Numerically 0.000888 below HAR-RV; improvement was not statistically stable |
| Residual RV | M0 HAR-RV | -7.841625 | Best among the reported residual rows |
| Residual RV | M1 HAR-market | -7.830605 | Worse than residual M0 |
| Residual RV | M6 CatBoost directed current | -7.838069 | Recovers part of M1's loss, but does not beat residual M0 |
| Residual RV | M7 CatBoost directed dynamic | approximately -7.837915 | No useful dynamic increment |

Ridge validation selected `alpha = 100` throughout the reported downstream
HAR models, indicating that the extra graph covariates required strong
shrinkage.

## Network structure

| Quantity | Development estimate |
|---|---:|
| CatBoost–Pearson edge Jaccard | approximately 0.056 |
| CatBoost reciprocity | approximately 0.072 |
| Pearson reciprocity | 1.000 by construction |
| Temporal Jaccard, Pearson | approximately 0.403 |
| Temporal Jaccard, raw CatBoost | approximately 0.299 |
| Temporal Jaccard, residual CatBoost | approximately 0.318 |
| Raw–residual CatBoost edge Jaccard | approximately 0.193 |
| Raw–residual weighted Jaccard | approximately 0.151 |
| Raw–residual overlap enrichment | approximately 3.75 |

## Defensible scientific conclusion

The directed nonlinear predictive network is structurally different from a
correlation network: it is asymmetric, concentrated and changes after the
market factor is removed. That structural novelty does not translate into a
robust one-day realized-variance forecasting gain over HAR-RV. Current graph
state contains, at most, a weak incremental signal; changes in graph state do
not improve forecasts consistently across Pearson, ElasticNet and CatBoost.

This is a substantive negative result, not proof that networks are absent. A
compact representation is

\[
y_{i,t+1}=f(H_{i,t})+\delta g(G_t)+\varepsilon_{i,t+1},
\qquad
\operatorname{Var}(\delta g(G_t))\ll\operatorname{Var}(\varepsilon_{i,t+1}).
\]

## Claims that are not supported

- CatBoost has not been shown to robustly beat HAR-RV.
- Predictive importance is not causal influence.
- The absence of a dynamic increment has not yet been confirmed on the sealed
  future holdout.
- The reported numbers must not be relabeled as confirmatory evidence.

