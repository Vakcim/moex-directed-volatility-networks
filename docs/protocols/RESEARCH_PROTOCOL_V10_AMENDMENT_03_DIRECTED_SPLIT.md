# V10 Amendment 03 — directed-sample warm-up and development split

Date: 2026-08-20

## Status before amendment

Market-factor construction, directed graph hyperparameter selection, raw graph
construction and graph-feature assembly had completed. Only graph, chronology,
coverage and finite-feature audits had been inspected. No M0--M8 forecast
predictions, model-comparison losses, placebo contrasts, or directed-network
outcome metrics had been computed.

## Trigger and evidence

The frozen v09 split was defined before the v10 directed graph clock. The v10
graph requires 504 graph-training dates, 126 importance dates and a preceding
graph state for dynamic features. Consequently, the assembled common sample
contained:

- 0 directed-valid dates in the legacy train period;
- 17 directed-valid dates in the legacy validation period;
- 241 directed-valid dates in the legacy test period;
- 258 directed-valid dates and 8,258 rows in total.

Using the legacy split would ask the first ridge-validation fit to train on an
empty common sample. This is a design incompatibility, not a forecasting
outcome.

## Amendment

A v10-specific split is constructed using only `TRADEDATE`, the legacy split
label and the precomputed Boolean `directed_sample_valid`. The builder is
forbidden to read target values, predictions or losses.

For the currently observed development data:

1. all directed-valid origins before the first directed-valid legacy-test date
   form the warm-up/training history (17 observed dates; minimum gates 15 dates,
   30 SECIDs and 500 rows);
2. the first 60 directed-valid origins from the legacy test period form the
   ridge-alpha validation set (three 20-date forecasting refits);
3. the remaining directed-valid origins through 2026-08-10 form the exploratory
   development test set (181 observed dates; minimum gate 120);
4. calendar dates between these valid origins inherit the corresponding
   chronological role, so the split remains ordered and non-overlapping;
5. dates after 2026-08-10 remain confirmatory and are not used here.

The exact boundaries, source hashes and output split hash are written before
any HAR forecast losses are computed.

## Unchanged parameters and safeguards

- The graph-learner selections (`cb02` and `en11`) remain frozen. They were
  selected only with the prior development-validation information and are not
  reselected with this split.
- The common M0--M7 sample, 30-label date floor, graph clocks, top-k, feature
  definitions and QLIKE objective do not change.
- Ridge alpha remains selected separately for each M0--M7 model from the frozen
  grid `{0, 0.01, 0.1, 1, 10, 100}`.
- The 60 alpha-validation dates are excluded from exploratory development
  inference and are not reported as test performance.
- Confirmatory start, 120-valid-date holdout requirement and sealed-loss rules
  remain unchanged.

## Interpretation

All results through 2026-08-10 remain development/exploratory. This amendment
does not recover or relabel them as confirmatory evidence; it only makes the
pre-confirmatory directed benchmark estimable without leakage or an empty
training sample.
