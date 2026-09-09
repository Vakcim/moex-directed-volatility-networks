# V10 Amendment 01 — all-82 causal target history

Date: 2026-08-19  
Record status: reconstructed on 2026-09-09 from the dated research dialogue;
the original file was absent from the review archive.

## Status before amendment

Only graph feasibility and graph-learner development selection diagnostics had
been inspected. No HAR forecast losses, placebo outcomes, confirmatory
aggregates, or post-2026-08-10 outcomes had been computed or viewed.

## Trigger

The first feasibility pass used `forecast_samples.parquet` as model history.
That panel contains only top-40 forecast rows and is too short for many graph
targets. CatBoost produced 542 valid block-target fits out of 960, ElasticNet
540 out of 960, and 418 out of 960 target states failed. The median number of
valid targets was 22.5; failed targets had a median of only 233 history rows.

## Amendment

- Forecast targets remain the point-in-time dynamic top-40 universe.
- Causal graph train and importance history is rebuilt from the all-82 RV
  panel.
- The original 400-row training and 80-row importance minima remain fixed.
- All pre-amendment graph-selection shards, selected configurations and raw
  graph artifacts are invalid and must be rebuilt.
- No forecast target after the development cutoff may be inspected to make
  this change.

This corrects the history source; it does not change the forecast estimand,
universe, loss, graph clock, top-k, or confirmatory boundary.

