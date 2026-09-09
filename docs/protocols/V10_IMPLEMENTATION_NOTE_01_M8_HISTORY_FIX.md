# V10 implementation note 01 — M8 causal all-82 history

Date: 2026-08-20

## Trigger

After M0--M7 and placebo prediction coverage had passed, but before M8 was
run, code inspection showed that `M8_direct_catboost` inherited the
`directed_sample_valid` training filter. The complete directed common sample
contains only 258 dates, while M8 retained a minimum of 400 target-history
rows. The implementation would therefore reject every target in every block.

No M8 predictions or M8 losses had been computed when the issue was found.

## Correction

The forecasting sample for M8 remains the exact directed development sample.
Only its training-history loader changes:

- target labels and OwnHAR histories come from the causal all-82 panel;
- MarketHAR comes from the frozen IMOEX market-factor branch;
- source groups are the same all-82 HAR source columns used elsewhere in v10;
- each target uses only observations strictly before the forecast block;
- history cannot cross the current temporal segment;
- the frozen minimum of 400 valid target-history rows is unchanged;
- CatBoost missing-source handling is retained; infinities are converted to
  missing values before fitting or prediction;
- the target's duplicated source group is removed from its own M8 inputs;
- configuration `cb02`, its seed, calibration and all forecast clocks remain
  unchanged.

M0--M7, placebo predictions, the directed split and all inspected forecasting
outcomes are unaffected. This is an implementation correction that restores
the M8 definition already stated in section 11.5 of the frozen protocol; it
does not introduce a new scientific parameter.
