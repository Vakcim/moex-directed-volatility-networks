# V10 implementation note 02 — NaN-safe rolling market-stress MAD

## Discovery point

This implementation defect was identified after the raw development analysis
had been run and H1/H2 development outcomes had been inspected.  The initial
H2 regression contained only 63 of 181 development dates.  The correction and
its timing must therefore be disclosed with every v10 result.

## Defect

The prespecified stress statistic uses a rolling 252-date median and median
absolute deviation (MAD), with at least 126 observed market dates.  Pandas'
rolling median correctly ignores occasional missing market observations, but
the original rolling MAD callback used `numpy.median`.  A single NaN in an
otherwise eligible window consequently made the MAD and `stress_z` missing for
the entire remaining 252-date window.

This was inconsistent with the already frozen `min_periods=126` rule: a window
with at least 126 finite observations was intended to remain estimable.

## Correction

The MAD callback now uses `numpy.nanmedian` for both the center and the absolute
deviations.  No window length, minimum-history requirement, clipping rule,
market factor, temporal segment, hypothesis, or inference parameter changes.

The development analyzer now reloads `stress_z` from the deterministic directed
feature artifact instead of trusting the stale convenience copy stored in old
prediction checkpoints.  It also writes `stress_coverage.csv`.

## Scope

`stress_z` is used only by the H2 regime regression.  It is not a predictor in
M0--M8 or any placebo model.  Therefore:

- existing prediction values remain frozen and are not refit;
- H1a--H1c, placebo contrasts, M8 metrics, and their loss differentials do not
  change;
- only H2 is recomputed from the corrected prespecified stress statistic.

The three RNFT predictions at the `1e-12` floor in M4/M5 are not modified by
this correction.  They remain part of the primary QLIKE evaluation as observed
linear-model failures.
