# V10 Amendment 02 — directed graph coverage gate

Date: 2026-08-19  
Record status: reconstructed on 2026-09-09 from the dated research dialogue;
the original file was absent from the review archive.

## Status before amendment

The all-82 history correction in Amendment 01 had been completed. Only graph
coverage and selection diagnostics were available; amended forecast losses had
not been computed.

## Trigger

With all-82 history, blocks contained 32–37 feasible targets and a median of
35. The earlier requirement of at least 35 covered targets on every valid date
left a valid-date rate of 0.6146, despite otherwise adequate median coverage.

## Amendment

- Every graph block still starts from exactly 40 point-in-time active targets.
- A forecast date is graph-valid when at least 32 of them are feasible.
- The graph-level gate still requires `valid_date_rate >= 0.90`.
- Median feasible targets must remain at least 35.
- Per-target history minima remain 400 training rows and 80 importance rows.
- All artifacts built under the pre-Amendment-01/02 coverage definition are
  invalid and must not be mixed with amended artifacts.

The change is a coverage estimability rule fixed before any amended forecast
loss was available.

