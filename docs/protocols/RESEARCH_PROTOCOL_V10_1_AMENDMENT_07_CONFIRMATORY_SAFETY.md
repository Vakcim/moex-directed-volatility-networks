# Prospective v10.1 Amendment 07 — confirmatory safety and unique dynamics

Date: 2026-09-09  
Status: prospective candidate; it is not frozen until the local pre-freeze
commands complete and their manifest is retained.

## Evidence status at the amendment point

All numerical findings available on 2026-09-09 are development/exploratory and
end on 2026-08-10. The dated research record contains no executed
`confirmatory` run, no `predictions_sealed.parquet`, and no viewed target, loss,
or aggregate metric after the cutoff. The historical v10 freeze created on
2026-08-21 contained 37 records and aggregate SHA-256
`304c7036bef5009edcdfaf432e4275d072516c8c18a7e0b7e1a73ccd1fbff8af`.
It remains historical and must never be overwritten.

## A. Engineering corrections that do not use outcomes

1. Confirmatory runs load ridge and CatBoost settings only from immutable
   `frozen_hyperparameters.json`, then verify hashes of the frozen sources.
2. Freeze files are create-once; an identical invocation is harmless and a
   different payload fails closed. `--force` cannot overwrite a freeze.
3. A partial final 20-date prediction checkpoint is detected from expected
   keys, recomputed and atomically replaced as new dates arrive. Exact
   checkpoints are reused; incompatible keys fail closed.
4. Assembly requires every core model to equal the complete expected
   `TRADEDATE × SECID` scoring sample. Confirmatory artifacts are rejected if
   they contain targets or losses.
5. Per-asset residual beta/IRV checkpoints are also compared with the complete
   current date table. A strict subset is recomputed atomically, preventing a
   previously completed stock shard from silently omitting newly appended days.
6. Training blocks may not cross temporal segments, and their history is
   restricted to the current segment.
7. The final requested IMOEX month is refreshed by default, and older monthly
   checkpoints must cover the first and last requested trading dates, so a
   narrow or partial cache cannot silently remain stale.
8. Reproducibility manifest v02 searches protocols recursively and uses new
   filenames, preserving the 2026-08-21 manifest. A v10.1 confirmatory run is
   refused unless this manifest exists and all recorded scientific code,
   protocol and design hashes still match.

## B. Prospective model-specification correction

The feature builder retained two compatibility aliases:

- `incoming_edge_turnover == incoming_jaccard_distance`;
- `g_change == weighted_incoming_edge_change`.

Historical v10 placed both members of each pair in dynamic ridge models. Exact
duplicate columns change an L2 penalty by implicitly reweighting those
constructs. v10.1 therefore retains all columns in the artifact but uses only
three unique dynamic predictors in M3, M5, M7 and their placebos:

1. `incoming_jaccard_distance`;
2. `delta_incoming_strength`;
3. `weighted_incoming_edge_change`.

No window, top-k, graph learner, target, loss, sample cutoff, or hypothesis is
changed. Ridge alpha must be reselected using only the original 60 validation
dates. The resulting model schema, selection table and source hashes must be
frozen before any v10.1 development or confirmatory loss is inspected.

## C. Inference reproducibility

Historical v10 used bootstrap seeds `260821 + comparison_index`; those outputs
remain unchanged. v10.1 uses the prespecified seed `260821` for every paired
comparison, recorded in each result row.

## Interpretation rule

v10.1 is a new prospective specification, not a retroactive correction that
upgrades development evidence. Historical v10 results must remain labeled
development-only. Confirmatory evidence is reported only after 120 valid dates
or the prespecified 2027-03-31 calendar cap.
