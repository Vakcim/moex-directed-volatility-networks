# Code audit — 2026-09-09

## Scope

Reviewed the supplied v10 code package and protocol. The archive did not
contain `data_v02`, prediction tables or final analysis tables, so numerical
results could be documented from the dated research record but not recomputed.

## High-priority findings resolved in the active code

| Finding | Risk | Resolution |
|---|---|---|
| Confirmatory runner read mutable development selections | Silent design drift | Load only frozen JSON and verify source hashes |
| Existing block file caused unconditional skip | New dates inside the current 20-day block could be omitted | Compare expected keys; recompute a strict subset atomically |
| Assembly compared models only to each other | All models could share the same incomplete sample | Require exact equality with the full expected scoring sample |
| Monthly IMOEX checkpoint could cover only an earlier narrow range | Stale market factor and residual features | Refresh the final month and validate endpoint coverage for older months |
| Per-asset residual shards skipped whenever files existed | New dates could be absent from IRV and beta panels | Compare each shard with the full current date table and recompute strict subsets atomically |
| Manifest v01 searched only the project root | Nested main protocol could be omitted | Recursive manifest v02 plus explicit protocol presence check |
| Frozen parameters did not freeze executable code | Post-freeze implementation drift | Confirmatory runner requires manifest v02 and verifies scientific code/protocol/design hashes |
| Dynamic design contained exact alias columns | Implicit double L2 weighting | Separate prospective v10.1 with unique constructs |
| Generic ridge history did not explicitly restrict segment | Possible cross-break history | Assert one segment per block and filter history to it |
| Protocol seed and comparison-index seed differed | Reproduction ambiguity | Preserve historical v10; record fixed seed 260821 in v10.1 rows |

## Preserved historical facts

- Main protocol frozen 2026-08-18.
- Development cutoff 2026-08-10.
- Historical freeze executed 2026-08-21.
- Historical manifest: 37 records; aggregate SHA-256
  `304c7036bef5009edcdfaf432e4275d072516c8c18a7e0b7e1a73ccd1fbff8af`.
- Stress-MAD correction was discovered after development H1/H2 inspection and
  remains disclosed as post-outcome.
- Residual ElasticNet was excluded by pre-forecast Amendment 06 feasibility.
- No confirmatory outcome had been run or inspected by 2026-09-09.

## Remaining verification requiring the user's local data

1. Run the test suite in the target environment.
2. Reselect v10.1 ridge alphas on exactly 60 validation dates.
3. Freeze v10.1 and create manifest v02 before inspecting v10.1 outcomes.
4. Recompute the public small result tables directly from retained result
   artifacts; replace `recovered_from_dated_research_dialogue_not_recomputed`
   labels only after hashes and values agree.
5. Keep raw licensed market data and sealed prediction shards private.
