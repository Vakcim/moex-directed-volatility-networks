# Daily research workflow

This research block fits the previously chosen weekday career routine:

- one algorithms problem;
- one theory topic;
- one small research/GitHub contribution;
- English and German practice.

Weekends remain reserved for rest and the trousers/sewing project. Research
tasks should not silently consume them. Use one small, auditable research unit
per weekday. A normal research session is 30–60 minutes; when time is short,
complete only the log and one verification step.

## Daily loop

1. **Choose one output (5 minutes).** Open one GitHub issue or copy the daily
   log template. The task must end in a file, test, figure, table or written
   decision.
2. **Work on one layer (20–40 minutes).** Do not mix data repair, model changes
   and paper writing in the same commit.
3. **Verify (5–10 minutes).** Run the narrowest relevant test, record the exact
   command and whether it passed.
4. **Write the evidence note (5 minutes).** State what changed, what was
   observed, whether outcomes were exposed, and the next smallest task.
5. **Commit.** Use a descriptive message such as `test: guard partial forecast
   checkpoints` or `docs: report development-only network overlap`.

## Weekday rotation for the research block

| Day type | Output |
|---|---|
| Reproducibility | One passing test, data audit or hash check |
| Scientific analysis | One prespecified table or robustness result |
| Visualization | One publication-quality figure with source script |
| Writing | One methods/results subsection or literature note |
| Career packaging | README improvement, issue, release note or short explanation |
| Review | Weekly summary: evidence gained, risks, next three issues |

The rotation is flexible; the rule is one verifiable output rather than a fixed
number of hours. Algorithms, theory and language work are tracked separately,
so a large research run does not count as completing the whole daily plan.

## First ten sessions

1. Copy this repository beside the private `data_v02`; run `make check`.
2. Audit and import upstream builders into `upstream/` without data or logs.
3. Run v10.1 alpha selection on exactly 60 validation dates; inspect coverage,
   not development performance.
4. Freeze v10.1 and save the manifest hash.
5. Generate machine-derived development metric CSVs and compare them with the
   dialogue-recovered values.
6. Create the M0/M2/M3/M6 forecast comparison figure.
7. Create the Pearson-versus-CatBoost topology figure.
8. Create the raw-versus-residual overlap figure.
9. Draft the methods section from the frozen protocol.
10. Rehearse a three-minute explanation of why a negative forecasting result is
    scientifically valuable.

## Rules that protect the research claim

- Never choose a v10.1 feature or parameter from development or confirmatory
  performance before the v10.1 freeze.
- Never inspect confirmatory targets or losses before the stopping rule.
- Label post-outcome diagnostics as exploratory.
- Do not call permutation importance causality.
- Never commit raw data, credentials, large model artifacts or local absolute
  paths.
- If a code defect is found, record discovery time, exposed outcomes, exact
  scope and whether a new specification version is required.

## Career translation

For every technical task, add one sentence answering: “What professional skill
does this demonstrate?” Examples: temporal validation, leakage prevention,
reproducible ML, statistical inference, research communication, or resilient
batch pipelines. These sentences become material for CV bullets and interviews.
