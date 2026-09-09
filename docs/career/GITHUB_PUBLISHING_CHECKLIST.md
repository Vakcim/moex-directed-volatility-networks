# GitHub publishing checklist

## Before the first push

- [ ] Run `git status --short` and inspect every file.
- [ ] Confirm `data_v02/`, parquet files, logs, videos and prediction parts are
  ignored.
- [ ] Search for secrets and absolute local paths:

  ```bash
  rg -n "(/home/|API_KEY|TOKEN|PASSWORD|SECRET)" \
    -g '!legacy_snapshot/**' -g '!docs/protocols/**'
  ```

- [ ] Run `make check`.
- [ ] Add `repository-code` to `CITATION.cff` after the GitHub URL exists.
- [ ] Choose and add a license; do not publish a placeholder license.
- [ ] Replace dialogue-recovered metrics only if local machine-generated values
  and hashes agree.

## Suggested history

Preserve authorship by making the commits under the user's own Git identity:

```bash
git init -b main
git add .
git commit -m "chore: import audited v10 research package"
git tag -a v10-development-snapshot -m "Development-only v10 snapshot"
```

After the local v10.1 freeze files have been copied into a private provenance
record, create a second commit and annotated tag:

```bash
git add docs v10_directed
git commit -m "research: freeze prospective v10.1 specification"
git tag -a v10.1-prospective-freeze -m "Prospective design before outcome inspection"
```

Do not commit a fake freeze from this review package: it lacks private data and
cannot select or hash the local artifacts.

## Repository presentation

- Short description: `Directed predictive networks and realized-volatility
  forecasting on MOEX, with leakage-aware validation.`
- Suggested topics: `financial-econometrics`, `volatility-forecasting`,
  `network-science`, `time-series`, `catboost`, `reproducible-research`.
- Pin the repository after README, results status and tests render correctly.
- Add one release containing code and small documentation only; exclude market
  data and sealed predictions.

## Pull-request discipline

Each PR or self-review should answer:

1. Did this change use validation, development or confirmatory information?
2. Does it change the scientific specification or only implementation?
3. Which tests and hashes establish the result?
4. Does an amendment need to be dated before execution?
