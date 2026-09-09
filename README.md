# Predictive Volatility Networks on MOEX

Leakage-aware construction of directed nonlinear predictive networks and their
incremental value for one-day-ahead realized-variance forecasting.

## Research question

Does the history of stock `j` add nonlinear out-of-sample information about
the next-day realized variance of stock `i`, after conditioning on the target's
own HAR history and a market-volatility factor?

The project distinguishes three objects that are often conflated:

\[
\text{co-movement} \ne \text{conditional predictability} \ne \text{causality}.
\]

Pearson graphs represent the first. Out-of-sample source-group permutation
importance from CatBoost approximates the second. The project makes no causal
claim.

## Current result

The development sample ends on **2026-08-10**. Directed CatBoost networks are
highly asymmetric and have little edge overlap with Pearson networks, but their
incremental forecasting signal is weak. CatBoost-current was numerically only
`0.000888` QLIKE below HAR-RV (`-7.145484` versus `-7.144596`) and the gain was
not statistically stable. Dynamic graph changes did not improve forecasts
consistently across Pearson, ElasticNet and CatBoost.

All reported values are **development/exploratory**, not confirmatory. No
post-cutoff outcome had been run or viewed as of 2026-09-09. See
[Development results](docs/results/DEVELOPMENT_RESULTS.md).

## What is methodologically useful here

- point-in-time dynamic top-40 universe drawn from an 82-stock history panel;
- train / importance / forecast graph windows separated in chronological time;
- directed group permutation importance evaluated out of sample;
- raw and causally market-residualized realized variance;
- HAR-RV, HAR-market, Pearson, ElasticNet and CatBoost comparisons;
- QLIKE, daily paired HAC inference, circular block bootstrap and Holm control;
- graph identity, lag-20 and frozen-graph placebos;
- explicit negative-result reporting and a sealed future holdout;
- immutable specifications, source hashes and resumable atomic checkpoints.

## Repository map

| Path | Purpose |
|---|---|
| `v10_directed/` | Executable research pipeline |
| `docs/protocols/` | Frozen protocol, amendments and implementation notes |
| `docs/results/` | Human-readable development evidence |
| `results/development/` | Small reported result tables; no raw market data |
| `results/confirmatory/` | Sealed-holdout status only |
| `docs/career/` | Daily research log and portfolio workflow |
| `tests/` | Unit tests for leakage, freeze and checkpoint safeguards |
| `legacy_snapshot/` | Exact supplied v10 code snapshot for provenance |

## Two specifications

`v10` reproduces the historical development design. Its dynamic ridge models
include two pairs of compatibility aliases retained in the feature artifact.

`v10.1` is prospective Amendment 07. It uses the same hypotheses and data
clock, but removes exact duplicate aliases from the model matrix, enforces
frozen confirmatory parameters, refreshes the final IMOEX month, validates
complete checkpoints and blocks cross-segment training. It writes only to
`benchmark_v10_1_*` and cannot overwrite historical v10.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r v10_directed/requirements_v10.txt
python -m pip install -r requirements-dev.txt
make check
```

The private/raw `data_v02` directory is intentionally not distributed. Its
required schema and artifact layout are documented in [data/README.md](data/README.md).

## Reproduce historical v10

The full command order is in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
The exact supplied pre-audit implementation is preserved under `legacy_snapshot`; the
active runner defaults to `--spec-version v10` and adds fail-closed engineering
guards without relabeling any historical result.

## Prepare prospective v10.1 without looking at outcomes

On the machine containing `data_v02`:

```bash
python -u v10_directed/preflight_v101.py \
  --root data_v02 --expect selection

python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10.1 --stage select

python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10.1 --stage freeze

python -u v10_directed/freeze_reproducibility_manifest_v02.py \
  --root data_v02 --spec-version v10.1 --phase preconfirmatory

python -u v10_directed/preflight_v101.py \
  --root data_v02 --expect freeze
```

Do not run the v10.1 development analyzer between selection and freeze. The
first command reads only the original 60 validation dates. The freeze is not
complete until the generated JSON files and manifest are retained outside the
working directory or committed to a private provenance record.

## Continue the sealed holdout

After new market data and causal features are updated:

```bash
python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10.1 --stage run --period confirmatory

python -u v10_directed/train_directed_har_benchmark_v01.py \
  --root data_v02 --spec-version v10.1 --stage assemble --period confirmatory
```

The assembled file contains predictions but no targets or losses. Analysis
stays locked until 120 valid dates or the 2027-03-31 cap.

## Career-facing deliverables

The repository already supports four honest portfolio artifacts:

1. reproducible research code with temporal leakage guards;
2. a concise negative-result case study;
3. a conference abstract and paper outline;
4. an auditable daily engineering/research log.

Use [the daily workflow](docs/career/DAILY_RESEARCH_WORKFLOW.md) and
[GitHub publishing checklist](docs/career/GITHUB_PUBLISHING_CHECKLIST.md) to
advance the work in small, visible increments without paying for a conference
or service.

## Scope and limitations

This is research software, not investment advice. Raw MOEX data are not
redistributed. The evidence currently covers one market and one development
period; graph importance is predictive, not causal; and the future holdout is
still sealed.
