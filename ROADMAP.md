# Roadmap

The goal is to turn one serious research project into visible evidence for
Quant Research, ML/Data Science internships and future graduate applications,
without requiring paid services. It occupies the research/GitHub block of the
weekday plan; algorithms, one theory topic and English/German continue in
parallel, while weekends remain for rest and sewing.

## Milestone 1 — trustworthy repository

- Run `make check` locally.
- Add the final repository URL to `CITATION.cff`.
- Choose a license deliberately.
- Create a public GitHub repository without `data_v02` or sealed predictions.
- Tag the imported historical snapshot and the prospective v10.1 code
  separately.

Definition of done: a stranger can understand the question, evidence status,
limitations and commands in ten minutes.

## Milestone 2 — prospective v10.1 freeze

- Select ridge alpha using only the original 60 validation dates.
- Freeze v10.1 parameters and feature schema.
- Generate manifest v02 and retain its SHA-256.
- Record the commands and hashes in `docs/career/RESEARCH_LOG.md`.

Definition of done: later choices can be audited against a dated, immutable
design without opening the holdout.

## Milestone 3 — publication-quality development package

- Export machine-generated metrics with hashes.
- Produce a forecast comparison table, network-overlap figure, temporal
  stability figure and raw-versus-residual figure.
- Add a concise methods diagram and data-flow diagram.
- Complete related work and limitations.

Definition of done: a 10-minute talk and a self-contained preprint draft can be
created from repository artifacts.

## Milestone 4 — visible career artifacts

- Pin the repository on GitHub.
- Add a one-page case study to a portfolio or LinkedIn Featured section.
- Record a short technical walkthrough using the conference abstract.
- Share a free preprint/code release only after checking MOEX data redistribution
  restrictions and coauthor/supervisor expectations.

Definition of done: applications can link to code, research writing and a
clear explanation of a negative result.

## Milestone 5 — sealed confirmatory evaluation

- Append predictions as new valid dates arrive.
- Run only technical completeness checks before the stopping rule.
- Open outcomes once, execute the frozen analyzer and report every prespecified
  comparison regardless of sign.

Definition of done: a clean confirmatory table, or an explicitly underpowered
table at the calendar cap, is added without retrospective tuning.
