# Research log

Use one entry per session. Keep failed attempts; they are part of the audit
trail.

## Entry template

### YYYY-MM-DD — short outcome

- **Question/task:**
- **Input status:** development / validation-only / sealed technical-only.
- **Command or file changed:**
- **Result:**
- **Verification:**
- **Outcome exposure:** none / development / confirmatory (the last is forbidden
  before the stopping rule).
- **Decision:**
- **Career evidence:**
- **Next smallest task:**

## 2026-09-09 — GitHub and confirmatory-safety audit

- **Question/task:** Turn the reviewed v10 package into a clean, defensible
  GitHub project and repair outcome-independent execution risks.
- **Input status:** code/protocol archive plus dated development-result record;
  no `data_v02` and no post-2026-08-10 outcomes.
- **Result:** separated historical v10 and prospective v10.1; restored missing
  amendment records; added frozen-parameter enforcement, atomic incremental
  checkpoints, exact assembly completeness, final-month refresh, recursive
  manifests, tests, CI and portfolio documentation.
- **Verification:** `make check` on Python 3.12.14; 21 unit tests passed. Legacy
  snapshot verification: `sha256sum -c SHA256SUMS`; all 18 supplied files
  matched.
- **Outcome exposure:** development results already known; no confirmatory
  outcome exposed.
- **Decision:** v10 results remain development-only. v10.1 is a candidate until
  local validation-only selection and create-once freeze complete.
- **Career evidence:** scientific versioning, reproducibility review, temporal
  leakage controls and clear negative-result communication.
- **Next smallest task:** run `make check` and record the environment versions.
