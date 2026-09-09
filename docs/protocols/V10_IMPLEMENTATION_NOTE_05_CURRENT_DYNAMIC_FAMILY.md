# V10 Implementation Note 05: cross-method current/dynamic family

The completed robustness run suggested a common negative dynamic increment
across Pearson, robust-linear, and CatBoost graph representations. This
post-development analysis evaluates the three paired comparisons on identical
dates and applies Holm adjustment within the family.

It also compares the CatBoost-current graph representation against HAR-RV,
Pearson-current, and robust-linear-current. These comparisons are exploratory
and do not replace the frozen core hypotheses.

Positive loss difference always means that the right-hand model has lower
QLIKE. Thus positive values in the current-graph family favor CatBoost-current;
positive values in the dynamic-increment family favor the dynamic model.

```bash
python -u V10_DIRECTED_CODE_PACKAGE/v10_directed/analyze_current_dynamic_family_v01.py \
  --root data_v02 2>&1 | tee analyze_current_dynamic_family.log
```
