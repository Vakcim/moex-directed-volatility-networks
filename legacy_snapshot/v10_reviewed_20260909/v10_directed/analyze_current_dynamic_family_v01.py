"""Exploratory cross-method current-vs-dynamic and current-graph inference.

Uses frozen primary predictions plus the separately generated robust-linear
predictions.  It does not alter the frozen core family.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_directed_results_v01 import add_losses, inference_table
from v10_common import holm, normalize_keys, read_table


CURRENT_GRAPH = [
    ("C1_catboost_current_vs_har", "M0_har_rv", "M6_catboost_directed_current"),
    ("C2_catboost_current_vs_pearson", "M2_pearson_current", "M6_catboost_directed_current"),
    ("C3_catboost_current_vs_robust_linear", "R4_linear_current_logclip", "M6_catboost_directed_current"),
]

DYNAMIC_INCREMENT = [
    ("D1_pearson_dynamic_increment", "M2_pearson_current", "M3_pearson_dynamic"),
    ("D2_linear_dynamic_increment", "R4_linear_current_logclip", "R5_linear_dynamic_logclip"),
    ("D3_catboost_dynamic_increment", "M6_catboost_directed_current", "M7_catboost_directed_dynamic"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_v02"))
    return parser.parse_args()


def adjusted(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for column in (
        "hac_p_two_sided",
        "hac_p_one_sided",
        "bootstrap_p_two_sided",
        "bootstrap_p_one_sided",
    ):
        table[f"holm_{column}"] = holm(table[column])
    table["analysis_status"] = "exploratory_post_development"
    return table


def run_family(predictions: pd.DataFrame, comparisons, family: str, output: Path) -> None:
    result, daily, audit = inference_table(predictions, comparisons, family)
    result = adjusted(result)
    result.to_csv(output / f"paired_inference_{family}.csv", index=False)
    daily.to_csv(output / f"daily_loss_differentials_{family}.csv", index=False)
    audit.assign(analysis_status="exploratory_post_development").to_csv(
        output / f"fairness_audit_{family}.csv", index=False
    )
    print(f"\n===== {family.upper()} =====")
    print(result.to_string(index=False))


def main() -> None:
    args = parse_args()
    primary = normalize_keys(read_table(
        args.root / "benchmark_v10_directed_development" / "predictions_development.parquet"
    ))
    robust = normalize_keys(read_table(
        args.root / "benchmark_v10_directed_robustness" / "predictions_development_robust_linear.parquet"
    ))
    predictions = add_losses(pd.concat([primary, robust], ignore_index=True, sort=False))
    output = args.root / "benchmark_v10_directed_robustness"
    run_family(predictions, CURRENT_GRAPH, "current_graph_exploratory", output)
    run_family(predictions, DYNAMIC_INCREMENT, "dynamic_increment_exploratory", output)


if __name__ == "__main__":
    main()
