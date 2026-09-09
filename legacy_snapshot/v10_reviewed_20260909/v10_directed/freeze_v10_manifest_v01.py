"""Create an immutable manifest for the completed v10 development experiment.

This does not change models, predictions, losses, or the confirmatory design.
Run the existing ``train_directed_har_benchmark_v01.py --stage freeze`` first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REQUIRED_DATA_FILES = [
    "benchmark_v10_directed_confirmatory/frozen_hyperparameters.json",
    "benchmark_v10_directed_confirmatory/frozen_design.json",
    "benchmark_v10_directed_development/hyperparameter_selection.csv",
    "benchmark_v10_directed_development/predictions_val.parquet",
    "benchmark_v10_directed_development/predictions_development.parquet",
    "benchmark_v10_directed_development/sample_coverage_val.csv",
    "benchmark_v10_directed_development/sample_coverage_development.csv",
    "benchmark_v10_directed_development/model_metrics.csv",
    "benchmark_v10_directed_development/paired_inference_core.csv",
    "benchmark_v10_directed_development/paired_inference_regime.csv",
    "benchmark_v10_directed_development/paired_inference_placebo.csv",
    "benchmark_v10_directed_development/fairness_audit.csv",
    "benchmark_v10_directed_development/stress_coverage.csv",
    "graphs_v10/hyperparameter_selection.csv",
    "graphs_v10/graph_design.json",
    "model_audit/directed_split_dates_v10.csv",
    "model_audit/directed_split_audit_v10.csv",
    "model_audit/directed_split_design_v10.json",
    "model_data/directed_graph_features.parquet",
    "market_factor_v10/market_factor_coverage.csv",
]

PROTOCOL_GLOBS = [
    "RESEARCH_PROTOCOL_V10*.md",
    "V10_IMPLEMENTATION_NOTE_*.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_v02"))
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Directory containing V10_DIRECTED_CODE_PACKAGE and protocol files.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def add_record(records: list[dict], path: Path, base: Path, group: str) -> None:
    records.append({
        "group": group,
        "path": str(path.resolve().relative_to(base.resolve())),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    })


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    project = args.project_root.resolve()
    code_dir = Path(__file__).resolve().parent
    output = root / "benchmark_v10_directed_confirmatory"
    manifest_json = output / "frozen_manifest_v10.json"
    manifest_csv = output / "frozen_manifest_v10.csv"

    if (manifest_json.exists() or manifest_csv.exists()) and not args.force:
        raise RuntimeError(
            "Frozen manifest already exists; inspect it instead of overwriting. "
            "Use --force only to correct a documented manifest-only error."
        )

    required = [root / relative for relative in REQUIRED_DATA_FILES]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Required freeze inputs are missing:\n" + "\n".join(missing))

    records: list[dict] = []
    for path in sorted(code_dir.glob("*.py")) + sorted(code_dir.glob("*.txt")) + sorted(code_dir.glob("*.md")):
        add_record(records, path, project, "code")
    for pattern in PROTOCOL_GLOBS:
        for path in sorted(project.glob(pattern)):
            if path.is_file():
                add_record(records, path, project, "protocol")
    for path in required:
        add_record(records, path, project, "data_or_result")

    table = pd.DataFrame(records).drop_duplicates("path").sort_values(["group", "path"])
    aggregate = hashlib.sha256(
        "\n".join(f"{row.sha256}  {row.path}" for row in table.itertuples()).encode("utf-8")
    ).hexdigest()
    payload = {
        "manifest_version": "v10-development-freeze-manifest-01",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_end": "2026-08-10",
        "confirmatory_holdout_status": "sealed",
        "files": len(table),
        "aggregate_sha256": aggregate,
        "records": table.to_dict("records"),
    }

    output.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    table.to_csv(manifest_csv, index=False)
    print(f"saved {len(table)} frozen records")
    print(f"aggregate_sha256={aggregate}")
    print(manifest_json)


if __name__ == "__main__":
    main()
