"""Create a recursive, immutable reproducibility manifest for v10/v10.1.

Unlike the historical v01 manifest, this version finds the main protocol in
``docs/protocols`` and never overwrites an existing freeze.  It can be run at
the prospective pre-confirmatory point, before development or holdout outcomes
for the new specification are inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from v10_integrity import sha256_file


COMMON_REQUIRED = [
    "graphs_v10/hyperparameter_selection.csv",
    "graphs_v10/graph_design.json",
    "model_audit/directed_split_dates_v10.csv",
    "model_audit/directed_split_audit_v10.csv",
    "model_audit/directed_split_design_v10.json",
]

DEVELOPMENT_RESULTS = [
    "predictions_val.parquet",
    "predictions_development.parquet",
    "sample_coverage_val.csv",
    "sample_coverage_development.csv",
    "model_metrics.csv",
    "paired_inference_core.csv",
    "paired_inference_regime.csv",
    "paired_inference_placebo.csv",
    "fairness_audit.csv",
    "stress_coverage.csv",
]


def slug(spec_version: str) -> str:
    return "v10" if spec_version == "v10" else "v10_1"


def development_name(spec_version: str) -> str:
    return f"benchmark_{slug(spec_version)}_directed_development"


def confirmatory_name(spec_version: str) -> str:
    return f"benchmark_{slug(spec_version)}_directed_confirmatory"


def required_data_files(spec_version: str, phase: str) -> list[str]:
    development = development_name(spec_version)
    confirmatory = confirmatory_name(spec_version)
    required = [
        f"{confirmatory}/frozen_hyperparameters.json",
        f"{confirmatory}/frozen_design.json",
        f"{development}/hyperparameter_selection.csv",
        *COMMON_REQUIRED,
    ]
    if spec_version == "v10.1":
        required.append(f"{development}/model_specification.json")
    if phase == "development-complete":
        required.extend(f"{development}/{name}" for name in DEVELOPMENT_RESULTS)
        required.extend([
            "model_data/directed_graph_features.parquet",
            "market_factor_v10/market_factor_coverage.csv",
        ])
    return required


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_v02"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--spec-version", choices=["v10", "v10.1"], default="v10.1")
    parser.add_argument(
        "--phase",
        choices=["preconfirmatory", "development-complete"],
        default="preconfirmatory",
    )
    return parser.parse_args()


def record(
    path: Path,
    base: Path,
    group: str,
    logical_root: str | None = None,
) -> dict:
    relative = str(path.resolve().relative_to(base.resolve()))
    display_path = f"{logical_root}/{relative}" if logical_root else relative
    return {
        "group": group,
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_create_once(path: Path, text: str) -> bool:
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return False
        raise RuntimeError(
            f"Manifest already exists with different content: {path}. "
            "Keep it immutable and create a new specification version."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    project = args.project_root.resolve()
    if not (project / "docs/protocols/RESEARCH_PROTOCOL_V10_DIRECTED_VOLATILITY_NETWORKS.md").exists():
        raise RuntimeError("Main v10 protocol is missing from docs/protocols")

    relative_data = required_data_files(args.spec_version, args.phase)
    data_paths = [root / relative for relative in relative_data]
    missing = [str(path) for path in data_paths if not path.exists()]
    if missing:
        raise RuntimeError("Required manifest inputs are missing:\n" + "\n".join(missing))

    code_paths = sorted((project / "v10_directed").glob("*.py"))
    code_paths += sorted((project / "v10_directed").glob("*.txt"))
    protocol_paths = sorted((project / "docs/protocols").rglob("*.md"))
    workflow_paths = sorted(
        path
        for path in (project / "docs").rglob("*.md")
        if (project / "docs/protocols") not in path.parents
    )
    workflow_paths += sorted((project / ".github/workflows").glob("*.yml"))
    workflow_paths += [
        path for path in (project / "README.md", project / "pyproject.toml") if path.exists()
    ]

    records = []
    records.extend(record(path, project, "code") for path in code_paths)
    records.extend(record(path, project, "protocol") for path in protocol_paths)
    records.extend(record(path, project, "workflow") for path in workflow_paths)
    records.extend(
        record(path, root, "data_or_result", "DATA_ROOT") for path in data_paths
    )
    table = pd.DataFrame(records).drop_duplicates("path").sort_values(["group", "path"])
    aggregate = hashlib.sha256(
        "\n".join(f"{row.sha256}  {row.path}" for row in table.itertuples()).encode("utf-8")
    ).hexdigest()
    payload = {
        "manifest_version": "reproducibility-manifest-02",
        "spec_version": args.spec_version,
        "phase": args.phase,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_end": "2026-08-10",
        "confirmatory_holdout_status": "sealed",
        "files": len(table),
        "aggregate_sha256": aggregate,
        "records": table.to_dict("records"),
    }

    output = root / confirmatory_name(args.spec_version)
    stem = f"frozen_manifest_{slug(args.spec_version)}_v02"
    json_path = output / f"{stem}.json"
    csv_path = output / f"{stem}.csv"
    if json_path.exists() or csv_path.exists():
        raise RuntimeError(
            f"Manifest already exists under {output}; it will not be overwritten."
        )
    json_text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    csv_text = table.to_csv(index=False)
    write_create_once(json_path, json_text)
    write_create_once(csv_path, csv_text)
    print(f"saved {len(table)} frozen records")
    print(f"aggregate_sha256={aggregate}")
    print(json_path)


if __name__ == "__main__":
    main()
