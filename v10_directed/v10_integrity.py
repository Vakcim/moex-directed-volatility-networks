"""Integrity helpers shared by the prospective v10.1 workflow.

The functions in this module deliberately avoid looking at forecast targets or
losses.  They protect immutable design files, validate frozen inputs and make
incremental prediction checkpoints safe when the final 20-day block grows.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd


CheckpointDecision = Literal["skip", "recompute"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_keys(
    frame: pd.DataFrame,
    keys: Iterable[str] = ("TRADEDATE", "SECID"),
) -> pd.DataFrame:
    """Return sorted, normalized unique keys or fail on malformed input."""
    columns = list(keys)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise RuntimeError(f"Missing checkpoint key columns: {missing}")
    out = frame[columns].copy()
    if "TRADEDATE" in out:
        out["TRADEDATE"] = pd.to_datetime(out["TRADEDATE"], errors="coerce").dt.normalize()
        if out["TRADEDATE"].isna().any():
            raise RuntimeError("Invalid TRADEDATE in checkpoint keys")
    if "SECID" in out:
        out["SECID"] = out["SECID"].astype(str).str.upper()
        if out["SECID"].eq("").any():
            raise RuntimeError("Empty SECID in checkpoint keys")
    if out.duplicated(columns).any():
        raise RuntimeError("Duplicate checkpoint keys")
    return out.sort_values(columns).reset_index(drop=True)


def checkpoint_decision(
    path: Path,
    expected: pd.DataFrame,
    *,
    keys: Iterable[str] = ("TRADEDATE", "SECID"),
    force: bool = False,
    expected_metadata: dict[str, object] | None = None,
) -> CheckpointDecision:
    """Decide whether an existing checkpoint is complete.

    An exact key match is reusable.  A strict subset is the expected state for
    a live, partially filled final block and is recomputed atomically.  Any
    foreign or extra key indicates a stale/incompatible shard and fails closed.
    """
    if force or not path.exists():
        return "recompute"
    expected_keys = canonical_keys(expected, keys)
    actual = pd.read_parquet(path)
    if expected_metadata:
        for column, expected_value in expected_metadata.items():
            if column not in actual:
                raise RuntimeError(f"Checkpoint {path} is missing metadata column {column}")
            values = actual[column].drop_duplicates().tolist()
            if values != [expected_value]:
                raise RuntimeError(
                    f"Checkpoint {path} metadata mismatch for {column}: "
                    f"expected={expected_value!r}, observed={values!r}"
                )
    actual_keys = canonical_keys(actual, keys)
    expected_set = set(map(tuple, expected_keys.itertuples(index=False, name=None)))
    actual_set = set(map(tuple, actual_keys.itertuples(index=False, name=None)))
    if actual_set == expected_set:
        return "skip"
    if actual_set < expected_set:
        return "recompute"
    extras = sorted(actual_set - expected_set)[:5]
    missing = sorted(expected_set - actual_set)[:5]
    raise RuntimeError(
        f"Checkpoint {path} is incompatible with the current block; "
        f"example_extra={extras}, example_missing={missing}. "
        "Inspect it or rerun this exact shard with --force."
    )


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_once_json(path: Path, payload: dict) -> bool:
    """Create an immutable JSON file; identical reruns are harmless."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == payload:
            return False
        raise RuntimeError(
            f"Frozen file already exists with different content: {path}. "
            "Never overwrite it; use a new specification version."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def resolve_frozen_source(root: Path, recorded: str) -> Path:
    """Resolve both legacy absolute paths and new root-relative freeze paths."""
    raw = Path(recorded)
    if raw.exists():
        return raw
    if not raw.is_absolute():
        return root / raw
    parts = list(raw.parts)
    if root.name in parts:
        position = len(parts) - 1 - parts[::-1].index(root.name)
        return root.joinpath(*parts[position + 1 :])
    raise RuntimeError(f"Cannot relocate frozen source path: {recorded}")


def validate_frozen_sources(root: Path, source_sha256: dict[str, str]) -> None:
    for recorded, expected_hash in sorted(source_sha256.items()):
        path = resolve_frozen_source(root, recorded)
        if not path.exists():
            raise FileNotFoundError(f"Frozen source is missing: {path}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"Frozen source hash mismatch: {path}; "
                f"expected={expected_hash}, observed={observed_hash}"
            )


def validate_reproducibility_manifest(
    data_root: Path,
    project_root: Path,
    manifest_path: Path,
    *,
    spec_version: str,
) -> str:
    """Validate all scientific records in a frozen v02 manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing reproducibility manifest: {manifest_path}. "
            "Create it before confirmatory prediction."
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != "reproducibility-manifest-02":
        raise RuntimeError("Unsupported reproducibility manifest version")
    if payload.get("spec_version") != spec_version:
        raise RuntimeError("Reproducibility manifest specification mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Reproducibility manifest has no records")
    checked = 0
    for record in records:
        group = record.get("group")
        if group not in {"code", "protocol", "data_or_result"}:
            continue
        recorded_path = str(record.get("path", ""))
        if recorded_path.startswith("DATA_ROOT/"):
            path = data_root / recorded_path.removeprefix("DATA_ROOT/")
        else:
            path = project_root / recorded_path
        if not path.exists():
            raise FileNotFoundError(f"Manifest input is missing: {path}")
        observed = sha256_file(path)
        expected = str(record.get("sha256", ""))
        if observed != expected:
            raise RuntimeError(
                f"Reproducibility manifest hash mismatch: {path}; "
                f"expected={expected}, observed={observed}"
            )
        checked += 1
    if checked == 0:
        raise RuntimeError("Reproducibility manifest contains no scientific records")
    return sha256_file(manifest_path)
