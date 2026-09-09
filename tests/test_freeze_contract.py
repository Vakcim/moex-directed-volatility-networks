from __future__ import annotations

import json
import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from freeze_reproducibility_manifest_v02 import required_data_files
from train_directed_har_benchmark_v01 import (
    FEATURE_SETS_V101,
    freeze,
    frozen_parameters,
)
from v10_integrity import (
    sha256_file,
    stable_json_sha256,
    validate_reproducibility_manifest,
)


class FreezeContractTests(unittest.TestCase):
    def test_v101_manifest_requires_model_specification(self) -> None:
        required = required_data_files("v10.1", "preconfirmatory")
        self.assertIn(
            "benchmark_v10_1_directed_development/model_specification.json",
            required,
        )
        self.assertIn(
            "benchmark_v10_1_directed_confirmatory/frozen_design.json",
            required,
        )

    def test_confirmatory_loader_uses_and_validates_frozen_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "graphs_v10/graph_design.json"
            source.parent.mkdir(parents=True)
            source.write_text('{"design": 1}\n')
            out = root / "benchmark_v10_1_directed_confirmatory"
            out.mkdir(parents=True)
            payload = {
                "spec_version": "v10.1",
                "ridge_alphas": {model: 100 for model in FEATURE_SETS_V101},
                "catboost_config_id": "cb02",
                "catboost_params": {"depth": 4},
                "feature_schema_sha256": stable_json_sha256(FEATURE_SETS_V101),
                "source_sha256": {"graphs_v10/graph_design.json": sha256_file(source)},
            }
            frozen = out / "frozen_hyperparameters.json"
            frozen.write_text(json.dumps(payload))
            (out / "frozen_design.json").write_text(json.dumps({
                "spec_version": "v10.1",
                "losses_sealed_until_close": True,
                "confirmatory_predictions_must_exclude_targets_and_losses": True,
            }))

            alphas, config_id, params, digest = frozen_parameters(root, "v10.1")
            self.assertEqual(set(alphas), set(FEATURE_SETS_V101))
            self.assertEqual(config_id, "cb02")
            self.assertEqual(params, {"depth": 4})
            self.assertEqual(digest, sha256_file(frozen))

            source.write_text('{"design": 2}\n')
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                frozen_parameters(root, "v10.1")

    def test_freeze_is_idempotent_but_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dev = root / "benchmark_v10_1_directed_development"
            graphs = root / "graphs_v10"
            dev.mkdir(parents=True)
            graphs.mkdir(parents=True)
            schema_hash = stable_json_sha256(FEATURE_SETS_V101)
            rows = [{
                "model": model,
                "alpha": 100.0,
                "selected": True,
                "feature_schema_sha256": schema_hash,
            } for model in FEATURE_SETS_V101]
            import pandas as pd

            selection_path = dev / "hyperparameter_selection.csv"
            pd.DataFrame(rows).to_csv(selection_path, index=False)
            (dev / "model_specification.json").write_text("{}\n")
            pd.DataFrame([{
                "learner": "catboost", "selected": True, "config_id": "cb02"
            }]).to_csv(graphs / "hyperparameter_selection.csv", index=False)
            (graphs / "graph_design.json").write_text("{}\n")
            args = argparse.Namespace(
                root=root,
                variant="raw",
                spec_version="v10.1",
                force=False,
            )
            with redirect_stdout(io.StringIO()):
                freeze(args)
            frozen = root / "benchmark_v10_1_directed_confirmatory/frozen_hyperparameters.json"
            original_hash = sha256_file(frozen)
            with redirect_stdout(io.StringIO()):
                freeze(args)
            self.assertEqual(sha256_file(frozen), original_hash)

            rows[0]["alpha"] = 10.0
            pd.DataFrame(rows).to_csv(selection_path, index=False)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                with redirect_stdout(io.StringIO()):
                    freeze(args)
            self.assertEqual(sha256_file(frozen), original_hash)

    def test_manifest_detects_post_freeze_code_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            project = base / "project"
            data = base / "data_v02"
            code = project / "v10_directed/runner.py"
            protocol = project / "docs/protocols/protocol.md"
            design = data / "graphs_v10/graph_design.json"
            for path in (code, protocol, design):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"content:{path.name}\n")
            records = [
                {"group": "code", "path": "v10_directed/runner.py", "sha256": sha256_file(code)},
                {"group": "protocol", "path": "docs/protocols/protocol.md", "sha256": sha256_file(protocol)},
                {"group": "data_or_result", "path": "DATA_ROOT/graphs_v10/graph_design.json", "sha256": sha256_file(design)},
            ]
            manifest = data / "manifest.json"
            manifest.write_text(json.dumps({
                "manifest_version": "reproducibility-manifest-02",
                "spec_version": "v10.1",
                "records": records,
            }))
            digest = validate_reproducibility_manifest(
                data, project, manifest, spec_version="v10.1"
            )
            self.assertEqual(digest, sha256_file(manifest))
            code.write_text("changed\n")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_reproducibility_manifest(
                    data, project, manifest, spec_version="v10.1"
                )
