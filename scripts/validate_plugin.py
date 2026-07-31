#!/usr/bin/env python3
"""Static repository validation that does not require Dataiku DSS."""

from __future__ import annotations

import argparse
import ast
import compileall
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PATHS = (
    "plugin.json",
    "LICENSE",
    "README.md",
    "code-env/python/desc.json",
    "code-env/python/spec/requirements.txt",
    "custom-recipes/cluster-defect-instances/recipe.json",
    "custom-recipes/cluster-defect-instances/recipe.py",
    "python-lib/Configs/config.yaml",
    "python-lib/defect_curation_core/pipeline.py",
    "python-lib/defect_curation_plugin/recipe_entrypoints.py",
    "resources/schemas/review_dataset.schema.json",
    "resources/schemas/instance_record.schema.json",
    "resources/schemas/run_manifest.schema.json",
)
FORBIDDEN_MODEL_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(f"Invalid JSON at {path.relative_to(ROOT)}: {exc}") from exc


def _validate_plugin_contract() -> None:
    for relative in REQUIRED_PATHS:
        assert (ROOT / relative).exists(), f"Missing required repository path: {relative}"

    plugin = _load_json(ROOT / "plugin.json")
    assert plugin["id"] == "rfdetr-dinov3-defect-clustering"
    assert plugin["version"]
    recipe = _load_json(ROOT / "custom-recipes/cluster-defect-instances/recipe.json")
    input_roles = {role["name"]: role for role in recipe["inputRoles"]}
    output_roles = {role["name"]: role for role in recipe["outputRoles"]}
    assert set(input_roles) == {"images_folder"}
    assert input_roles["images_folder"]["acceptsManagedFolder"] is True
    assert set(output_roles) == {"review_dataset", "artifact_bundle"}
    assert output_roles["review_dataset"]["acceptsDataset"] is True
    assert output_roles["artifact_bundle"]["acceptsManagedFolder"] is True
    assert {item["name"] for item in recipe["params"]} == {
        "cluster_count",
        "detection_threshold",
        "box_padding_fraction",
        "max_detections_per_image",
        "force_recompute",
    }


def _validate_schemas() -> None:
    schema_paths = sorted((ROOT / "resources" / "schemas").glob("*.json"))
    assert schema_paths, "No JSON schemas found"
    seen_ids: dict[str, str] = {}
    for path in schema_paths:
        document = _load_json(path)
        assert document.get("$schema"), f"Schema has no $schema: {path.name}"
        schema_id = document.get("$id")
        if schema_id:
            assert schema_id not in seen_ids, (
                f"Duplicate schema $id {schema_id!r}: {seen_ids[schema_id]} and {path.name}"
            )
            seen_ids[schema_id] = path.name
        try:
            import jsonschema
        except ImportError:
            continue
        jsonschema.Draft202012Validator.check_schema(document)


def _validate_no_dataiku_in_core() -> None:
    core = ROOT / "python-lib" / "defect_curation_core"
    violations: list[str] = []
    for path in core.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            if any(name == "dataiku" or name.startswith("dataiku.") for name in names):
                violations.append(path.relative_to(ROOT).as_posix())
    assert not violations, f"Core package imports dataiku: {sorted(set(violations))}"


def _validate_no_runtime_downloads() -> None:
    source_roots = [ROOT / "python-lib", ROOT / "custom-recipes"]
    forbidden = (
        "requests.get",
        "requests.post",
        "urllib.request",
        "wget",
        "curl ",
        "torch.hub.load",
        'pretrained=True',
        "hf_hub:",
    )
    violations: list[str] = []
    for source_root in source_roots:
        for path in source_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    violations.append(f"{path.relative_to(ROOT)}: {token}")
    assert not violations, "Potential runtime download calls found:\n" + "\n".join(violations)


def _validate_no_bundled_model_weights() -> None:
    violations = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix.lower() in FORBIDDEN_MODEL_SUFFIXES
    )
    assert not violations, (
        "Model/checkpoint files must be provisioned separately, not bundled in the plugin:\n"
        + "\n".join(violations)
    )


def _validate_no_symlinks() -> None:
    violations = sorted(
        path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_symlink()
    )
    assert not violations, "Repository symlinks are not allowed in release inputs:\n" + "\n".join(violations)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-compile", action="store_true", help="Skip Python bytecode compilation")
    args = parser.parse_args()

    _validate_plugin_contract()
    _validate_schemas()
    _validate_no_dataiku_in_core()
    _validate_no_runtime_downloads()
    _validate_no_bundled_model_weights()
    _validate_no_symlinks()
    if not args.no_compile:
        ok = compileall.compile_dir(str(ROOT / "python-lib"), quiet=1)
        ok &= compileall.compile_dir(str(ROOT / "custom-recipes"), quiet=1)
        assert ok, "Python compilation failed"
    print("Plugin repository validation passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
