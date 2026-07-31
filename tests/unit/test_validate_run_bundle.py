from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_module(repo_root: Path):
    path = repo_root / "scripts" / "validate_run_bundle.py"
    spec = importlib.util.spec_from_file_location("validate_run_bundle", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_validator_accepts_valid_artifact(tmp_path: Path, repo_root: Path) -> None:
    module = load_module(repo_root)
    artifact = tmp_path / "data.bin"
    artifact.write_bytes(b"payload")
    digest = module.sha256(artifact)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_id": "run", "artifacts": {"data.bin": {"sha256": digest}}}),
        encoding="utf-8",
    )
    (tmp_path / "checksums.sha256").write_text(f"{digest}  data.bin\n", encoding="utf-8")
    manifest, errors = module.validate_bundle(tmp_path)
    assert manifest is not None
    assert errors == []


def test_bundle_validator_rejects_path_traversal(tmp_path: Path, repo_root: Path) -> None:
    module = load_module(repo_root)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"run_id": "run", "artifacts": {"../outside": {"sha256": "0" * 64}}}),
        encoding="utf-8",
    )
    _manifest, errors = module.validate_bundle(tmp_path)
    assert any("invalid bundle-relative path" in error for error in errors)


def test_bundle_validator_reports_malformed_checksum_line(tmp_path: Path, repo_root: Path) -> None:
    module = load_module(repo_root)
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "run", "artifacts": {}}), encoding="utf-8")
    (tmp_path / "checksums.sha256").write_text("not-a-checksum\n", encoding="utf-8")
    _manifest, errors = module.validate_bundle(tmp_path)
    assert any("malformed checksums.sha256" in error for error in errors)
