#!/usr/bin/env python3
"""Validate a locally exported run bundle using manifest and checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_bundle_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path while forbidding absolute/traversal/symlink escape."""

    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError(f"invalid bundle-relative path: {relative!r}")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"invalid bundle-relative path: {relative!r}")
    candidate = (root / Path(*pure.parts)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"bundle path escapes root: {relative!r}") from exc
    return candidate


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest.json must contain a JSON object")
    return payload


def validate_bundle(root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None, [f"manifest.json not found below {root}"]
    try:
        manifest = _load_manifest(manifest_path)
    except Exception as exc:
        return None, [f"invalid manifest.json: {exc}"]

    errors: list[str] = []
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        errors.append("manifest artifacts must be an object")
        artifacts = {}
    for relative, metadata in artifacts.items():
        try:
            path = safe_bundle_path(root, relative)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(metadata, dict):
            errors.append(f"invalid artifact metadata: {relative}")
            continue
        expected = metadata.get("sha256")
        if not isinstance(expected, str) or not _HEX_64.fullmatch(expected):
            errors.append(f"invalid artifact digest: {relative}")
            continue
        if not path.is_file():
            errors.append(f"missing: {relative}")
            continue
        observed = sha256(path)
        if observed != expected:
            errors.append(f"digest mismatch: {relative}")

    checksums = root / "checksums.sha256"
    if checksums.is_file():
        for line_number, line in enumerate(checksums.read_text(encoding="utf-8").splitlines(), start=1):
            if not line:
                continue
            try:
                expected, relative = line.split("  ", 1)
                if not _HEX_64.fullmatch(expected):
                    raise ValueError("digest is not lowercase SHA-256")
                path = safe_bundle_path(root, relative)
            except ValueError as exc:
                errors.append(f"malformed checksums.sha256 line {line_number}: {exc}")
                continue
            if not path.is_file() or sha256(path) != expected:
                errors.append(f"checksums.sha256 mismatch: {relative}")
    return manifest, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    manifest, errors = validate_bundle(args.run_directory)
    if errors:
        for error in errors:
            print(error)
        return 1
    assert manifest is not None
    print(f"Run bundle {manifest.get('run_id')} is internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
