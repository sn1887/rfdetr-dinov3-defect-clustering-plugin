#!/usr/bin/env python3
"""Build a deterministic, directly installable Dataiku plugin repository zip."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".github-cache",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "venv",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}
FIXED_TIMESTAMP = (2026, 7, 29, 0, 0, 0)


def include(path: Path) -> bool:
    # Never dereference symlinks while packaging. Besides making archives less
    # reproducible, a repository-local symlink could unintentionally copy data
    # from outside the plugin tree into a release artifact.
    if path.is_symlink():
        return False
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name in {".coverage", ".DS_Store"}:
        return False
    return path.is_file()


def build(output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted((path for path in ROOT.rglob("*") if include(path)), key=lambda p: p.relative_to(ROOT).as_posix())
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=FIXED_TIMESTAMP)
            mode = 0o755 if os.access(path, os.X_OK) else 0o644
            info.external_attr = mode << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            archive.writestr(info, path.read_bytes())
    os.replace(temporary, output)
    print(f"Wrote {output} ({len(files)} files, {output.stat().st_size} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "dist" / "rfdetr-dinov3-defect-clustering-plugin.zip",
    )
    args = parser.parse_args()
    build(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
