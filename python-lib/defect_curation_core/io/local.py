"""Filesystem adapters used by local development and integration tests."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from defect_curation_core.hashing import normalize_relative_path


class LocalImageSource:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Image source root does not exist: {self.root}")

    @property
    def identifier(self) -> str:
        return str(self.root)

    def list_paths(self) -> list[str]:
        return [path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()]

    def _resolve(self, path: str) -> Path:
        normalized = normalize_relative_path(path)
        resolved = (self.root / normalized).resolve()
        if self.root not in resolved.parents:
            raise ValueError(f"Path escapes source root: {path}")
        return resolved

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def size(self, path: str) -> int | None:
        return self._resolve(path).stat().st_size


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def identifier(self) -> str:
        return str(self.root)

    def _resolve(self, path: str) -> Path:
        normalized = normalize_relative_path(path)
        resolved = (self.root / normalized).resolve()
        if self.root not in resolved.parents:
            raise ValueError(f"Path escapes artifact root: {path}")
        return resolved

    def exists(self, path: str) -> bool:
        return self._resolve(path).is_file()

    def read_bytes(self, path: str) -> bytes:
        return self._resolve(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)

    def upload_file(self, path: str, local_path: str) -> None:
        self.write_bytes(path, Path(local_path).read_bytes())

    def list_paths(self, prefix: str = "") -> list[str]:
        normalized_prefix = prefix.replace("\\", "/").lstrip("/")
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file() and path.relative_to(self.root).as_posix().startswith(normalized_prefix)
        )


class MemoryReviewDatasetSink:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def write_rows(self, rows: Sequence[dict[str, object]]) -> None:
        self.rows = [dict(row) for row in rows]
