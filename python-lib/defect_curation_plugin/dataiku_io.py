"""Adapters from Dataiku managed folders/datasets to pure-core protocols."""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, BinaryIO

from defect_curation_core.errors import ArtifactError, ConfigurationError
from defect_curation_core.hashing import normalize_relative_path

from defect_curation_plugin.dataiku_schema import REVIEW_DATASET_COLUMNS, REVIEW_DATASET_SCHEMA


def _try_local_root(folder: Any) -> Path | None:
    """Return a local folder root when DSS exposes one, otherwise ``None``."""

    try:
        raw = folder.get_path()
    except Exception:
        return None
    if raw is None:
        return None
    path = Path(str(raw))
    return path if path.is_dir() else None


def _read_stream(stream: BinaryIO, *, chunk_size: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _close_if_possible(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


class DataikuManagedFolderImageSource:
    """Image source supporting both filesystem and remote managed folders."""

    def __init__(self, folder: Any, *, identifier: str, read_chunk_size: int = 1_048_576) -> None:
        if read_chunk_size < 1:
            raise ConfigurationError("Managed-folder read chunk size must be positive")
        self.folder = folder
        self.identifier = identifier
        self.read_chunk_size = int(read_chunk_size)
        self._local_root = _try_local_root(folder)
        self._raw_paths: dict[str, str] | None = None

    def _refresh_path_map(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for raw in self.folder.list_paths_in_partition():
            # DSS commonly returns paths with a leading slash. The core contract
            # is always normalized relative POSIX paths.
            normalized = normalize_relative_path(str(raw))
            if normalized in result and result[normalized] != str(raw):
                raise ConfigurationError(
                    f"Managed folder exposes colliding normalized paths: {result[normalized]!r} and {raw!r}"
                )
            result[normalized] = str(raw)
        self._raw_paths = result
        return result

    def list_paths(self) -> list[str]:
        return list(self._refresh_path_map())

    def _remote_path(self, normalized: str) -> str:
        if self._raw_paths is None or normalized not in self._raw_paths:
            self._refresh_path_map()
        assert self._raw_paths is not None
        return self._raw_paths.get(normalized, normalized)

    def size(self, relative_path: str) -> int | None:
        normalized = normalize_relative_path(relative_path)
        if self._local_root is not None:
            try:
                return (self._local_root / Path(normalized)).stat().st_size
            except OSError:
                return None
        try:
            details = self.folder.get_path_details(self._remote_path(normalized))
            if not details or not bool(details.get("exists", True)) or bool(details.get("directory", False)):
                return None
            value = details.get("size")
            return None if value is None else int(value)
        except Exception:
            return None

    def read_bytes(self, relative_path: str) -> bytes:
        normalized = normalize_relative_path(relative_path)
        if self._local_root is not None:
            candidate = (self._local_root / Path(normalized)).resolve()
            try:
                candidate.relative_to(self._local_root.resolve())
            except ValueError as exc:
                raise ConfigurationError("Managed-folder path escaped the folder root") from exc
            return candidate.read_bytes()

        stream_or_context = self.folder.get_download_stream(self._remote_path(normalized))
        # Dataiku stream objects support the context-manager protocol in current
        # DSS releases, but this fallback keeps the adapter compatible with older
        # implementations and simple contract-test fakes.
        if hasattr(stream_or_context, "__enter__"):
            with stream_or_context as stream:
                return _read_stream(stream, chunk_size=self.read_chunk_size)
        try:
            return _read_stream(stream_or_context, chunk_size=self.read_chunk_size)
        finally:
            _close_if_possible(stream_or_context)


class DataikuManagedFolderArtifactStore:
    """Artifact store for local or remote Dataiku managed folders."""

    def __init__(self, folder: Any, *, identifier: str) -> None:
        self.folder = folder
        self.identifier = identifier
        self._local_root = _try_local_root(folder)

    @staticmethod
    def _path(relative_path: str) -> str:
        return normalize_relative_path(relative_path)

    def _local_path(self, normalized: str) -> Path:
        assert self._local_root is not None
        root = self._local_root.resolve()
        candidate = (root / Path(normalized)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ConfigurationError("Artifact path escaped the managed-folder root") from exc
        return candidate

    def list_paths(self, prefix: str = "") -> list[str]:
        normalized_prefix = prefix.replace("\\", "/").lstrip("/")
        if self._local_root is not None:
            root = self._local_root.resolve()
            return sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_file() and path.relative_to(root).as_posix().startswith(normalized_prefix)
            )
        result: list[str] = []
        for raw in self.folder.list_paths_in_partition():
            try:
                normalized = normalize_relative_path(str(raw))
            except ConfigurationError:
                continue
            if normalized.startswith(normalized_prefix):
                result.append(normalized)
        return sorted(result)

    def exists(self, relative_path: str) -> bool:
        normalized = self._path(relative_path)
        if self._local_root is not None:
            return self._local_path(normalized).is_file()
        try:
            details = self.folder.get_path_details(normalized)
            return (
                bool(details)
                and bool(details.get("exists", True))
                and not bool(details.get("directory", False))
            )
        except Exception:
            return False

    def read_bytes(self, relative_path: str) -> bytes:
        normalized = self._path(relative_path)
        if self._local_root is not None:
            try:
                return self._local_path(normalized).read_bytes()
            except OSError as exc:
                raise ArtifactError(f"Could not read artifact {normalized}") from exc
        try:
            source = self.folder.get_download_stream(normalized)
            if hasattr(source, "__enter__"):
                with source as stream:
                    return _read_stream(stream, chunk_size=1_048_576)
            try:
                return _read_stream(source, chunk_size=1_048_576)
            finally:
                _close_if_possible(source)
        except Exception as exc:
            raise ArtifactError(f"Could not read artifact {normalized}") from exc

    def write_bytes(self, relative_path: str, data: bytes) -> None:
        normalized = self._path(relative_path)
        if self._local_root is not None:
            target = self._local_path(normalized)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, target)
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                raise ArtifactError(f"Could not write artifact {normalized}") from exc
            return
        try:
            upload_data = getattr(self.folder, "upload_data", None)
            if callable(upload_data):
                upload_data(normalized, data)
            else:
                self.folder.upload_stream(normalized, io.BytesIO(data))
        except Exception as exc:
            raise ArtifactError(f"Could not write artifact {normalized}") from exc

    def upload_file(self, relative_path: str, local_path: str) -> None:
        normalized = self._path(relative_path)
        if self._local_root is not None:
            target = self._local_path(normalized)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            try:
                with open(local_path, "rb") as src, open(temporary, "wb") as dst:
                    while chunk := src.read(4 * 1024 * 1024):
                        dst.write(chunk)
                os.replace(temporary, target)
            except OSError as exc:
                temporary.unlink(missing_ok=True)
                raise ArtifactError(f"Could not upload artifact {normalized}") from exc
            return
        try:
            self.folder.upload_file(normalized, local_path)
        except Exception as exc:
            raise ArtifactError(f"Could not upload artifact {normalized}") from exc


class DataikuReviewDatasetSink:
    """Write review rows using a stable, explicitly declared DSS schema."""

    def __init__(self, dataset: Any, *, identifier: str) -> None:
        self.dataset = dataset
        self.identifier = identifier

    def write_rows(self, rows: Sequence[dict[str, object]]) -> None:
        try:
            self.dataset.write_schema(REVIEW_DATASET_SCHEMA)
            with self.dataset.get_writer() as writer:
                for row in rows:
                    unknown = set(row) - set(REVIEW_DATASET_COLUMNS)
                    missing = set(REVIEW_DATASET_COLUMNS) - set(row)
                    if unknown:
                        raise ConfigurationError(
                            f"Review row contains columns outside the fixed schema: {sorted(unknown)}"
                        )
                    if missing:
                        raise ConfigurationError(
                            f"Review row is missing fixed-schema columns: {sorted(missing)}"
                        )
                    writer.write_row_dict(
                        {column: row.get(column) for column in REVIEW_DATASET_COLUMNS}
                    )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ArtifactError(f"Could not write review dataset {self.identifier}") from exc
