"""Small storage and output protocols that keep Dataiku imports out of core."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class ImageSource(Protocol):
    """Read-only image-folder abstraction."""

    @property
    def identifier(self) -> str:
        ...

    def list_paths(self) -> list[str]:
        ...

    def read_bytes(self, path: str) -> bytes:
        ...

    def size(self, path: str) -> int | None:
        ...


class ArtifactStore(Protocol):
    """Filesystem-like output store used for caches and run publication."""

    @property
    def identifier(self) -> str:
        ...

    def exists(self, path: str) -> bool:
        ...

    def read_bytes(self, path: str) -> bytes:
        ...

    def write_bytes(self, path: str, data: bytes) -> None:
        ...

    def upload_file(self, path: str, local_path: str) -> None:
        ...

    def list_paths(self, prefix: str = "") -> list[str]:
        ...


class ReviewDatasetSink(Protocol):
    """Fixed-schema sink for the one-row-per-image review dataset."""

    def write_rows(self, rows: Sequence[dict[str, object]]) -> None:
        ...
