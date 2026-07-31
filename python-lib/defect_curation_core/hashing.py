"""Content hashing and canonical serialization helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from defect_curation_core.errors import ConfigurationError


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def hash_join(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def normalize_relative_path(path: str) -> str:
    """Normalize a managed-folder path to a safe POSIX relative path."""

    candidate = path.replace("\\", "/").lstrip("/")
    pure = PurePosixPath(candidate)
    if not candidate or candidate == ".":
        raise ConfigurationError("Managed-folder path is empty")
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ConfigurationError(f"Unsafe managed-folder path: {path!r}")
    return pure.as_posix()


def validate_expected_sha256(actual: str, expected: str | None, *, label: str) -> None:
    if expected is None or not expected.strip():
        return
    normalized = expected.strip().lower()
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ConfigurationError(f"{label} expected SHA-256 is not a 64-character hexadecimal digest")
    if actual.lower() != normalized:
        raise ConfigurationError(f"{label} SHA-256 mismatch: expected {normalized}, got {actual.lower()}")
