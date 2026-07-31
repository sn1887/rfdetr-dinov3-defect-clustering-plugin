"""Validated detection, embedding, and positional-basis caches."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass

import numpy as np

from defect_curation_core.artifacts.signatures import (
    DETECTION_CACHE_SCHEMA_VERSION,
)
from defect_curation_core.errors import ArtifactError
from defect_curation_core.hashing import canonical_json_bytes, sha256_bytes
from defect_curation_core.io.protocols import ArtifactStore
from defect_curation_core.types import Detection


@dataclass(frozen=True, slots=True)
class CachedBasis:
    array: np.ndarray
    sha256: str


class CacheManager:
    def __init__(self, store: ArtifactStore) -> None:
        self.store = store

    @staticmethod
    def detection_path(detector_signature: str, image_sha256: str) -> str:
        return f"cache/detections/{detector_signature}/{image_sha256}.json"

    @staticmethod
    def embedding_path(embedding_signature: str, instance_key: str) -> str:
        return f"cache/embeddings/{embedding_signature}/{instance_key}.f16.npy"

    @staticmethod
    def basis_path(basis_input_signature: str) -> str:
        return f"specs/positional_basis/{basis_input_signature}.f32.npy"

    @staticmethod
    def basis_spec_path(basis_input_signature: str) -> str:
        return f"specs/positional_basis/{basis_input_signature}.json"

    def load_detections(
        self,
        *,
        detector_signature: str,
        image_sha256: str,
        width: int,
        height: int,
        threshold: float,
        max_detections: int,
    ) -> list[Detection] | None:
        path = self.detection_path(detector_signature, image_sha256)
        if not self.store.exists(path):
            return None
        try:
            payload = json.loads(self.store.read_bytes(path).decode("utf-8"))
            if payload.get("schema_version") != DETECTION_CACHE_SCHEMA_VERSION:
                return None
            if payload.get("detector_signature") != detector_signature:
                return None
            if payload.get("image_sha256") != image_sha256:
                return None
            if int(payload.get("image_width")) != width or int(payload.get("image_height")) != height:
                return None
            raw_detections = payload.get("detections", [])
            if not isinstance(raw_detections, list) or len(raw_detections) > max_detections:
                return None
            detections = [Detection.from_dict(item) for item in raw_detections]
            for detection in detections:
                box = detection.bbox
                if not 0.0 <= detection.score <= 1.0 or detection.score < threshold:
                    return None
                if (
                    box.xmin < 0.0
                    or box.ymin < 0.0
                    or box.xmax > float(width)
                    or box.ymax > float(height)
                    or box.width < 1.0
                    or box.height < 1.0
                ):
                    return None
            return detections
        except Exception as exc:
            raise ArtifactError(f"Corrupt detection cache entry {path}: {exc}") from exc

    def save_detections(
        self,
        *,
        detector_signature: str,
        image_sha256: str,
        width: int,
        height: int,
        detections: list[Detection],
    ) -> None:
        path = self.detection_path(detector_signature, image_sha256)
        payload = {
            "schema_version": DETECTION_CACHE_SCHEMA_VERSION,
            "detector_signature": detector_signature,
            "image_sha256": image_sha256,
            "image_width": int(width),
            "image_height": int(height),
            "detections": [item.to_dict() for item in detections],
        }
        self.store.write_bytes(path, canonical_json_bytes(payload))

    def load_embedding(
        self,
        *,
        embedding_signature: str,
        instance_key: str,
        expected_dimension: int,
    ) -> np.ndarray | None:
        path = self.embedding_path(embedding_signature, instance_key)
        if not self.store.exists(path):
            return None
        try:
            value = np.load(io.BytesIO(self.store.read_bytes(path)), allow_pickle=False)
            if value.shape != (expected_dimension,) or value.dtype != np.float16:
                return None
            as_float = value.astype(np.float32)
            if not np.isfinite(as_float).all():
                return None
            norm = float(np.linalg.norm(as_float))
            if not 0.98 <= norm <= 1.02:
                return None
            return value.copy()
        except Exception as exc:
            raise ArtifactError(f"Corrupt embedding cache entry {path}: {exc}") from exc

    def save_embedding(
        self,
        *,
        embedding_signature: str,
        instance_key: str,
        vector: np.ndarray,
    ) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ArtifactError("Embedding cache value must be a finite 1D vector")
        norm = float(np.linalg.norm(value))
        if norm <= 1e-12:
            raise ArtifactError("Cannot cache a zero-norm embedding")
        persisted = (value / norm).astype(np.float16)
        buffer = io.BytesIO()
        np.save(buffer, persisted, allow_pickle=False)
        self.store.write_bytes(self.embedding_path(embedding_signature, instance_key), buffer.getvalue())
        return persisted.copy()

    def load_basis(
        self,
        *,
        basis_input_signature: str,
        expected_shape: tuple[int, int],
    ) -> CachedBasis | None:
        path = self.basis_path(basis_input_signature)
        spec_path = self.basis_spec_path(basis_input_signature)
        if not self.store.exists(path) or not self.store.exists(spec_path):
            return None
        try:
            raw = self.store.read_bytes(path)
            spec = json.loads(self.store.read_bytes(spec_path).decode("utf-8"))
            actual_sha = sha256_bytes(raw)
            if spec.get("basis_input_signature") != basis_input_signature:
                return None
            if spec.get("sha256") != actual_sha:
                return None
            array = np.load(io.BytesIO(raw), allow_pickle=False)
            if array.shape != expected_shape or array.dtype != np.float32 or not np.isfinite(array).all():
                return None
            gram = array.T @ array
            if not np.allclose(gram, np.eye(expected_shape[1], dtype=np.float32), atol=1e-4, rtol=0.0):
                return None
            return CachedBasis(array=array, sha256=actual_sha)
        except Exception as exc:
            raise ArtifactError(f"Corrupt positional-basis cache {path}: {exc}") from exc

    def save_basis(self, *, basis_input_signature: str, array: np.ndarray) -> CachedBasis:
        value = np.asarray(array, dtype=np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise ArtifactError("Positional basis must be a finite 2D float32 matrix")
        if value.shape[1] < 1 or value.shape[1] >= value.shape[0]:
            raise ArtifactError("Positional basis must have between 1 and C-1 columns")
        gram = value.T @ value
        if not np.allclose(gram, np.eye(value.shape[1], dtype=np.float32), atol=1e-4, rtol=0.0):
            raise ArtifactError("Positional basis columns must be orthonormal")
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        raw = buffer.getvalue()
        digest = sha256_bytes(raw)
        self.store.write_bytes(self.basis_path(basis_input_signature), raw)
        self.store.write_bytes(
            self.basis_spec_path(basis_input_signature),
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "basis_input_signature": basis_input_signature,
                    "shape": list(value.shape),
                    "dtype": "float32",
                    "sha256": digest,
                }
            ),
        )
        return CachedBasis(array=value, sha256=digest)

    def write_signature_spec(self, *, signature: str, kind: str, payload: dict[str, object]) -> None:
        document = {
            "schema_version": 1,
            "kind": kind,
            "signature": signature,
            "payload": payload,
        }
        self.store.write_bytes(f"specs/{signature}.json", canonical_json_bytes(document))
