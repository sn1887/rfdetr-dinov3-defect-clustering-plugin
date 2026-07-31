from __future__ import annotations

import numpy as np
import pytest
from defect_curation_core.artifacts.cache import CacheManager
from defect_curation_core.errors import ArtifactError
from defect_curation_core.io.local import LocalArtifactStore
from defect_curation_core.types import BBoxXYXY, Detection


def test_detection_embedding_and_basis_cache_roundtrip(tmp_path) -> None:
    manager = CacheManager(LocalArtifactStore(tmp_path))
    detections = [Detection(BBoxXYXY(1, 2, 10, 12), 0.9, 0)]
    manager.save_detections(
        detector_signature="a" * 64,
        image_sha256="b" * 64,
        width=100,
        height=80,
        detections=detections,
    )
    assert manager.load_detections(
        detector_signature="a" * 64,
        image_sha256="b" * 64,
        width=100,
        height=80,
        threshold=0.35,
        max_detections=20,
    ) == detections

    vector = np.arange(1, 9, dtype=np.float32)
    manager.save_embedding(
        embedding_signature="c" * 64,
        instance_key="d" * 64,
        vector=vector,
    )
    loaded = manager.load_embedding(
        embedding_signature="c" * 64,
        instance_key="d" * 64,
        expected_dimension=8,
    )
    assert loaded is not None
    assert loaded.dtype == np.float16
    np.testing.assert_allclose(np.linalg.norm(loaded.astype(np.float32)), 1.0, atol=1e-3)

    basis = np.eye(4, 2, dtype=np.float32)
    saved = manager.save_basis(basis_input_signature="e" * 64, array=basis)
    loaded_basis = manager.load_basis(
        basis_input_signature="e" * 64,
        expected_shape=(4, 2),
    )
    assert loaded_basis is not None
    assert loaded_basis.sha256 == saved.sha256
    np.testing.assert_array_equal(loaded_basis.array, basis)


def test_detection_cache_rejects_entries_incompatible_with_runtime_contract(tmp_path) -> None:
    manager = CacheManager(LocalArtifactStore(tmp_path))
    signature = "a" * 64
    image_hash = "b" * 64

    manager.save_detections(
        detector_signature=signature,
        image_sha256=image_hash,
        width=100,
        height=80,
        detections=[Detection(BBoxXYXY(1, 2, 10, 12), 0.4, 0)],
    )
    assert manager.load_detections(
        detector_signature=signature,
        image_sha256=image_hash,
        width=100,
        height=80,
        threshold=0.5,
        max_detections=20,
    ) is None

    manager.save_detections(
        detector_signature=signature,
        image_sha256=image_hash,
        width=100,
        height=80,
        detections=[Detection(BBoxXYXY(1, 2, 101, 12), 0.9, 0)],
    )
    assert manager.load_detections(
        detector_signature=signature,
        image_sha256=image_hash,
        width=100,
        height=80,
        threshold=0.5,
        max_detections=20,
    ) is None


def test_basis_cache_rejects_non_orthonormal_columns(tmp_path) -> None:
    manager = CacheManager(LocalArtifactStore(tmp_path))
    invalid = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(ArtifactError, match="orthonormal"):
        manager.save_basis(basis_input_signature="e" * 64, array=invalid)
