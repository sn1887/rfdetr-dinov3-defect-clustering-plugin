from __future__ import annotations

from defect_curation_core.detection.postprocess import (
    build_instance_identity,
    postprocess_detections,
)
from defect_curation_core.types import BBoxXYXY, Detection


def test_postprocess_threshold_clip_sort_and_cap() -> None:
    detections = [
        Detection(BBoxXYXY(-2, -2, 8, 8), 0.8, 0),
        Detection(BBoxXYXY(5, 5, 20, 20), 0.9, 0),
        Detection(BBoxXYXY(0, 0, 1, 1), 0.1, 0),
    ]
    result = postprocess_detections(
        detections,
        image_width=10,
        image_height=10,
        threshold=0.5,
        max_detections=1,
    )
    assert len(result) == 1
    assert result[0].score == 0.9
    assert result[0].bbox == BBoxXYXY(5, 5, 10.0, 10.0)


def test_instance_identity_is_stable_and_occurrence_sensitive() -> None:
    kwargs = dict(
        image_path="nested/a.png",
        image_sha256="a" * 64,
        bbox=BBoxXYXY(1, 2, 3, 4),
        detector_signature="b" * 64,
    )
    first = build_instance_identity(**kwargs)
    assert first == build_instance_identity(**kwargs)
    assert first != build_instance_identity(**kwargs, occurrence_index=1)
    assert first[0].startswith("di_")


def test_instance_id_is_path_unique_but_cache_key_is_content_addressed() -> None:
    shared = dict(
        image_sha256="a" * 64,
        bbox=BBoxXYXY(1, 2, 3, 4),
        detector_signature="b" * 64,
    )
    first_id, first_key = build_instance_identity(image_path="a.png", **shared)
    second_id, second_key = build_instance_identity(image_path="copy/a.png", **shared)
    assert first_id != second_id
    assert first_key == second_key
