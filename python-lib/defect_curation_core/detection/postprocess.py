"""Deterministic RF-DETR output validation and instance identity."""

from __future__ import annotations

from collections.abc import Iterable

from defect_curation_core.hashing import hash_join
from defect_curation_core.types import BBoxXYXY, Detection


def postprocess_detections(
    detections: Iterable[Detection],
    *,
    image_width: int,
    image_height: int,
    threshold: float,
    max_detections: int,
    clip_to_image: bool = True,
    drop_degenerate: bool = True,
) -> list[Detection]:
    validated: list[Detection] = []
    for detection in detections:
        if detection.score < threshold:
            continue
        bbox = detection.bbox.clip(image_width, image_height) if clip_to_image else detection.bbox
        if drop_degenerate and (bbox.width < 1.0 or bbox.height < 1.0):
            continue
        validated.append(
            Detection(
                bbox=bbox,
                score=min(max(float(detection.score), 0.0), 1.0),
                raw_class_id=detection.raw_class_id,
            )
        )

    validated.sort(
        key=lambda item: (
            -item.score,
            item.bbox.xmin,
            item.bbox.ymin,
            item.bbox.xmax,
            item.bbox.ymax,
            -1 if item.raw_class_id is None else item.raw_class_id,
        )
    )
    return validated[:max_detections]


def quantized_bbox_key(bbox: BBoxXYXY) -> str:
    return ",".join(f"{coordinate:.4f}" for coordinate in bbox.as_list())


def build_instance_identity(
    *,
    image_path: str,
    image_sha256: str,
    bbox: BBoxXYXY,
    detector_signature: str,
    occurrence_index: int = 0,
) -> tuple[str, str]:
    """Return a path-unique review ID and a content-addressed cache key.

    Identical image bytes stored at two managed-folder paths are two review
    records and therefore need distinct ``instance_id`` values. Their crops
    are nevertheless byte/geometry equivalent, so ``instance_key`` remains
    content-addressed and can safely share the embedding cache.
    """

    instance_key = hash_join(
        (image_sha256, quantized_bbox_key(bbox), detector_signature, str(int(occurrence_index)))
    )
    instance_id_hash = hash_join((image_path, instance_key))
    return f"di_{instance_id_hash[:16]}", instance_key


def build_image_embedding_identity(
    *,
    image_path: str,
    image_sha256: str,
    embedding_scope: str = "full_image_v1",
) -> tuple[str, str]:
    """Return a path-unique clustering ID and content-addressed image embedding key."""

    instance_key = hash_join((image_sha256, embedding_scope))
    instance_id_hash = hash_join((image_path, instance_key))
    return f"di_{instance_id_hash[:16]}", instance_key
