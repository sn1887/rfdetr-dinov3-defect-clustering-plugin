"""Dataiku object-detection JSON and image-level review rows."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence

from defect_curation_core.types import ImageRecord, InstanceRecord

DETECTION_CATEGORY = "defect"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_review_rows(
    *,
    run_id: str,
    images: Sequence[ImageRecord],
    instances: Sequence[InstanceRecord],
    review_order: dict[str, int],
    embedding_granularity: str = "object",
) -> list[dict[str, object]]:
    by_image: dict[str, list[InstanceRecord]] = defaultdict(list)
    for instance in instances:
        by_image[instance.image_path].append(instance)

    rows: list[dict[str, object]] = []
    for image in images:
        # An ERROR row is intentionally review-inert. Detection/instance evidence
        # remains in the run bundle, but the Dataiku labeling dataset must not
        # offer boxes against bytes that failed or changed during this run.
        source_instances = [] if image.status == "ERROR" else by_image.get(image.image_path, [])
        image_detections = [] if image.status == "ERROR" else list(image.detections)
        image_instances = sorted(
            source_instances,
            key=lambda item: (-item.detector_score, item.instance_id),
        )
        detection_bbox = [
            {
                "bbox": list(detection.bbox.to_xywh_int()),
                "category": DETECTION_CATEGORY,
            }
            for detection in image_detections
        ]
        detection_score = [round(float(detection.score), 8) for detection in image_detections]
        clustered = [instance for instance in image_instances if instance.cluster_id is not None]
        primary = min(clustered, key=lambda item: (-item.detector_score, item.instance_id)) if clustered else None
        if embedding_granularity == "image":
            detection_bbox_cluster = (
                [
                    {
                        "bbox": list(detection.bbox.to_xywh_int()),
                        "category": str(primary.cluster_id),
                    }
                    for detection in image_detections
                ]
                if primary is not None
                else []
            )
            context = [
                {
                    "instance_id": instance.instance_id,
                    "embedding_granularity": instance.embedding_granularity,
                    "bbox_xywh": list(instance.bbox_xywh),
                    "num_detections": len(image_detections),
                    "max_detector_score": round(float(instance.detector_score), 8),
                    "cluster_id": instance.cluster_id,
                    "cosine_to_centroid": (
                        None
                        if instance.cosine_to_centroid is None
                        else round(float(instance.cosine_to_centroid), 8)
                    ),
                    "cluster_rank": instance.cluster_rank,
                    "cluster_size": instance.cluster_size,
                    "embedding_status": instance.embedding_status,
                    "warning_codes": list(instance.warning_codes),
                }
                for instance in image_instances
            ]
        else:
            detection_bbox_cluster = [
                {
                    "bbox": list(instance.bbox_xywh),
                    "category": str(instance.cluster_id),
                }
                for instance in image_instances
                if instance.cluster_id is not None
            ]
            context = [
                {
                    "instance_id": instance.instance_id,
                    "embedding_granularity": instance.embedding_granularity,
                    "bbox_xywh": list(instance.bbox_xywh),
                    "detector_score": round(float(instance.detector_score), 8),
                    "cluster_id": instance.cluster_id,
                    "cosine_to_centroid": (
                        None
                        if instance.cosine_to_centroid is None
                        else round(float(instance.cosine_to_centroid), 8)
                    ),
                    "cluster_rank": instance.cluster_rank,
                    "cluster_size": instance.cluster_size,
                    "embedding_status": instance.embedding_status,
                    "warning_codes": list(instance.warning_codes),
                }
                for instance in image_instances
            ]
        rows.append(
            {
                "image_path": image.image_path,
                "image_status": image.status,
                "image_width": image.width,
                "image_height": image.height,
                "num_defects": len(image_detections),
                "primary_cluster_id": None if primary is None else primary.cluster_id,
                "review_order": review_order.get(image.image_path),
                "detection_bbox": _json(detection_bbox),
                "detection_score": _json(detection_score),
                "detection_bbox_cluster": _json(detection_bbox_cluster),
                "instances_json": _json(context),
                "error_code": image.error_code,
                "error_message": image.error_message,
                "run_id": run_id,
            }
        )
    # Materialize cluster-local review order in the dataset itself. Clustered
    # rows appear first, followed by unclustered no-detection records and errors.
    rows.sort(
        key=lambda row: (
            0 if row["primary_cluster_id"] is not None else (1 if row["image_status"] == "NO_DETECTION" else 2),
            int(row["primary_cluster_id"]) if row["primary_cluster_id"] is not None else 0,
            int(row["review_order"]) if row["review_order"] is not None else 0,
            str(row["image_path"]),
        )
    )
    return rows
