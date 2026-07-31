"""Dataiku object-detection JSON and image-level review rows."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence

from defect_curation_core.types import ImageRecord, InstanceRecord

PLACEHOLDER_CATEGORY = "UNVALIDATED_DEFECT"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_review_rows(
    *,
    run_id: str,
    images: Sequence[ImageRecord],
    instances: Sequence[InstanceRecord],
    review_order: dict[str, int],
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
        image_instances = sorted(
            source_instances,
            key=lambda item: (-item.detector_score, item.instance_id),
        )
        prelabels = [
            {
                "bbox": list(instance.bbox_xywh),
                "category": PLACEHOLDER_CATEGORY,
            }
            for instance in image_instances
        ]
        context = [
            {
                "instance_id": instance.instance_id,
                "bbox_xywh": list(instance.bbox_xywh),
                "detector_score": round(float(instance.detector_score), 8),
                "cluster_id": instance.cluster_id,
                "cosine_to_centroid": (
                    None if instance.cosine_to_centroid is None else round(float(instance.cosine_to_centroid), 8)
                ),
                "cluster_rank": instance.cluster_rank,
                "cluster_size": instance.cluster_size,
                "embedding_status": instance.embedding_status,
                "warning_codes": list(instance.warning_codes),
            }
            for instance in image_instances
        ]
        clustered = [instance for instance in image_instances if instance.cluster_id is not None]
        primary = (
            min(clustered, key=lambda item: (-item.detector_score, item.instance_id))
            if clustered
            else None
        )
        rows.append(
            {
                "image_path": image.image_path,
                "image_status": image.status,
                "image_width": image.width,
                "image_height": image.height,
                "num_defects": len(image_instances),
                "primary_cluster_id": None if primary is None else primary.cluster_id,
                "review_order": review_order.get(image.image_path),
                "prelabels_json": _json(prelabels),
                "instances_json": _json(context),
                "error_code": image.error_code,
                "error_message": image.error_message,
                "run_id": run_id,
            }
        )
    # Materialize the representative cross-cluster order in the dataset itself.
    # Dataiku readers therefore encounter reviewable images first, followed by
    # no-detection records and finally error rows. The explicit review_order
    # column remains the authoritative ordering key downstream.
    rows.sort(
        key=lambda row: (
            0
            if row["review_order"] is not None
            else (1 if row["image_status"] == "NO_DETECTION" else 2),
            int(row["review_order"]) if row["review_order"] is not None else 0,
            str(row["image_path"]),
        )
    )
    return rows
