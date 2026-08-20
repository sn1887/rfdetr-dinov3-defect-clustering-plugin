"""Fixed DSS output schema for the image-level review dataset."""

from __future__ import annotations

from typing import Final

REVIEW_DATASET_COLUMNS: Final[tuple[str, ...]] = (
    "image_path",
    "image_status",
    "image_width",
    "image_height",
    "num_defects",
    "primary_cluster_id",
    "review_order",
    "detection_bbox",
    "detection_score",
    "detection_bbox_cluster",
    "instances_json",
    "error_code",
    "error_message",
    "run_id",
)

# Dataiku logical types. The writer deliberately creates this schema before rows
# are emitted so all-null integer columns remain numeric on scheduled builds.
REVIEW_DATASET_SCHEMA: Final[list[dict[str, str]]] = [
    {"name": "image_path", "type": "string"},
    {"name": "image_status", "type": "string"},
    {"name": "image_width", "type": "bigint"},
    {"name": "image_height", "type": "bigint"},
    {"name": "num_defects", "type": "bigint"},
    {"name": "primary_cluster_id", "type": "bigint"},
    {"name": "review_order", "type": "bigint"},
    {"name": "detection_bbox", "type": "string"},
    {"name": "detection_score", "type": "string"},
    {"name": "detection_bbox_cluster", "type": "string"},
    {"name": "instances_json", "type": "string"},
    {"name": "error_code", "type": "string"},
    {"name": "error_message", "type": "string"},
    {"name": "run_id", "type": "string"},
]
