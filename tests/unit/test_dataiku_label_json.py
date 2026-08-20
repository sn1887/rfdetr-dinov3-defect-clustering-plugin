from __future__ import annotations

import json

from defect_curation_core.labeling.dataiku_object_detection import build_review_rows
from defect_curation_core.types import BBoxXYXY, Detection, ImageRecord, InstanceRecord


def test_success_row_contains_detection_columns_and_cluster_context() -> None:
    image = ImageRecord(image_path="nested/a.png", width=100, height=80, status="OK")
    image.detections = [Detection(BBoxXYXY(10, 20, 40, 50), 0.9, 0)]
    instance = InstanceRecord(
        instance_id="di_0123456789abcdef",
        instance_key="a" * 64,
        image_path=image.image_path,
        image_sha256="b" * 64,
        bbox=BBoxXYXY(10, 20, 40, 50),
        bbox_xywh=(10, 20, 30, 30),
        detector_score=0.9,
        embedding_status="COMPUTED",
        cluster_id=3,
        cosine_to_centroid=0.8,
        cluster_rank=1,
        cluster_size=4,
    )
    row = build_review_rows(
        run_id="00000000-0000-0000-0000-000000000000",
        images=[image],
        instances=[instance],
        review_order={image.image_path: 1},
    )[0]
    assert json.loads(row["detection_bbox"]) == [
        {"bbox": [10, 20, 30, 30], "category": "defect"}
    ]
    assert json.loads(row["detection_score"]) == [0.9]
    assert json.loads(row["detection_bbox_cluster"]) == [
        {"bbox": [10, 20, 30, 30], "category": "3"}
    ]
    context = json.loads(row["instances_json"])
    assert context[0]["cluster_id"] == 3
    assert row["primary_cluster_id"] == 3


def test_error_row_is_review_inert_even_when_detector_instance_exists() -> None:
    image = ImageRecord(
        image_path="a.png",
        width=100,
        height=80,
        status="ERROR",
        error_code="IMAGE_CHANGED_DURING_RUN",
        error_message="changed",
    )
    instance = InstanceRecord(
        instance_id="di_0123456789abcdef",
        instance_key="a" * 64,
        image_path=image.image_path,
        image_sha256="b" * 64,
        bbox=BBoxXYXY(10, 20, 40, 50),
        bbox_xywh=(10, 20, 30, 30),
        detector_score=0.9,
    )
    row = build_review_rows(run_id="run", images=[image], instances=[instance], review_order={})[0]
    assert row["num_defects"] == 0
    assert row["detection_bbox"] == "[]"
    assert row["detection_score"] == "[]"
    assert row["detection_bbox_cluster"] == "[]"
    assert row["instances_json"] == "[]"


def test_rows_are_materialized_in_review_then_no_detection_then_error_order() -> None:
    images = [
        ImageRecord(image_path="error.png", status="ERROR"),
        ImageRecord(image_path="normal.png", status="NO_DETECTION"),
        ImageRecord(image_path="second.png", status="OK"),
        ImageRecord(image_path="first.png", status="OK"),
    ]
    rows = build_review_rows(
        run_id="run",
        images=images,
        instances=[],
        review_order={},
    )
    assert [row["image_path"] for row in rows] == [
        "normal.png",
        "error.png",
        "first.png",
        "second.png",
    ]


def test_primary_cluster_tie_breaks_on_smallest_instance_id() -> None:
    image = ImageRecord(image_path="a.png", status="OK")
    common = dict(
        image_path="a.png",
        image_sha256="b" * 64,
        bbox=BBoxXYXY(0, 0, 10, 10),
        bbox_xywh=(0, 0, 10, 10),
        detector_score=0.9,
        embedding_status="CACHED",
    )
    instances = [
        InstanceRecord(instance_id="di_b", instance_key="1" * 64, cluster_id=7, **common),
        InstanceRecord(instance_id="di_a", instance_key="2" * 64, cluster_id=3, **common),
    ]
    row = build_review_rows(
        run_id="run",
        images=[image],
        instances=instances,
        review_order={"a.png": 1},
    )[0]
    assert row["primary_cluster_id"] == 3
