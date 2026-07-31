from __future__ import annotations

from defect_curation_core.clustering.review_order import assign_cluster_ranks, build_review_order
from defect_curation_core.types import BBoxXYXY, InstanceRecord


def instance(identifier: str, image: str, cluster: int, similarity: float, score: float = 0.9):
    return InstanceRecord(
        instance_id=identifier,
        instance_key=identifier * 4,
        image_path=image,
        image_sha256="a" * 64,
        bbox=BBoxXYXY(0, 0, 10, 10),
        bbox_xywh=(0, 0, 10, 10),
        detector_score=score,
        cluster_id=cluster,
        cosine_to_centroid=similarity,
    )


def test_rank_and_small_cluster_first_interleave_with_image_deduplication() -> None:
    values = [
        instance("i1", "same.png", 0, 0.99),
        instance("i2", "b.png", 0, 0.90),
        instance("i3", "same.png", 1, 0.98),
        instance("i4", "c.png", 2, 0.95),
        instance("i5", "d.png", 2, 0.80),
        instance("i6", "e.png", 2, 0.70),
    ]
    assign_cluster_ranks(values)
    assert values[0].cluster_rank == 1
    assert values[0].cluster_size == 2
    order = build_review_order(values)
    assert order["same.png"] == 1
    assert order["c.png"] == 2
    assert len(order) == 5
    assert sorted(order.values()) == [1, 2, 3, 4, 5]
