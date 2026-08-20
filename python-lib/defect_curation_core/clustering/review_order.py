"""Deterministic within-cluster ranks and cluster-local review order."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from defect_curation_core.types import InstanceRecord


def assign_cluster_ranks(instances: Sequence[InstanceRecord]) -> None:
    grouped: dict[int, list[InstanceRecord]] = defaultdict(list)
    for instance in instances:
        if instance.cluster_id is not None:
            grouped[int(instance.cluster_id)].append(instance)

    for _cluster_id, members in grouped.items():
        members.sort(
            key=lambda item: (
                -float(item.cosine_to_centroid if item.cosine_to_centroid is not None else -np.inf),
                -float(item.detector_score),
                item.instance_id,
            )
        )
        size = len(members)
        for rank, instance in enumerate(members, start=1):
            instance.cluster_rank = rank
            instance.cluster_size = size


def build_review_order(instances: Sequence[InstanceRecord]) -> dict[str, int]:
    """Rank each image within its primary cluster."""

    primary_by_image: dict[str, InstanceRecord] = {}
    for instance in instances:
        if instance.cluster_id is None:
            continue
        existing = primary_by_image.get(instance.image_path)
        if existing is None or (
            -float(instance.detector_score),
            instance.instance_id,
        ) < (
            -float(existing.detector_score),
            existing.instance_id,
        ):
            primary_by_image[instance.image_path] = instance

    grouped: dict[int, list[InstanceRecord]] = defaultdict(list)
    for instance in primary_by_image.values():
        grouped[int(instance.cluster_id)].append(instance)

    ordered_images: dict[str, int] = {}
    for members in grouped.values():
        members.sort(
            key=lambda item: (
                -float(item.cosine_to_centroid if item.cosine_to_centroid is not None else -np.inf),
                -float(item.detector_score),
                item.instance_id,
            )
        )
        for rank, instance in enumerate(members, start=1):
            ordered_images[instance.image_path] = rank
    return ordered_images
