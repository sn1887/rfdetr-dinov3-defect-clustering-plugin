"""Deterministic within-cluster ranks and cross-cluster review interleaving."""

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
    """Round-robin representative instances, de-duplicating image paths."""

    grouped: dict[int, list[InstanceRecord]] = defaultdict(list)
    for instance in instances:
        if instance.cluster_id is not None:
            grouped[int(instance.cluster_id)].append(instance)

    for members in grouped.values():
        members.sort(
            key=lambda item: (
                -float(item.cosine_to_centroid if item.cosine_to_centroid is not None else -np.inf),
                -float(item.detector_score),
                item.instance_id,
            )
        )

    cluster_order = sorted(grouped, key=lambda cluster_id: (len(grouped[cluster_id]), cluster_id))
    cursors = {cluster_id: 0 for cluster_id in cluster_order}
    ordered_images: dict[str, int] = {}

    while True:
        progressed = False
        for cluster_id in cluster_order:
            cursor = cursors[cluster_id]
            members = grouped[cluster_id]
            if cursor >= len(members):
                continue
            progressed = True
            instance = members[cursor]
            cursors[cluster_id] = cursor + 1
            if instance.image_path not in ordered_images:
                ordered_images[instance.image_path] = len(ordered_images) + 1
        if not progressed:
            break
    return ordered_images
