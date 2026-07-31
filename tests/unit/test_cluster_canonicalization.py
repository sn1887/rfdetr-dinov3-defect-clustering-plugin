from __future__ import annotations

import numpy as np
from defect_curation_core.clustering.spherical_kmeans import fit_spherical_kmeans


def run(values: np.ndarray, ids: list[str]):
    return fit_spherical_kmeans(
        values,
        instance_ids=ids,
        k=2,
        niter=30,
        nredo=4,
        seed=42,
        backend="numpy_reference",
        max_points_per_centroid=256,
        assignment_chunk_size=2,
        max_empty_cluster_retries=2,
        canonicalize_ids=True,
    )


def test_spherical_kmeans_groups_and_canonicalizes() -> None:
    values = np.array(
        [
            [1.0, 0.02],
            [0.99, -0.01],
            [-1.0, 0.01],
            [-0.98, -0.02],
        ],
        dtype=np.float32,
    )
    ids = ["b", "a", "d", "c"]
    result = run(values, ids)
    assert result.assignments[0] == result.assignments[1]
    assert result.assignments[2] == result.assignments[3]
    assert result.assignments[0] != result.assignments[2]
    np.testing.assert_allclose(np.linalg.norm(result.centroids, axis=1), 1.0, atol=1e-6)
    assert np.all(result.similarities <= 1.000001)


def test_canonical_ids_stable_under_input_permutation() -> None:
    values = np.array([[1, 0], [0.9, 0.1], [-1, 0], [-0.9, -0.1]], dtype=np.float32)
    ids = ["a", "b", "c", "d"]
    first = run(values, ids)
    permutation = np.array([2, 0, 3, 1])
    second = run(values[permutation], [ids[index] for index in permutation])
    first_by_id = {identifier: int(first.assignments[index]) for index, identifier in enumerate(ids)}
    second_by_id = {
        ids[index]: int(second.assignments[position])
        for position, index in enumerate(permutation)
    }
    assert first_by_id == second_by_id
