"""Spherical k-means in cosine geometry with deterministic canonical IDs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from defect_curation_core.errors import DependencyError, FatalPipelineError
from defect_curation_core.types import ClusterResult


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise FatalPipelineError(f"Embeddings must be a non-empty matrix, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise FatalPipelineError("Embeddings contain non-finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise FatalPipelineError("Embeddings contain zero-norm rows")
    return np.ascontiguousarray(array / norms, dtype=np.float32)


def _assign_chunks(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    count = embeddings.shape[0]
    assignments = np.empty(count, dtype=np.int32)
    similarities = np.empty(count, dtype=np.float32)
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        chunk = np.asarray(embeddings[start:stop], dtype=np.float32)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        chunk = chunk / np.maximum(norms, 1e-12)
        scores = chunk @ centroids.T
        selected = np.argmax(scores, axis=1)
        assignments[start:stop] = selected.astype(np.int32, copy=False)
        selected_scores = scores[np.arange(stop - start), selected]
        similarities[start:stop] = np.clip(selected_scores, -1.0, 1.0).astype(np.float32, copy=False)
    return assignments, similarities


def assign_to_centroids(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    *,
    assignment_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign L2-normalized embeddings to existing spherical k-means centroids."""

    values = np.asarray(embeddings)
    if values.ndim != 2:
        raise FatalPipelineError(f"Expected a 2D embedding matrix, got shape {values.shape}")
    centroid_values = _normalize_rows(np.asarray(centroids, dtype=np.float32))
    if values.shape[1] != centroid_values.shape[1]:
        raise FatalPipelineError(
            f"Embedding dimension {values.shape[1]} does not match centroid dimension {centroid_values.shape[1]}"
        )
    if int(assignment_chunk_size) < 1:
        raise FatalPipelineError("assignment_chunk_size must be positive")
    return _assign_chunks(values, centroid_values, chunk_size=int(assignment_chunk_size))


def _training_sample(
    embeddings: np.ndarray,
    *,
    k: int,
    max_points_per_centroid: int,
    seed: int,
) -> np.ndarray:
    max_points = min(embeddings.shape[0], k * max_points_per_centroid)
    if max_points == embeddings.shape[0]:
        sample = np.asarray(embeddings, dtype=np.float32)
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(embeddings.shape[0], size=max_points, replace=False))
        sample = np.asarray(embeddings[indices], dtype=np.float32)
    return _normalize_rows(sample)


def _fit_faiss(
    training: np.ndarray,
    *,
    k: int,
    niter: int,
    nredo: int,
    seed: int,
) -> np.ndarray:
    try:
        import faiss
    except ImportError as exc:
        raise DependencyError(
            "faiss-cpu is required for production spherical k-means; the numpy_reference backend is test-only"
        ) from exc

    kmeans = faiss.Kmeans(
        d=training.shape[1],
        k=k,
        niter=niter,
        nredo=nredo,
        seed=seed,
        spherical=True,
        gpu=False,
        verbose=False,
    )
    kmeans.train(np.ascontiguousarray(training, dtype=np.float32))
    centroids = np.asarray(kmeans.centroids, dtype=np.float32).reshape(k, training.shape[1])
    return _normalize_rows(centroids)


def _kmeans_plus_plus(values: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    count, dimension = values.shape
    centroids = np.empty((k, dimension), dtype=np.float32)
    first = int(rng.integers(0, count))
    centroids[0] = values[first]
    closest_distance = 1.0 - values @ centroids[0]
    closest_distance = np.maximum(closest_distance, 0.0)
    for index in range(1, k):
        total = float(closest_distance.sum())
        if total <= 1e-12:
            candidate = int(rng.integers(0, count))
        else:
            candidate = int(rng.choice(count, p=closest_distance / total))
        centroids[index] = values[candidate]
        distance = np.maximum(1.0 - values @ centroids[index], 0.0)
        closest_distance = np.minimum(closest_distance, distance)
    return _normalize_rows(centroids)


def _fit_numpy_reference(
    training: np.ndarray,
    *,
    k: int,
    niter: int,
    nredo: int,
    seed: int,
) -> np.ndarray:
    """Small deterministic reference implementation for unit/integration tests."""

    best_centroids: np.ndarray | None = None
    best_objective = -np.inf
    for restart in range(nredo):
        rng = np.random.default_rng(seed + restart)
        centroids = _kmeans_plus_plus(training, k, rng)
        for _ in range(niter):
            scores = training @ centroids.T
            assignments = np.argmax(scores, axis=1)
            updated = np.zeros_like(centroids)
            counts = np.bincount(assignments, minlength=k)
            for cluster in range(k):
                if counts[cluster] == 0:
                    farthest = int(np.argmin(np.max(scores, axis=1)))
                    updated[cluster] = training[farthest]
                else:
                    updated[cluster] = training[assignments == cluster].mean(axis=0)
            updated = _normalize_rows(updated)
            if np.allclose(updated, centroids, atol=1e-6, rtol=0.0):
                centroids = updated
                break
            centroids = updated
        objective = float(np.max(training @ centroids.T, axis=1).sum())
        if objective > best_objective:
            best_objective = objective
            best_centroids = centroids.copy()
    assert best_centroids is not None
    return best_centroids


def _canonicalize(
    assignments: np.ndarray,
    similarities: np.ndarray,
    centroids: np.ndarray,
    instance_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    k = centroids.shape[0]
    representatives: list[tuple[str, int]] = []
    ids = np.asarray(instance_ids, dtype=object)
    for cluster in range(k):
        members = np.flatnonzero(assignments == cluster)
        if members.size == 0:
            raise FatalPipelineError(f"K-means produced empty cluster {cluster}")
        best_similarity = similarities[members].max()
        tied = members[np.isclose(similarities[members], best_similarity, atol=1e-8, rtol=0.0)]
        representative = min(str(ids[index]) for index in tied)
        representatives.append((representative, cluster))

    ordered_old_ids = [old_id for _, old_id in sorted(representatives)]
    old_to_new = np.empty(k, dtype=np.int32)
    for new_id, old_id in enumerate(ordered_old_ids):
        old_to_new[old_id] = new_id
    remapped = old_to_new[assignments]
    reordered_centroids = centroids[np.asarray(ordered_old_ids, dtype=np.int64)]
    return remapped.astype(np.int32, copy=False), reordered_centroids.astype(np.float32, copy=False)


def fit_spherical_kmeans(
    embeddings: np.ndarray,
    *,
    instance_ids: Sequence[str],
    k: int,
    niter: int,
    nredo: int,
    seed: int,
    backend: str,
    max_points_per_centroid: int,
    assignment_chunk_size: int,
    max_empty_cluster_retries: int,
    canonicalize_ids: bool = True,
) -> ClusterResult:
    values = np.asarray(embeddings)
    if values.ndim != 2:
        raise FatalPipelineError(f"Expected a 2D embedding matrix, got shape {values.shape}")
    count = values.shape[0]
    if len(instance_ids) != count:
        raise FatalPipelineError("instance_ids length does not match embedding count")
    if not 2 <= k <= count:
        raise FatalPipelineError(f"cluster_count must be in [2, {count}], got {k}")

    last_empty: list[int] = []
    for retry in range(max_empty_cluster_retries + 1):
        run_seed = seed + retry
        training = _training_sample(
            values,
            k=k,
            max_points_per_centroid=max_points_per_centroid,
            seed=run_seed,
        )
        if backend == "faiss_cpu":
            centroids = _fit_faiss(training, k=k, niter=niter, nredo=nredo, seed=run_seed)
        elif backend == "numpy_reference":
            centroids = _fit_numpy_reference(training, k=k, niter=niter, nredo=nredo, seed=run_seed)
        else:
            raise FatalPipelineError(f"Unsupported clustering backend: {backend}")

        assignments, similarities = _assign_chunks(
            values,
            centroids,
            chunk_size=assignment_chunk_size,
        )
        counts = np.bincount(assignments, minlength=k)
        last_empty = np.flatnonzero(counts == 0).tolist()
        if not last_empty:
            if canonicalize_ids:
                assignments, centroids = _canonicalize(
                    assignments,
                    similarities,
                    centroids,
                    instance_ids,
                )
            return ClusterResult(
                assignments=assignments,
                similarities=similarities,
                centroids=centroids,
                seed_used=run_seed,
            )

    raise FatalPipelineError(
        f"Spherical k-means still produced empty clusters {last_empty} after {max_empty_cluster_retries + 1} attempts"
    )
