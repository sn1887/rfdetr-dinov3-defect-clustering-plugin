"""Fractional detector-box pooling over DINOv3 patch tokens."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from defect_curation_core.embeddings.positional_debias import project_out_subspace
from defect_curation_core.types import BBoxXYXY


@dataclass(frozen=True, slots=True)
class PoolingResult:
    vector: torch.Tensor
    effective_patch_weight: float
    box_used: BBoxXYXY
    expanded_for_minimum: bool


def fractional_patch_weights(
    box: BBoxXYXY,
    *,
    grid_height: int,
    grid_width: int,
    canvas_height: int,
    canvas_width: int,
) -> np.ndarray:
    if grid_height <= 0 or grid_width <= 0 or canvas_height <= 0 or canvas_width <= 0:
        raise ValueError("Grid and canvas dimensions must be positive")
    clipped = box.clip(canvas_width, canvas_height)
    patch_width = canvas_width / grid_width
    patch_height = canvas_height / grid_height

    x0 = np.arange(grid_width, dtype=np.float64) * patch_width
    x1 = x0 + patch_width
    y0 = np.arange(grid_height, dtype=np.float64) * patch_height
    y1 = y0 + patch_height

    overlap_x = np.maximum(0.0, np.minimum(x1, clipped.xmax) - np.maximum(x0, clipped.xmin))
    overlap_y = np.maximum(0.0, np.minimum(y1, clipped.ymax) - np.maximum(y0, clipped.ymin))
    weights = np.outer(overlap_y, overlap_x) / (patch_width * patch_height)
    return weights.astype(np.float32, copy=False)


def _clip_to_bounds(box: BBoxXYXY, bounds: BBoxXYXY) -> BBoxXYXY:
    return BBoxXYXY(
        max(bounds.xmin, min(box.xmin, bounds.xmax)),
        max(bounds.ymin, min(box.ymin, bounds.ymax)),
        max(bounds.xmin, min(box.xmax, bounds.xmax)),
        max(bounds.ymin, min(box.ymax, bounds.ymax)),
    )


def _scaled_box(box: BBoxXYXY, scale: float, bounds: BBoxXYXY) -> BBoxXYXY:
    center_x = 0.5 * (box.xmin + box.xmax)
    center_y = 0.5 * (box.ymin + box.ymax)
    half_width = 0.5 * max(box.width, 1e-6) * scale
    half_height = 0.5 * max(box.height, 1e-6) * scale
    return _clip_to_bounds(
        BBoxXYXY(
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        bounds,
    )


def expand_box_to_minimum_weight(
    box: BBoxXYXY,
    *,
    minimum_weight: float,
    grid_height: int,
    grid_width: int,
    canvas_height: int,
    canvas_width: int,
    expansion_bounds: BBoxXYXY | None = None,
) -> tuple[BBoxXYXY, np.ndarray, bool]:
    bounds = (
        BBoxXYXY(0.0, 0.0, float(canvas_width), float(canvas_height))
        if expansion_bounds is None
        else _clip_to_bounds(
            expansion_bounds,
            BBoxXYXY(0.0, 0.0, float(canvas_width), float(canvas_height)),
        )
    )
    if bounds.width <= 0.0 or bounds.height <= 0.0:
        raise ValueError("Expansion bounds must cover a positive canvas region")
    box = _clip_to_bounds(box, bounds)
    initial = fractional_patch_weights(
        box,
        grid_height=grid_height,
        grid_width=grid_width,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
    )
    if float(initial.sum()) + 1e-6 >= minimum_weight:
        return box, initial, False

    full = bounds
    full_weights = fractional_patch_weights(
        full,
        grid_height=grid_height,
        grid_width=grid_width,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
    )
    if float(full_weights.sum()) + 1e-6 < minimum_weight:
        raise ValueError("Minimum effective patch weight exceeds the available patch grid")

    low = 1.0
    high = 2.0
    high_box = _scaled_box(box, high, bounds)
    high_weights = fractional_patch_weights(
        high_box,
        grid_height=grid_height,
        grid_width=grid_width,
        canvas_height=canvas_height,
        canvas_width=canvas_width,
    )
    while float(high_weights.sum()) + 1e-6 < minimum_weight and high < 1_000_000.0:
        high *= 2.0
        high_box = _scaled_box(box, high, bounds)
        high_weights = fractional_patch_weights(
            high_box,
            grid_height=grid_height,
            grid_width=grid_width,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
        )
        if high_box == full:
            break

    best_box = high_box
    best_weights = high_weights
    for _ in range(40):
        midpoint = 0.5 * (low + high)
        candidate_box = _scaled_box(box, midpoint, bounds)
        candidate_weights = fractional_patch_weights(
            candidate_box,
            grid_height=grid_height,
            grid_width=grid_width,
            canvas_height=canvas_height,
            canvas_width=canvas_width,
        )
        if float(candidate_weights.sum()) + 1e-6 >= minimum_weight:
            high = midpoint
            best_box = candidate_box
            best_weights = candidate_weights
        else:
            low = midpoint
    return best_box, best_weights, True


def pool_fractional_box_features(
    features: torch.Tensor,
    box_in_canvas: BBoxXYXY,
    *,
    canvas_size: int,
    minimum_effective_patch_weight: float,
    positional_basis: torch.Tensor | None = None,
    expansion_bounds: BBoxXYXY | None = None,
) -> PoolingResult:
    if features.ndim != 3:
        raise ValueError(f"Expected [C,H,W] patch map, got {tuple(features.shape)}")
    channels, grid_height, grid_width = features.shape
    if channels <= 0:
        raise ValueError("Feature map has no channels")

    box_used, weights_np, expanded = expand_box_to_minimum_weight(
        box_in_canvas,
        minimum_weight=minimum_effective_patch_weight,
        grid_height=grid_height,
        grid_width=grid_width,
        canvas_height=canvas_size,
        canvas_width=canvas_size,
        expansion_bounds=expansion_bounds,
    )
    tensor = features.float()
    if positional_basis is None:
        tensor = functional.normalize(tensor, p=2, dim=0, eps=1e-12)
    else:
        tensor = project_out_subspace(tensor, positional_basis)

    weights = torch.from_numpy(weights_np).to(device=tensor.device, dtype=tensor.dtype)
    denominator = weights.sum()
    if not torch.isfinite(denominator) or float(denominator.item()) <= 0.0:
        raise ValueError("Fractional pooling produced zero or non-finite total weight")
    pooled = (tensor * weights.unsqueeze(0)).sum(dim=(1, 2)) / denominator
    pooled_norm = torch.linalg.vector_norm(pooled)
    if not torch.isfinite(pooled_norm) or float(pooled_norm.item()) <= 1e-12:
        raise ValueError("Fractional pooling produced a zero-norm descriptor")
    pooled = pooled / pooled_norm
    if not torch.isfinite(pooled).all():
        raise ValueError("Pooled embedding contains non-finite values")
    return PoolingResult(
        vector=pooled.float().cpu(),
        effective_patch_weight=float(denominator.item()),
        box_used=box_used,
        expanded_for_minimum=expanded,
    )
