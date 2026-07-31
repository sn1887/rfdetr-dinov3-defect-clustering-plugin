from __future__ import annotations

import numpy as np
import pytest
import torch
from defect_curation_core.embeddings.pooling import (
    expand_box_to_minimum_weight,
    fractional_patch_weights,
    pool_fractional_box_features,
)
from defect_curation_core.types import BBoxXYXY


def test_fractional_weights_exact_patch() -> None:
    weights = fractional_patch_weights(
        BBoxXYXY(16, 16, 32, 32),
        grid_height=4,
        grid_width=4,
        canvas_height=64,
        canvas_width=64,
    )
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[1, 1] = 1.0
    np.testing.assert_allclose(weights, expected)


def test_thin_box_expands_to_minimum_effective_weight() -> None:
    used, weights, expanded = expand_box_to_minimum_weight(
        BBoxXYXY(10, 10, 11, 11),
        minimum_weight=4.0,
        grid_height=32,
        grid_width=32,
        canvas_height=512,
        canvas_width=512,
    )
    assert expanded is True
    assert float(weights.sum()) >= 4.0 - 1e-5
    assert used.width > 1 and used.height > 1


def test_minimum_weight_expansion_stays_inside_letterbox_content() -> None:
    content = BBoxXYXY(64, 128, 448, 384)
    used, weights, expanded = expand_box_to_minimum_weight(
        BBoxXYXY(64, 200, 65, 201),
        minimum_weight=4.0,
        grid_height=32,
        grid_width=32,
        canvas_height=512,
        canvas_width=512,
        expansion_bounds=content,
    )
    assert expanded is True
    assert float(weights.sum()) >= 4.0 - 1e-5
    assert used.xmin >= content.xmin
    assert used.ymin >= content.ymin
    assert used.xmax <= content.xmax
    assert used.ymax <= content.ymax


def test_pooling_returns_unit_vector_and_uses_box() -> None:
    features = torch.zeros((3, 4, 4), dtype=torch.float32)
    features[0] = 1.0
    features[1, 2:, 2:] = 5.0
    result = pool_fractional_box_features(
        features,
        BBoxXYXY(0, 0, 32, 32),
        canvas_size=64,
        minimum_effective_patch_weight=1.0,
    )
    assert torch.isclose(torch.linalg.norm(result.vector), torch.tensor(1.0), atol=1e-6)
    assert result.vector[0] > 0.99


def test_pooling_projects_positional_basis() -> None:
    features = torch.zeros((3, 2, 2), dtype=torch.float32)
    features[0] = 1.0
    features[1] = 1.0
    basis = torch.tensor([[1.0], [0.0], [0.0]])
    result = pool_fractional_box_features(
        features,
        BBoxXYXY(0, 0, 32, 32),
        canvas_size=32,
        minimum_effective_patch_weight=1.0,
        positional_basis=basis,
    )
    assert abs(float(result.vector[0])) < 1e-6
    assert float(result.vector[1]) > 0.99


def test_pooling_rejects_descriptor_fully_removed_by_projection() -> None:
    features = torch.zeros((3, 2, 2), dtype=torch.float32)
    features[0] = 1.0
    basis = torch.tensor([[1.0], [0.0], [0.0]])
    with pytest.raises(ValueError, match="zero-norm descriptor"):
        pool_fractional_box_features(
            features,
            BBoxXYXY(0, 0, 32, 32),
            canvas_size=32,
            minimum_effective_patch_weight=1.0,
            positional_basis=basis,
        )
