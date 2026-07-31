from __future__ import annotations

import torch
from defect_curation_core.embeddings.positional_debias import (
    compute_positional_basis,
    l2_normalize_patches,
    project_out_subspace,
)


def test_patch_normalization() -> None:
    features = torch.randn(8, 4, 4)
    normalized = l2_normalize_patches(features)
    norms = torch.linalg.norm(normalized, dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=0)


def test_basis_is_orthonormal_and_projection_removes_it() -> None:
    generator = torch.Generator().manual_seed(7)
    features = torch.randn((8, 5, 5), generator=generator)
    basis = compute_positional_basis(features, components=3)
    torch.testing.assert_close(basis.T @ basis, torch.eye(3), atol=1e-5, rtol=0)

    projected = project_out_subspace(features, basis)
    flat = projected.reshape(8, -1)
    residual = basis.T @ flat
    assert float(residual.abs().max()) < 1e-5
    norms = torch.linalg.norm(projected, dim=0)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)
