"""INSID3-inspired positional-subspace estimation and orthogonal projection."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def _as_bchw(features: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if features.ndim == 3:
        return features.unsqueeze(0), True
    if features.ndim == 4:
        return features, False
    raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got {tuple(features.shape)}")


def l2_normalize_patches(features: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    tensor, squeezed = _as_bchw(features.float())
    normalized = functional.normalize(tensor, p=2, dim=1, eps=eps)
    return normalized[0] if squeezed else normalized


def compute_positional_basis(features: torch.Tensor, *, components: int) -> torch.Tensor:
    """Return the leading channel-space SVD basis from one dense feature map."""

    tensor, _ = _as_bchw(features.float())
    if tensor.shape[0] != 1:
        raise ValueError("Positional basis must be estimated from exactly one basis image")
    channels = tensor.shape[1]
    if not 1 <= components < channels:
        raise ValueError("components must be in [1, channels)")

    normalized = l2_normalize_patches(tensor)[0]
    matrix = normalized.reshape(channels, -1)
    centered = matrix - matrix.mean(dim=1, keepdim=True)
    u, _, _ = torch.linalg.svd(centered, full_matrices=False)
    basis = u[:, :components].contiguous()
    return basis.float()


def project_out_subspace(features: torch.Tensor, basis: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Project dense descriptors away from basis and re-normalize each patch."""

    tensor, squeezed = _as_bchw(features.float())
    batch, channels, height, width = tensor.shape
    if basis.ndim != 2 or basis.shape[0] != channels:
        raise ValueError(
            f"Basis shape {tuple(basis.shape)} is incompatible with {channels} feature channels"
        )
    normalized = functional.normalize(tensor, p=2, dim=1, eps=eps)
    flattened = normalized.reshape(batch, channels, height * width)
    basis = basis.to(device=flattened.device, dtype=flattened.dtype)
    projected = flattened - torch.einsum("ck,bkp->bcp", basis, torch.einsum("ck,bcp->bkp", basis, flattened))
    projected = functional.normalize(projected, p=2, dim=1, eps=eps).reshape(batch, channels, height, width)
    return projected[0] if squeezed else projected
