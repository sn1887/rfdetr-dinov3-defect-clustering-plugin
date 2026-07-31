"""Frozen, offline timm DINOv3 ViT-B/16 dense-feature adapter."""

from __future__ import annotations

import contextlib
import importlib.metadata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from defect_curation_core.errors import DependencyError, ModelLoadError

DINO_LOADER_IMPLEMENTATION_VERSION = "timm_safetensors_forward_intermediates_v1"
DINO_MODEL_ID = "timm/vit_base_patch16_dinov3.lvd1689m"
DINO_TIMM_VERSION = "1.0.20"


class DinoV3Adapter:
    """Load a local safetensors artifact into timm without Hub access."""

    def __init__(
        self,
        *,
        artifact_path: str,
        model_id: str,
        timm_version: str,
        device: str,
        input_size: int,
        patch_size: int,
        embedding_dim: int,
        expected_prefix_tokens: int,
        inference_precision: str,
    ) -> None:
        weights = self.resolve_artifact(artifact_path)
        if model_id != DINO_MODEL_ID or patch_size != 16 or embedding_dim != 768:
            raise ModelLoadError(f"The qualified adapter supports only {DINO_MODEL_ID}")
        try:
            installed_timm = importlib.metadata.version("timm")
        except importlib.metadata.PackageNotFoundError as exc:
            raise DependencyError("timm is not installed in the recipe code environment") from exc
        if timm_version != DINO_TIMM_VERSION or installed_timm != timm_version:
            raise ModelLoadError(
                f"Installed timm {installed_timm} does not match the qualified version {timm_version}"
            )

        try:
            import timm
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise DependencyError("PyTorch, timm, and safetensors are required for DINOv3") from exc

        if device == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError("CUDA was required but torch.cuda.is_available() is false")

        architecture = model_id.removeprefix("timm/")
        try:
            # pretrained=False is the security boundary: timm builds only the local
            # architecture and is never given a Hub URI or permission to fetch weights.
            model = timm.create_model(architecture, pretrained=False, num_classes=0)
            state = load_file(str(weights), device="cpu")
            self._validate_state_dict(state, torch=torch)
            incompatible = model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise ModelLoadError(
                    "DINOv3 artifact does not exactly match the qualified timm architecture"
                )
            model.eval()
            model.requires_grad_(False)
            model.to(device)
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Could not load the approved local timm DINOv3 artifact: {exc}") from exc

        if not hasattr(model, "forward_intermediates"):
            raise ModelLoadError("Loaded timm DINOv3 model has no forward_intermediates() feature API")
        declared_prefix = getattr(model, "num_prefix_tokens", None)
        if declared_prefix != int(expected_prefix_tokens):
            raise ModelLoadError(
                f"Expected {expected_prefix_tokens} DINOv3 prefix tokens, model declares {declared_prefix}"
            )

        self._torch = torch
        self._model = model
        self.device = device
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.embedding_dim = int(embedding_dim)
        self.expected_prefix_tokens = int(expected_prefix_tokens)
        self.inference_precision = inference_precision
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    @staticmethod
    def resolve_artifact(path: str | Path) -> Path:
        location = Path(path)
        if location.is_file():
            candidate = location
        elif location.is_dir():
            candidates = sorted(location.glob("*.safetensors"))
            if len(candidates) != 1:
                raise ModelLoadError(
                    f"Offline DINOv3 snapshot must contain exactly one top-level .safetensors file: {location}"
                )
            candidate = candidates[0]
        else:
            raise ModelLoadError(f"DINOv3 local artifact does not exist: {location}")
        if candidate.suffix != ".safetensors":
            raise ModelLoadError("DINOv3 runtime artifacts must use the safetensors format")
        return candidate

    @staticmethod
    def _validate_state_dict(state: Mapping[str, Any], *, torch: Any) -> None:
        if not state or not all(isinstance(key, str) for key in state):
            raise ModelLoadError("DINOv3 safetensors artifact is not a non-empty state dictionary")
        if not all(torch.is_tensor(value) for value in state.values()):
            raise ModelLoadError("DINOv3 state dictionary contains a non-tensor value")

    def preprocess(self, image: Image.Image):
        if image.mode != "RGB" or image.size != (self.input_size, self.input_size):
            raise ValueError(
                f"DINOv3 adapter expects RGB {self.input_size}x{self.input_size} crops, got {image.mode} {image.size}"
            )
        array = np.array(image, dtype=np.float32, copy=True) / 255.0
        tensor = self._torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - self._mean) / self._std

    def _autocast_context(self):
        torch = self._torch
        if self.device == "cuda" and self.inference_precision == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise ModelLoadError("Configured bf16 inference but the CUDA device does not support bfloat16")
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def extract_patch_maps(self, images: Sequence[Image.Image]):
        if not images:
            return self._torch.empty((0, self.embedding_dim, 0, 0), dtype=self._torch.float32)
        torch = self._torch
        batch = torch.stack([self.preprocess(image) for image in images], dim=0).to(
            self.device, non_blocking=True
        )
        with torch.inference_mode(), self._autocast_context():
            features = self._model.forward_intermediates(
                batch,
                indices=(-1,),
                return_prefix_tokens=True,
                norm=True,
                stop_early=True,
                output_fmt="NLC",
                intermediates_only=True,
            )
        if not isinstance(features, (list, tuple)) or len(features) != 1:
            raise ModelLoadError("timm DINOv3 final-layer feature API returned an unexpected structure")
        item = features[0]
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ModelLoadError("timm DINOv3 did not return separate patch and prefix tokens")
        patches, prefixes = item
        batch_size = len(images)
        expected_patches = (self.input_size // self.patch_size) ** 2
        if tuple(prefixes.shape) != (batch_size, self.expected_prefix_tokens, self.embedding_dim):
            raise ModelLoadError(
                "Unexpected DINOv3 prefix-token shape: "
                f"expected {(batch_size, self.expected_prefix_tokens, self.embedding_dim)}, got {tuple(prefixes.shape)}"
            )
        expected_tokens = (batch_size, expected_patches, self.embedding_dim)
        if tuple(patches.shape) != expected_tokens:
            raise ModelLoadError(
                f"Expected DINOv3 patch-token shape {expected_tokens}, got {tuple(patches.shape)}"
            )
        grid = self.input_size // self.patch_size
        output = patches.transpose(1, 2).reshape(batch_size, self.embedding_dim, grid, grid)
        result = output.float().cpu()
        if tuple(result.shape) != (batch_size, self.embedding_dim, grid, grid):
            raise ModelLoadError(f"Unexpected DINOv3 spatial feature shape: {tuple(result.shape)}")
        if not torch.isfinite(result).all():
            raise ModelLoadError("DINOv3 produced non-finite dense features")
        return result

    def normalized_zero_basis_features(self):
        black = Image.new("RGB", (self.input_size, self.input_size), color=(0, 0, 0))
        try:
            return self.extract_patch_maps([black])[0]
        finally:
            black.close()

    def close(self) -> None:
        self._model = None
        torch = self._torch
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
