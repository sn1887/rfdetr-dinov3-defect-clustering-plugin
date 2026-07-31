"""Translate Dataiku plugin/recipe parameters into validated Hydra overrides."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from defect_curation_core.config import compose_config, format_override
from defect_curation_core.errors import ConfigurationError, DependencyError


def _required(settings: Mapping[str, Any], name: str) -> Any:
    value = settings.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ConfigurationError(f"Missing required plugin setting: {name}")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_device(policy: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise DependencyError("PyTorch is not installed in the selected code environment") from exc

    cuda_available = bool(torch.cuda.is_available())
    if policy == "cuda_required":
        if not cuda_available:
            raise ConfigurationError(
                "The plugin device policy requires CUDA, but PyTorch reports no available CUDA device"
            )
        return "cuda"
    if policy == "cpu_allowed":
        return "cuda" if cuda_available else "cpu"
    raise ConfigurationError(f"Unknown device policy: {policy!r}")


def build_pipeline_config(
    *,
    plugin_config: Mapping[str, Any],
    recipe_config: Mapping[str, Any],
    config_root: str | Path | None = None,
):
    device = resolve_device(str(plugin_config.get("device_policy", "cuda_required")))
    temporary_root = _optional_text(plugin_config.get("temporary_root"))
    if temporary_root is not None:
        root = Path(temporary_root)
        if not root.is_absolute():
            raise ConfigurationError("temporary_root must be an absolute path")
        if not root.is_dir():
            raise ConfigurationError(f"temporary_root does not exist or is not a directory: {root}")

    expected_checkpoint = _optional_text(plugin_config.get("rfdetr_checkpoint_sha256"))
    expected_artifact = _optional_text(plugin_config.get("dinov3_artifact_sha256"))

    overrides = [
        format_override("detector.checkpoint_path", str(_required(plugin_config, "rfdetr_checkpoint_path"))),
        format_override("detector.expected_checkpoint_sha256", expected_checkpoint),
        format_override("detector.model_class", str(_required(plugin_config, "rfdetr_model_class"))),
        format_override(
            "detector.package_compatibility",
            str(_required(plugin_config, "rfdetr_package_compatibility")),
        ),
        format_override("detector.device", device),
        format_override("embedding.model_id", str(_required(plugin_config, "dinov3_model_id"))),
        format_override("embedding.timm_version", str(_required(plugin_config, "dinov3_timm_version"))),
        format_override("embedding.artifact_path", str(_required(plugin_config, "dinov3_artifact_path"))),
        format_override(
            "embedding.artifact_revision", str(_required(plugin_config, "dinov3_artifact_revision"))
        ),
        format_override("embedding.expected_artifact_sha256", expected_artifact),
        format_override("embedding.device", device),
        format_override("embedding.inference_precision", "bf16" if device == "cuda" else "float32"),
        format_override("clustering.k", int(_required(recipe_config, "cluster_count"))),
        format_override(
            "detector.threshold", float(recipe_config.get("detection_threshold", 0.35))
        ),
        format_override(
            "embedding.box_padding_fraction",
            float(recipe_config.get("box_padding_fraction", 0.15)),
        ),
        format_override(
            "detector.max_detections_per_image",
            int(recipe_config.get("max_detections_per_image", 20)),
        ),
        format_override("runtime.force_recompute", bool(recipe_config.get("force_recompute", False))),
        format_override(
            "runtime.max_failure_fraction",
            float(plugin_config.get("max_failure_fraction", 0.10)),
        ),
        format_override("runtime.temporary_root", temporary_root),
    ]
    return compose_config(overrides, config_root=config_root)
