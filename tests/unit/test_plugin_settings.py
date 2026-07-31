from __future__ import annotations

from pathlib import Path

import pytest
from defect_curation_core.errors import ConfigurationError
from defect_curation_plugin.plugin_settings import build_pipeline_config, resolve_device


def settings(tmp_path: Path):
    return {
        "rfdetr_checkpoint_path": str((tmp_path / "rfdetr.pth").resolve()),
        "rfdetr_checkpoint_sha256": "",
        "rfdetr_model_class": "RFDETRSmall",
        "rfdetr_package_compatibility": "==1.5.2",
        "dinov3_model_id": "timm/vit_base_patch16_dinov3.lvd1689m",
        "dinov3_timm_version": "1.0.20",
        "dinov3_artifact_path": str((tmp_path / "model.safetensors").resolve()),
        "dinov3_artifact_revision": "deadbeef",
        "dinov3_artifact_sha256": "",
        "device_policy": "cpu_allowed",
        "temporary_root": "",
        "max_failure_fraction": 0.2,
    }


def test_plugin_settings_map_to_locked_config(tmp_path: Path) -> None:
    cfg = build_pipeline_config(
        plugin_config=settings(tmp_path),
        recipe_config={
            "cluster_count": 7,
            "detection_threshold": 0.2,
            "box_padding_fraction": 0.1,
            "max_detections_per_image": 30,
            "force_recompute": True,
        },
    )
    assert cfg.clustering.k == 7
    assert cfg.detector.threshold == 0.2
    assert cfg.embedding.box_padding_fraction == 0.1
    assert cfg.runtime.force_recompute is True
    assert cfg.embedding.device == "cpu"
    assert cfg.embedding.inference_precision == "float32"


def test_cuda_required_fails_when_cuda_is_unavailable() -> None:
    import torch

    if torch.cuda.is_available():
        pytest.skip("This assertion is specific to CPU-only CI")
    with pytest.raises(ConfigurationError):
        resolve_device("cuda_required")
