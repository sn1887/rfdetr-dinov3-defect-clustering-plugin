from __future__ import annotations

from pathlib import Path

import pytest
from defect_curation_core.config import compose_config, config_hash, format_override, resolved_yaml
from defect_curation_core.errors import ConfigurationError
from omegaconf import OmegaConf


def valid_overrides(tmp_path: Path) -> list[str]:
    return [
        format_override("detector.checkpoint_path", str((tmp_path / "rfdetr.pth").resolve())),
        format_override("detector.package_compatibility", "==1.5.2"),
        format_override("detector.device", "cpu"),
        format_override("embedding.artifact_path", str((tmp_path / "model.safetensors").resolve())),
        format_override("embedding.artifact_revision", "deadbeef"),
        format_override("embedding.device", "cpu"),
        format_override("embedding.inference_precision", "float32"),
        format_override("clustering.k", 4),
    ]


def test_compose_config_applies_overrides_and_is_read_only(tmp_path: Path) -> None:
    cfg = compose_config(valid_overrides(tmp_path))
    assert cfg.clustering.k == 4
    assert cfg.clustering.seed == cfg.seed == 42
    assert cfg.detector.device == cfg.embedding.device == "cpu"
    assert len(config_hash(cfg)) == 64
    assert OmegaConf.is_readonly(cfg)


def test_unknown_override_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        compose_config([*valid_overrides(tmp_path), "unknown.setting=1"])


def test_invalid_runtime_value_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        compose_config([*valid_overrides(tmp_path), "runtime.max_failure_fraction=1.2"])


@pytest.mark.parametrize(
    "override",
    ["detector.model_class=auto", "detector.package_compatibility=>=1.5.2"],
)
def test_rfdetr_1_5_2_requires_exact_version_and_concrete_variant(
    tmp_path: Path, override: str
) -> None:
    with pytest.raises(ConfigurationError):
        compose_config([*valid_overrides(tmp_path), override])


def test_persisted_config_redacts_deployment_paths(tmp_path: Path) -> None:
    cfg = compose_config(valid_overrides(tmp_path))
    rendered = resolved_yaml(cfg, redact_paths=True)
    assert str(tmp_path) not in rendered
    assert "<redacted-model-path>" in rendered
    assert "<redacted-model-path-or-snapshot>" in rendered
    assert "<redacted-model-path>" in rendered


def test_config_hash_does_not_depend_on_host_specific_model_paths(tmp_path: Path) -> None:
    first = compose_config(valid_overrides(tmp_path / "one"))
    second = compose_config(valid_overrides(tmp_path / "two"))
    assert config_hash(first) == config_hash(second)


@pytest.mark.parametrize(
    "override",
    [
        "embedding.input_size=768",
        "embedding.patch_size=14",
        "embedding.granularity=tile",
        "embedding.positional_debias.components=64",
        "detector.class_agnostic=false",
        "clustering.canonicalize_ids=false",
        "runtime.sort_paths=false",
        "runtime.write_checksums=false",
        "runtime.input_read_chunk_size=0",
    ],
)
def test_locked_mvp_invariants_are_rejected(tmp_path: Path, override: str) -> None:
    with pytest.raises(ConfigurationError):
        compose_config([*valid_overrides(tmp_path), override])
