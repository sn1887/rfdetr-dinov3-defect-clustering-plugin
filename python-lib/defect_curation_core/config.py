"""Structured Hydra/OmegaConf configuration and invariant validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir
except ImportError:  # Development fallback; production requirements include hydra-core.
    hydra_compose = None
    initialize_config_dir = None

from omegaconf import MISSING, DictConfig, OmegaConf

from defect_curation_core.errors import ConfigurationError
from defect_curation_core.hashing import sha256_bytes


@dataclass
class DetectorConfig:
    name: str = "rfdetr"
    checkpoint_path: str = MISSING
    expected_checkpoint_sha256: str | None = None
    model_class: str = "RFDETRSmall"
    package_compatibility: str = "==1.5.2"
    class_agnostic: bool = True
    threshold: float = 0.35
    max_detections_per_image: int = 20
    clip_to_image: bool = True
    drop_degenerate: bool = True
    safe_checkpoint_loading: bool = True
    device: str = "cuda"
    batch_size: int = 4
    oom_retries: int = 8
    preprocessing_spec: str = "rfdetr_package_default_v1"


@dataclass
class PositionalDebiasConfig:
    enabled: bool = True
    components: int = 20
    basis_image: str = "normalized_zero"


@dataclass
class PoolingConfig:
    method: str = "fractional_box_mean"
    min_effective_patch_weight: float = 4.0


@dataclass
class EmbeddingConfig:
    name: str = "dinov3_box"
    model_id: str = "timm/vit_base_patch16_dinov3.lvd1689m"
    timm_version: str = "1.0.20"
    artifact_path: str = MISSING
    expected_artifact_sha256: str | None = None
    artifact_revision: str = MISSING
    expected_prefix_tokens: int = 5
    input_size: int = 512
    patch_size: int = 16
    embedding_dim: int = 768
    box_padding_fraction: float = 0.15
    letterbox_fill: str = "imagenet_mean"
    layer: str = "final"
    patch_normalization: str = "l2"
    positional_debias: PositionalDebiasConfig = field(default_factory=PositionalDebiasConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    output_normalization: str = "l2"
    inference_precision: str = "bf16"
    persist_dtype: str = "float16"
    device: str = "cuda"
    batch_size: int = 16
    oom_retries: int = 8


@dataclass
class ClusteringConfig:
    name: str = "spherical_kmeans"
    k: int = MISSING
    backend: str = "faiss_cpu"
    spherical: bool = True
    niter: int = 25
    nredo: int = 5
    seed: int = 42
    fit_dtype: str = "float32"
    canonicalize_ids: bool = True
    max_points_per_centroid: int = 256
    assignment_chunk_size: int = 100_000
    max_empty_cluster_retries: int = 3


@dataclass
class RuntimeConfig:
    image_extensions: list[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png"])
    continue_on_image_error: bool = True
    max_failure_fraction: float = 0.10
    min_successful_instances: int = 2
    force_recompute: bool = False
    sort_paths: bool = True
    temporary_root: str | None = None
    write_checksums: bool = True
    input_read_chunk_size: int = 1_048_576


@dataclass
class PipelineConfig:
    schema_version: int = 1
    seed: int = 42
    pipeline_name: str = "rfdetr_dinov3_spherical_kmeans"
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def default_config_root() -> Path:
    return Path(__file__).resolve().parents[1] / "Configs"


def format_override(key: str, value: Any) -> str:
    """Produce a Hydra-safe dotlist override."""

    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif value is None:
        rendered = "null"
    elif isinstance(value, str):
        rendered = json.dumps(value)
    else:
        rendered = str(value)
    return f"{key}={rendered}"


def compose_config(
    overrides: Sequence[str],
    *,
    config_root: str | Path | None = None,
) -> DictConfig:
    root = Path(config_root) if config_root is not None else default_config_root()
    if not root.is_dir():
        raise ConfigurationError(f"Hydra config root does not exist: {root}")

    try:
        if hydra_compose is not None and initialize_config_dir is not None:
            with initialize_config_dir(config_dir=str(root.resolve()), version_base=None):
                composed = hydra_compose(config_name="config", overrides=list(overrides))
        else:
            # Offline developer/CI fallback. It preserves the same checked-in
            # Hydra config groups and dotlist override semantics, but production
            # environments are expected to install hydra-core as declared.
            root_document = OmegaConf.load(root / "config.yaml")
            defaults = list(root_document.get("defaults", []))
            root_document = OmegaConf.create(OmegaConf.to_container(root_document, resolve=False))
            if "defaults" in root_document:
                del root_document["defaults"]
            grouped: dict[str, Any] = {}
            for entry in defaults:
                if entry == "_self_":
                    continue
                if OmegaConf.is_config(entry):
                    entry = OmegaConf.to_container(entry, resolve=False)
                if not isinstance(entry, dict) or len(entry) != 1:
                    raise ConfigurationError(f"Unsupported config default entry in fallback composer: {entry!r}")
                group, name = next(iter(entry.items()))
                grouped[str(group)] = OmegaConf.load(root / str(group) / f"{name}.yaml")
            composed = OmegaConf.merge(
                OmegaConf.create(grouped),
                root_document,
                OmegaConf.from_dotlist(list(overrides)),
            )
        structured = OmegaConf.structured(PipelineConfig)
        cfg = OmegaConf.merge(structured, composed)
        OmegaConf.set_struct(cfg, True)
        OmegaConf.resolve(cfg)
    except ConfigurationError:
        raise
    except Exception as exc:  # Hydra/OmegaConf raises several configuration-specific types.
        raise ConfigurationError(f"Could not compose the pipeline configuration: {exc}") from exc

    validate_config(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


def validate_config(cfg: DictConfig) -> None:
    try:
        detector = cfg.detector
        embedding = cfg.embedding
        clustering = cfg.clustering
        runtime = cfg.runtime

        if cfg.schema_version != 1:
            raise ConfigurationError(f"Unsupported configuration schema version: {cfg.schema_version}")
        if not detector.class_agnostic:
            raise ConfigurationError("The MVP requires a class-agnostic RF-DETR checkpoint")
        if not detector.safe_checkpoint_loading:
            raise ConfigurationError("Unsafe checkpoint loading is not allowed")
        if str(detector.package_compatibility) != "==1.5.2":
            raise ConfigurationError("RF-DETR package compatibility must be exactly ==1.5.2")
        if str(detector.model_class).strip().lower() == "auto":
            raise ConfigurationError(
                "detector.model_class='auto' is unsupported for rfdetr 1.5.2; select a concrete qualified variant"
            )
        if not 0.0 < float(detector.threshold) < 1.0:
            raise ConfigurationError("detector.threshold must be strictly between 0 and 1")
        if not 1 <= int(detector.max_detections_per_image) <= 200:
            raise ConfigurationError("detector.max_detections_per_image must be in [1, 200]")
        if int(detector.batch_size) < 1 or int(embedding.batch_size) < 1:
            raise ConfigurationError("Inference batch sizes must be positive")
        if int(detector.oom_retries) < 0 or int(embedding.oom_retries) < 0:
            raise ConfigurationError("Inference OOM retry counts must be non-negative")
        if detector.device not in {"cuda", "cpu"} or embedding.device not in {"cuda", "cpu"}:
            raise ConfigurationError("Model devices must be 'cuda' or 'cpu'")
        if detector.device != embedding.device:
            raise ConfigurationError("RF-DETR and DINOv3 must use the same qualified device policy")
        if not detector.clip_to_image or not detector.drop_degenerate:
            raise ConfigurationError("RF-DETR postprocessing must clip boxes and drop degenerate boxes")

        for label, raw_path in (
            ("RF-DETR checkpoint", detector.checkpoint_path),
            ("DINOv3 local artifact", embedding.artifact_path),
        ):
            if not Path(str(raw_path)).is_absolute():
                raise ConfigurationError(f"{label} path must be absolute")

        if int(embedding.input_size) != 512 or int(embedding.patch_size) != 16:
            raise ConfigurationError("The locked MVP DINOv3 geometry is 512x512 with 16px patches")
        if int(embedding.embedding_dim) != 768:
            raise ConfigurationError("The locked ViT-B/16 MVP embedding dimension is 768")
        if not 0.0 <= float(embedding.box_padding_fraction) <= 0.5:
            raise ConfigurationError("embedding.box_padding_fraction must be in [0, 0.5]")
        if embedding.model_id != "timm/vit_base_patch16_dinov3.lvd1689m":
            raise ConfigurationError("The MVP locks the DINOv3 backbone to timm/vit_base_patch16_dinov3.lvd1689m")
        if embedding.timm_version != "1.0.20":
            raise ConfigurationError("The qualified DINOv3 loader requires timm==1.0.20")
        if not str(embedding.artifact_revision).strip():
            raise ConfigurationError("embedding.artifact_revision must identify the immutable model artifact")
        if int(embedding.expected_prefix_tokens) != 5:
            raise ConfigurationError("The qualified timm DINOv3 model has one class and four register tokens")
        if embedding.letterbox_fill != "imagenet_mean":
            raise ConfigurationError("The MVP letterbox fill is locked to imagenet_mean")
        if embedding.layer != "final":
            raise ConfigurationError("The MVP uses only final-layer DINOv3 patch tokens")
        if embedding.patch_normalization != "l2" or embedding.output_normalization != "l2":
            raise ConfigurationError("Patch and final embeddings must use L2 normalization")
        if embedding.pooling.method != "fractional_box_mean":
            raise ConfigurationError("The MVP pooling method is fractional_box_mean")
        if float(embedding.pooling.min_effective_patch_weight) <= 0.0:
            raise ConfigurationError("Minimum effective patch weight must be positive")
        if embedding.positional_debias.enabled:
            components = int(embedding.positional_debias.components)
            if components != 20:
                raise ConfigurationError("The release candidate locks positional debiasing to 20 components")
            if embedding.positional_debias.basis_image != "normalized_zero":
                raise ConfigurationError("The locked positional basis image is normalized_zero")
        if embedding.inference_precision not in {"bf16", "float32"}:
            raise ConfigurationError("embedding.inference_precision must be bf16 or float32")
        if embedding.device == "cpu" and embedding.inference_precision != "float32":
            raise ConfigurationError("CPU DINOv3 inference must use float32 precision")
        if embedding.persist_dtype != "float16":
            raise ConfigurationError("Persisted embeddings are locked to float16")

        if int(clustering.k) < 2:
            raise ConfigurationError("clustering.k must be at least 2")
        if clustering.backend not in {"faiss_cpu", "numpy_reference"}:
            raise ConfigurationError("Unsupported clustering backend")
        if not clustering.spherical:
            raise ConfigurationError("The MVP requires spherical k-means")
        if int(clustering.niter) < 1 or int(clustering.nredo) < 1:
            raise ConfigurationError("K-means iterations and restarts must be positive")
        if clustering.fit_dtype != "float32":
            raise ConfigurationError("K-means fitting is locked to float32")
        if not clustering.canonicalize_ids:
            raise ConfigurationError("Cluster ID canonicalization must remain enabled")
        if int(clustering.max_points_per_centroid) < 1:
            raise ConfigurationError("max_points_per_centroid must be positive")
        if int(clustering.assignment_chunk_size) < 1:
            raise ConfigurationError("assignment_chunk_size must be positive")
        if int(clustering.max_empty_cluster_retries) < 0:
            raise ConfigurationError("max_empty_cluster_retries must be non-negative")

        if not 0.0 <= float(runtime.max_failure_fraction) <= 1.0:
            raise ConfigurationError("runtime.max_failure_fraction must be in [0, 1]")
        if int(runtime.min_successful_instances) < 2:
            raise ConfigurationError("runtime.min_successful_instances must be at least 2")
        extensions = [str(item) for item in runtime.image_extensions]
        if extensions != [".jpg", ".jpeg", ".png"]:
            raise ConfigurationError("The MVP image extensions are locked to .jpg, .jpeg, and .png")
        if not runtime.continue_on_image_error or not runtime.sort_paths or not runtime.write_checksums:
            raise ConfigurationError(
                "The MVP requires row-level continuation, deterministic path sorting, and checksums"
            )
        if int(runtime.input_read_chunk_size) < 1:
            raise ConfigurationError("runtime.input_read_chunk_size must be positive")
        if runtime.temporary_root is not None and not Path(str(runtime.temporary_root)).is_absolute():
            raise ConfigurationError("runtime.temporary_root must be absolute when set")
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Invalid resolved configuration: {exc}") from exc


_REDACTED_PATHS: tuple[tuple[str, str, str], ...] = (
    ("detector", "checkpoint_path", "<redacted-model-path>"),
    ("embedding", "artifact_path", "<redacted-model-path-or-snapshot>"),
    ("runtime", "temporary_root", "<redacted-temporary-root>"),
)


def config_as_dict(cfg: DictConfig, *, redact_paths: bool = False) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert isinstance(value, dict)
    if redact_paths:
        for section, key, replacement in _REDACTED_PATHS:
            section_value = value.get(section)
            if isinstance(section_value, dict) and section_value.get(key) is not None:
                section_value[key] = replacement
    return value


def resolved_yaml(cfg: DictConfig, *, redact_paths: bool = False) -> str:
    if not redact_paths:
        return OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    return OmegaConf.to_yaml(OmegaConf.create(config_as_dict(cfg, redact_paths=True)), sort_keys=True)


def config_hash(cfg: DictConfig) -> str:
    # Model content and code identity are recorded separately by immutable hashes/revisions.
    # Excluding deployment filesystem paths makes this semantic config hash portable and
    # prevents sensitive paths from leaking into artifacts.
    return sha256_bytes(resolved_yaml(cfg, redact_paths=True).encode("utf-8"))
